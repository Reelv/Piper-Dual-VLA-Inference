
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CURRENT_FILE = Path(__file__).resolve()
PARENT_DIR = CURRENT_FILE.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))


def _load_deploy_config():
    config_path = PARENT_DIR / "deploy_policy.yml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_checkpoint_path(model_path):
    if not model_path:
        return None, None, None
    parts = Path(model_path).resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if len(parts) > idx + 3:
            return parts[idx + 1], parts[idx + 2], parts[idx + 3]
    return None, None, None


def _resolve_checkpoint_dir(model_path, train_config_name, model_name, checkpoint_id):
    if model_path and str(model_path).lower() != "none":
        return model_path
    if not train_config_name or not model_name or checkpoint_id is None:
        raise ValueError("train_config_name, model_name, checkpoint_id required for pi05")
    return os.path.join(
        "policy",
        "pi05",
        "checkpoints",
        str(train_config_name),
        str(model_name),
        str(checkpoint_id),
    )


def _get_assets_id(checkpoint_dir):
    assets_dir = os.path.join(checkpoint_dir, "assets")
    entries = os.listdir(assets_dir)
    if not entries:
        raise ValueError(f"no assets found under {assets_dir}")
    return entries[0]


class PI05_DUAL:
    def __init__(self, model_path, task_name):
        self.task_name = task_name

        deploy_config = _load_deploy_config()
        train_config_name = deploy_config.get("train_config_name")
        model_name = deploy_config.get("model_name")
        checkpoint_id = deploy_config.get("checkpoint_id")
        self.pi0_step = int(deploy_config.get("pi0_step", 20))

        if not train_config_name or not model_name or checkpoint_id is None:
            parsed_train, parsed_model, parsed_ckpt = _parse_checkpoint_path(model_path)
            train_config_name = train_config_name or parsed_train
            model_name = model_name or parsed_model
            checkpoint_id = checkpoint_id if checkpoint_id is not None else parsed_ckpt

        if not train_config_name:
            raise ValueError("train_config_name is required for pi05")

        checkpoint_dir = _resolve_checkpoint_dir(
            model_path,
            train_config_name,
            model_name,
            checkpoint_id,
        )
        assets_id = _get_assets_id(checkpoint_dir)

        config = _config.get_config(train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id,
            sample_kwargs={"num_steps": 20},
        )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.random_set_language()
        #lzr 6.1 增加动作平滑
        self._prev_raw_action = None   # 上一帧实际执行的 action
        self._ema_alpha = 0.3         # 平滑系数，0~1，越小越平滑
        #lzr 6.1 增加动作插值过渡
        self._last_chunk_tail = None
        self._last_chunk_vel  = None           # 新增：记录速度
        self._blend_steps = 10                  # 不需要太多步

    def set_img_size(self, img_size):
        self.img_size = img_size

    def random_set_language(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(root_dir, "task_instructions", f"{self.task_name}.json")
        with open(json_path, "r", encoding="utf-8") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict.get("instructions", [])
        if not instructions:
            raise ValueError(f"no instructions found in {json_path}")
        self.instruction = np.random.choice(instructions)
        print(f"successfully set instruction:{self.instruction}")

    def update_observation_window(self, img_arr, state):
        imgs_array = []

        if isinstance(img_arr[0], bytes):
            for data in img_arr:
                jpeg_bytes = np.array(data).tobytes().rstrip(b"\0")
                nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                imgs_array.append(cv2.imdecode(nparr, 1))
        else:
            imgs_array = img_arr

        img_front, img_right, img_left = imgs_array[0], imgs_array[1], imgs_array[2]
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        actions = self.policy.infer(self.observation_window)["actions"]
        return actions[: self.pi0_step]
    #lzr 6.1
    def smooth_action(self, raw_action):
        if self._prev_raw_action is None:
            self._prev_raw_action = raw_action
            return raw_action
        smoothed = self._ema_alpha * raw_action + (1 - self._ema_alpha) * self._prev_raw_action
        self._prev_raw_action = smoothed
        return smoothed
    def blend_with_prev_chunk(self, action_chunk):
        if self._last_chunk_tail is None:
            self._last_chunk_tail = action_chunk[-1].copy()
            return action_chunk
        n = min(self._blend_steps, len(action_chunk))
        for i in range(n):
            t = (i + 1) / (n + 1)  # 逻辑不变，但 n 更小更合理
            action_chunk[i] = (1 - t) * self._last_chunk_tail + t * action_chunk[i]
        self._last_chunk_tail = action_chunk[-1].copy()
        return action_chunk
    def blend_with_prev_chunk_bezier(self, action_chunk):
        """
        用三次贝塞尔曲线在 chunk 边界做过渡
        保证位置和速度都连续
        """
        if self._last_chunk_tail is None or self._last_chunk_vel is None:
            # 第一个 chunk，记录尾部状态
            self._last_chunk_tail = action_chunk[-1].copy()
            self._last_chunk_vel = action_chunk[-1] - action_chunk[-2] \
                if len(action_chunk) > 1 else np.zeros_like(action_chunk[-1])
            return action_chunk

        n = self._blend_steps
        p0 = self._last_chunk_tail          # 起点：上一chunk最后一步
        v0 = self._last_chunk_vel           # 起点速度
        p3 = action_chunk[min(n, len(action_chunk)-1)].copy()  # 终点
        v3 = action_chunk[1] - action_chunk[0] \
            if len(action_chunk) > 1 else np.zeros_like(p3)    # 终点速度

        # 控制点：由速度方向决定，scale 控制"拉伸程度"
        scale = n / 3.0
        p1 = p0 + scale * v0               # 控制点1
        p2 = p3 - scale * v3               # 控制点2

        for i in range(min(n, len(action_chunk))):
            t = (i + 1) / (n + 1)          # t ∈ (0, 1)
            # 三次贝塞尔公式
            mt = 1 - t
            bezier = (mt**3 * p0
                    + 3 * mt**2 * t * p1
                    + 3 * mt * t**2 * p2
                    + t**3 * p3)
            action_chunk[i] = bezier

        # 更新尾部状态
        self._last_chunk_tail = action_chunk[-1].copy()
        self._last_chunk_vel = action_chunk[-1] - action_chunk[-2] \
            if len(action_chunk) > 1 else np.zeros_like(action_chunk[-1])

        return action_chunk
    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        #lzr 6.1 增加重置动作平滑状态
        self._prev_raw_action = None
        self._last_chunk_tail = None
        print("successfully unset obs and language instruction")


def _get_arm_state(arm_state):
    if "joint" in arm_state:
        joint = np.array(arm_state["joint"]).reshape(-1)
    elif "qpos" in arm_state:
        joint = np.array(arm_state["qpos"]).reshape(-1)
    else:
        joint = np.zeros(6, dtype=np.float32)

    gripper = np.array(arm_state.get("gripper", 0.0)).reshape(-1)
    return joint, gripper


def input_transform(data):
    left_joint, left_gripper = _get_arm_state(data[0]["left_arm"])
    right_joint, right_gripper = _get_arm_state(data[0]["right_arm"])

    state = np.concatenate([
        left_joint,
        left_gripper,
        right_joint,
        right_gripper,
    ])

    img_arr = (
        data[1]["cam_head"]["color"],
        data[1]["cam_right_wrist"]["color"],
        data[1]["cam_left_wrist"]["color"],
    )
    return img_arr, state


def output_transform(action):
    left_joint = action[:6]
    left_gripper = action[6]
    right_joint = action[7:13]
    right_gripper = action[13]

    # GRIPPER_SCALE = 1.0 / 0.08  # = 12.5
    # left_gripper = float(left_gripper) * GRIPPER_SCALE
    # right_gripper = float(right_gripper) * GRIPPER_SCALE


    # if left_gripper <= 0.7:
    #     left_gripper = 0.0
    # else:  
    #     left_gripper = 1.0
    # if right_gripper < 0.7:
    #     right_gripper = 0.0
    # else:
    #     right_gripper = 1.0

    print(f"left_gripper: {left_gripper}, right_gripper: {right_gripper}")
    
    move_data = {
        "arm": {
            "left_arm": {
                "joint": left_joint,
                "gripper": left_gripper,
            },
            "right_arm": {
                "joint": right_joint,
                "gripper": right_gripper,
            },
        }
    }
    return move_data


PI0_DUAL = PI05_DUAL


# ═══════════════════════════════════════════════════════════════════════════════
# RTC (Real-Time Chunking) 扩展
# 参考: "Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025)
# 来自 kai0 项目的 train_deploy_alignment 模块
# ═══════════════════════════════════════════════════════════════════════════════

class StreamActionBuffer:
    """
    Action chunk 缓冲区, 用于 RTC 推理中的时间平滑.
    对相邻 chunk 的重叠部分做线性混合 (100% old → 0% new).

    来自 kai0/train_deploy_alignment/inference/ 的实现,
    这里作为独立工具类提供, 供 deploy_pi05_real_acc_rtc.py 使用.
    """
    def __init__(self, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        import threading
        from collections import deque
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()
        self.k = 0
        self.last_action = None

    def integrate_new_chunk(self, actions_chunk, max_k=0, min_m=8):
        import numpy as np
        from collections import deque
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            if str(self.smooth_method).lower() == "raw":
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return

            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy()
                            for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float)
                 + w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self):
        import numpy as np
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def clear(self):
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0