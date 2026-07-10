#!/usr/bin/env python3
"""
自适应 Action Chunking 推理脚本 (AAC: Adaptive Action Chunking).

参考论文: "Adaptive Action Chunking for Real-Time Edge VLA Control"
核心思想：根据当前场景的动态程度，动态调整每次推理的 action chunk 大小 k。

策略:
- 场景变化剧烈（chunk 边界动作跳变大） → 减小 k，提高推理频率，增强响应能力
- 场景变化平缓（chunk 边界动作连续）   → 增大 k，降低推理频率，动作更平滑

同时保留了:
- Bezier 贝塞尔曲线过渡（保证 chunk 间位置 + 速度连续）
- EMA 指数平滑（逐帧消除高频抖动）


配置文件是 deploy_pi05_real.yml，命令行参数覆盖配置文件参数。
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from robot.robot.base_robot_node import build_robot_node
from robot.utils.base.data_handler import debug_print, is_enter_pressed
from robot.policy.pi05.inference_model import PI05_DUAL, input_transform, output_transform


# ── 自适应 Chunking 参数 ──────────────────────────────────────────────────────
# 这些参数控制 k 的动态调整行为，可在命令行或 yaml 中覆盖
AAC_DEFAULTS = {
    "k_min": 50,            # 最小 chunk 大小（场景剧烈变化时）
    "k_max": 100,           # 最大 chunk 大小（场景平稳时）
    "k_init": 50,          # 初始 chunk 大小
    "delta_high": 0.03,    # 动作跳变阈值：超过此值认为场景变化剧烈
    "delta_low": 0.005,     # 动作跳变阈值：低于此值认为场景平稳
    "k_step_up": 5,        # 每次增大 k 的步长
    "k_step_down": 5,     # 每次减小 k 的步长（减小更快，安全优先）
    "ema_alpha": 0.3,      # 逐帧 EMA 平滑系数
    "blend_steps": 30,     # Bezier 过渡步数
}


OUTPUT_VIDEO_DIR = str(Path(__file__).with_name("output_video"))
DEFAULT_FPS = 30
CAMERA_KEYS = ("cam_head", "cam_right_wrist", "cam_left_wrist")


# ═══════════════════════════════════════════════════════════════════════════════
# FFmpeg 管道写入器
# ═══════════════════════════════════════════════════════════════════════════════
class FFmpegWriter:
    def __init__(self, path: str, width: int, height: int, fps: int = DEFAULT_FPS):
        self.path = path
        cmd = [
            "ffmpeg",
            "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-vcodec", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        print(f"[FFmpegWriter] saved → {self.path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 自适应 k-Selector
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveKSelector:
    """
    基于动作连续性的启发式 k-Selector。

    灵感来源：AAC 的 oracle labeling 策略 ——
    回看未来几步 action 的方差，方差小 → 大 k，方差大 → 小 k。

    在线版本无法回看"未来"，所以用**跨 chunk 边界动作跳变**作为代理信号：
    - 新 chunk 第 1 步 vs 旧 chunk 最后 1 步的差异 → 反推出场景动态程度
    """

    def __init__(
        self,
        k_min: int = 3,
        k_max: int = 50,
        k_init: int = 10,
        delta_high: float = 0.05,
        delta_low: float = 0.01,
        k_step_up: int = 5,
        k_step_down: int = 10,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.k_current = k_init
        self.delta_high = delta_high
        self.delta_low = delta_low
        self.k_step_up = k_step_up
        self.k_step_down = k_step_down

        # 存储上一个 chunk 的尾部信息，用于计算跨 chunk 跳变
        self._prev_chunk_tail_action = None
        self._prev_chunk_second_last = None  # 用于估算速度方向

    def reset(self):
        self._prev_chunk_tail_action = None
        self._prev_chunk_second_last = None

    def get_k(self) -> int:
        """返回当前 k 值（推理前调用）"""
        return self.k_current

    def update(self, action_chunk: np.ndarray):
        """
        根据当前 chunk 与上一 chunk 的边界跳变程度，更新 k。

        调用时机：每次推理拿到新 action_chunk 后立即调用。
        """
        if len(action_chunk) < 2:
            return

        if self._prev_chunk_tail_action is not None:
            # 计算跨 chunk 边界跳变
            # delta = ||new_chunk[0] - old_chunk[-1]||
            delta = float(np.linalg.norm(action_chunk[0] - self._prev_chunk_tail_action))

            # AAC 启发式：跳变大 → 场景动态 → 减小 k；跳变小 → 场景平稳 → 增大 k
            if delta > self.delta_high:
                # 场景剧烈变化，快速减小 k 以提高响应
                self.k_current = max(self.k_min, self.k_current - self.k_step_down)
                debug_print("AAC", f"Δ={delta:.4f} > high={self.delta_high} → k↓ = {self.k_current}", "INFO")
            elif delta < self.delta_low:
                # 场景平稳，增大 k 以减少推理开销、增加平滑度
                self.k_current = min(self.k_max, self.k_current + self.k_step_up)
                debug_print("AAC", f"Δ={delta:.4f} < low={self.delta_low} → k↑ = {self.k_current}", "INFO")
            else:
                debug_print("AAC", f"Δ={delta:.4f} ∈ [{self.delta_low}, {self.delta_high}] → k = {self.k_current} (hold)", "INFO")

        # 更新尾部状态（用于下一次跨 chunk 比较）
        self._prev_chunk_tail_action = action_chunk[-1].copy()
        self._prev_chunk_second_last = action_chunk[-2].copy() if len(action_chunk) > 1 else None


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════
def _get_class(import_name, class_name):
    module = importlib.import_module(import_name)
    return getattr(module, class_name)


def _load_yaml(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}


def _normalize_value(value):
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Action Chunking 推理")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--robot_name", type=str, default=None)
    parser.add_argument("--robot_class", type=str, default=None)
    parser.add_argument("--episode_num", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--control_dt", type=float, default=None)
    parser.add_argument("--video", type=str, default=None, help="cam name, e.g. cam_head")
    parser.add_argument("--node", action="store_true")

    # ── AAC 特有参数 ──
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_init", type=int, default=None)
    parser.add_argument("--delta_high", type=float, default=None)
    parser.add_argument("--delta_low", type=float, default=None)
    parser.add_argument("--k_step_up", type=int, default=None)
    parser.add_argument("--k_step_down", type=int, default=None)
    parser.add_argument("--ema_alpha", type=float, default=None)
    parser.add_argument("--blend_steps", type=int, default=None)

    args = parser.parse_args()

    config_path = (
        Path(args.config) if args.config else Path(__file__).with_name("deploy_pi05_real.yml")
    )
    config = _load_yaml(config_path)

    merged = {}
    merged.update(config)
    for key in (
        "model_path", "task_name", "robot_name", "robot_class",
        "episode_num", "max_step", "control_dt", "video",
        # AAC params
        "k_min", "k_max", "k_init", "delta_high", "delta_low",
        "k_step_up", "k_step_down", "ema_alpha", "blend_steps",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value

    merged["node"] = bool(args.node or merged.get("node", False))
    for key in list(merged.keys()):
        merged[key] = _normalize_value(merged[key])

    # 默认值
    merged.setdefault("episode_num", 1)
    merged.setdefault("max_step", 1000000)
    merged.setdefault("control_dt", 1 / 30)
    merged.setdefault("video", None)
    for k, v in AAC_DEFAULTS.items():
        merged.setdefault(k, v)

    missing = [
        key for key in ("model_path", "task_name", "robot_name", "robot_class")
        if not merged.get(key)
    ]
    if missing:
        raise SystemExit(f"Missing required config keys: {', '.join(missing)}")

    return argparse.Namespace(**merged)


# ═══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    os.environ.setdefault("INFO_LEVEL", "INFO")
    args = parse_args()

    # ── 初始化模型 ──
    model = PI05_DUAL(args.model_path, args.task_name)
    # 注入 AAC 参数到模型
    model._ema_alpha = args.ema_alpha
    model._blend_steps = args.blend_steps

    # ── 初始化自适应 k-Selector ──
    k_selector = AdaptiveKSelector(
        k_min=args.k_min,
        k_max=args.k_max,
        k_init=args.k_init,
        delta_high=args.delta_high,
        delta_low=args.delta_low,
        k_step_up=args.k_step_up,
        k_step_down=args.k_step_down,
    )

    # ── 初始化机器人 ──
    robot_class = _get_class(f"my_robot.{args.robot_name}", args.robot_class)
    if args.node:
        robot_class = build_robot_node(robot_class)
    robot = robot_class()
    robot.set_up()

    for episode in range(args.episode_num):
        step = 0
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()
        k_selector.reset()

        # ── 初始化 FFmpeg 写入器 ──
        video_writers: dict[str, FFmpegWriter] = {}
        if args.video is not None:
            _, cam_data = robot.get()
            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
            for cam_key in CAMERA_KEYS:
                first_frame = cam_data[cam_key]["color"][:, :, ::-1]
                height, width = first_frame.shape[:2]
                video_path = os.path.join(
                    OUTPUT_VIDEO_DIR, f"episode_{episode}_{cam_key}.mp4"
                )
                video_writers[cam_key] = FFmpegWriter(video_path, width, height, DEFAULT_FPS)
                print(f"Video saving enabled: {video_path}")

        # ── 等待启动 ──
        print(f"当前推理 Instruction: {model.instruction}")
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("start to inference (AAC mode), press ENTER to end...")
            else:
                print("waiting for start command, press ENTER to start...")
                time.sleep(1)

        # ═════════════════════════════════════════════════════════════════════
        # 推理主循环 — AAC 模式
        # ═════════════════════════════════════════════════════════════════════
        chunk_round = 0
        while step < args.max_step and is_start:
            # 1. 获取观测
            data = robot.get()
            img_arr, state = input_transform(data)

            # 2. 获取当前自适应 k 值，设置给模型
            current_k = k_selector.get_k()
            model.pi0_step = current_k

            # 3. 推理
            model.update_observation_window(img_arr, state)
            action_chunk = model.get_action()       # 返回 current_k 步 action

            # 4. AAC 核心：根据边界跳变更新 k（下次推理生效）
            k_selector.update(action_chunk)

            # 5. Bezier 贝塞尔过渡（保证 chunk 间位置+速度连续）
            action_chunk = model.blend_with_prev_chunk_bezier(action_chunk)

            # 6. 逐帧执行，每帧做 EMA 平滑
            for action in action_chunk:
                smoothed_action = model.smooth_action(action)

                # 录像
                if video_writers:
                    _, cam_data = robot.get()
                    for cam_key, writer in video_writers.items():
                        frame = cam_data[cam_key]["color"]
                        writer.write(frame)

                if step % 50 == 0:
                    debug_print("main",
                                f"step: {step}/{args.max_step} | chunk_round: {chunk_round} | k: {current_k}",
                                "INFO")

                move_data = output_transform(smoothed_action)
                robot.move(move_data)
                step += 1

                time.sleep(args.control_dt)

                if step >= args.max_step or is_enter_pressed():
                    debug_print("main", "enter pressed, the episode end", "INFO")
                    is_start = False
                    break

            chunk_round += 1

        # ── 收尾 ──
        for writer in video_writers.values():
            writer.release()
        debug_print("main", f"finish episode {episode}, running steps {step}", "INFO")


if __name__ == "__main__":
    main()
