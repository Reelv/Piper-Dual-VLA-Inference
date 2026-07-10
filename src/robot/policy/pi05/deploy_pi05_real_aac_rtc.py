#!/usr/bin/env python3
"""
AAC + RTC 联合推理脚本.

AAC (Adaptive Action Chunking):
  参考: "Adaptive Action Chunking for Real-Time Edge VLA Control"
  - AdaptiveKSelector: 根据跨 chunk 动作跳变动态调整 k
  - Bezier 贝塞尔曲线过渡: 保证 chunk 间位置 + 速度连续

RTC (Real-Time Chunking):
  参考: "Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025)
  来自 kai0 项目 train_deploy_alignment 模块
  - StreamActionBuffer: 相邻 chunk 重叠部分线性混合 (100% old → 0% new)
  - 异步推理: 推理线程与执行线程分离
  - prev_chunk 引导 + 延迟估计: RTC payload 携带已执行前缀

两者结合:
  ┌──────────────────────────────────────────────────────┐
  │  推理线程 (异步, 频率 ~inference_rate Hz)              │
  │  1. 获取观测                                          │
  │  2. AdaptiveKSelector.get_k() → 动态 k                │
  │  3. model.get_action() → action_chunk [k步]            │
  │  4. k_selector.update(chunk) → 更新 k                 │
  │  5. Bezier 贝塞尔过渡 (chunk 间位置+速度连续)           │
  │  6. StreamActionBuffer.integrate → RTC 时间平滑        │
  │  7. 更新 RTC prev_chunk + 延迟估计                     │
  └──────────────────────┬───────────────────────────────┘
                         │ StreamActionBuffer
  ┌──────────────────────▼───────────────────────────────┐
  │  执行线程 (频率 ~1/control_dt Hz)                     │
  │  1. buf.pop_next_action() → 逐帧取动作                 │
  │  2. EMA 逐帧平滑                                      │
  │  3. robot.move()                                      │
  └──────────────────────────────────────────────────────┘

Usage:
  python deploy_pi05_real_aac_rtc.py
  python deploy_pi05_real_aac_rtc.py --config my_config.yml
"""

import argparse
import importlib
import os
import subprocess
import sys
import threading
import time
from collections import deque
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


OUTPUT_VIDEO_DIR = str(Path(__file__).with_name("output_video"))
DEFAULT_FPS = 30
CAMERA_KEYS = ("cam_head", "cam_right_wrist", "cam_left_wrist")


# ═══════════════════════════════════════════════════════════════════════════════
# 默认参数
# ═══════════════════════════════════════════════════════════════════════════════
AAC_DEFAULTS = {
    "k_min": 30,
    "k_max": 100,
    "k_init": 50,
    "delta_high": 0.03,
    "delta_low": 0.005,
    "k_step_up": 5,
    "k_step_down": 10,
    "blend_steps": 30,
}

RTC_DEFAULTS = {
    "inference_rate": 4,
    "smooth_method": "temporal",
    "min_smooth_steps": 8,
    "latency_k": 0,
    "decay_alpha": 0.25,
    "enable_rtc": True,
    "mask_prefix_delay": False,
    "max_guidance_weight": 0.5,
    "ema_alpha": 0.4,
}


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
# AdaptiveKSelector — AAC 自适应 k-Selector
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveKSelector:
    """基于跨 chunk 动作跳变的自适应 k-Selector."""

    def __init__(self, k_min=3, k_max=50, k_init=10,
                 delta_high=0.05, delta_low=0.01,
                 k_step_up=5, k_step_down=10):
        self.k_min = k_min
        self.k_max = k_max
        self.k_current = k_init
        self.delta_high = delta_high
        self.delta_low = delta_low
        self.k_step_up = k_step_up
        self.k_step_down = k_step_down
        self._prev_chunk_tail_action = None

    def reset(self):
        self._prev_chunk_tail_action = None

    def get_k(self) -> int:
        return self.k_current

    def update(self, action_chunk: np.ndarray):
        if len(action_chunk) < 2:
            return
        if self._prev_chunk_tail_action is not None:
            delta = float(np.linalg.norm(action_chunk[0] - self._prev_chunk_tail_action))
            if delta > self.delta_high:
                self.k_current = max(self.k_min, self.k_current - self.k_step_down)
                debug_print("AAC", f"Δ={delta:.4f} > high={self.delta_high} → k↓ = {self.k_current}", "INFO")
            elif delta < self.delta_low:
                self.k_current = min(self.k_max, self.k_current + self.k_step_up)
                debug_print("AAC", f"Δ={delta:.4f} < low={self.delta_low} → k↑ = {self.k_current}", "INFO")
            else:
                debug_print("AAC", f"Δ={delta:.4f} ∈ [{self.delta_low}, {self.delta_high}] → k = {self.k_current} (hold)", "INFO")
        self._prev_chunk_tail_action = action_chunk[-1].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# StreamActionBuffer — RTC chunk 队列 + 时间平滑
# ═══════════════════════════════════════════════════════════════════════════════
class StreamActionBuffer:
    """
    Action chunk 缓冲区:
      - latency_k 裁剪: 丢弃 chunk 前 k 步补偿延迟
      - 时间平滑: 相邻 chunk 重叠部分线性混合 (100% old → 0% new)
    """

    def __init__(self, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk: deque = deque()
        self.k: int = 0
        self.last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k=0, min_m=8):
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
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
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

    def has_any(self) -> bool:
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def size(self) -> int:
        with self.lock:
            return len(self.cur_chunk)

    def clear(self):
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0


# ═══════════════════════════════════════════════════════════════════════════════
# AAC + RTC 联合推理线程
# ═══════════════════════════════════════════════════════════════════════════════
def inference_thread_aac_rtc(
    args,
    model: PI05_DUAL,
    robot,
    k_selector: AdaptiveKSelector,
    stream_buffer: StreamActionBuffer,
    shutdown_event: threading.Event,
):
    """
    AAC + RTC 异步推理线程.

    AAC 侧: AdaptiveKSelector 动态调整 k
    RTC 侧: prev_chunk 引导 + 延迟估计

    流程:
      1. 获取观测, 获取自适应 k
      2. 推理 (携带 RTC payload)
      3. AAC 更新 k → Bezier 贝塞尔过渡
      4. 推入 StreamActionBuffer (RTC 时间平滑)
    """
    # RTC 状态
    delay_buffer: deque = deque(maxlen=20)
    pred_delay_steps = 0
    rtc_prev_chunk: np.ndarray | None = None
    prev_chunk_lock = threading.Lock()

    inference_interval = 1.0 / max(args.inference_rate, 1)
    chunk_round = 0

    while not shutdown_event.is_set():
        loop_start = time.time()

        try:
            # ── 1. 获取观测 ──
            data = robot.get()
            img_arr, state = input_transform(data)

            # ── 2. AAC: 获取当前自适应 k ──
            current_k = k_selector.get_k()
            model.pi0_step = current_k

            # ── 3. 更新观测窗口 ──
            model.update_observation_window(img_arr, state)

            # ── 4. RTC: 注入 payload ──
            if model.observation_window is not None and args.enable_rtc:
                model.observation_window["execute_horizon"] = current_k
                model.observation_window["enable_rtc"] = True
                with prev_chunk_lock:
                    pc = np.array(rtc_prev_chunk) if rtc_prev_chunk is not None else None
                if pc is not None:
                    model.observation_window["prev_action_chunk"] = pc
                model.observation_window["inference_delay"] = int(max(0, pred_delay_steps))
                model.observation_window["mask_prefix_delay"] = args.mask_prefix_delay
                model.observation_window["max_guidance_weight"] = args.max_guidance_weight

            # ── 5. 推理 ──
            t0 = time.time()
            action_chunk = model.get_action()
            rtt = time.time() - t0

            # ── 6. RTC: 更新延迟估计 ──
            if rtt is not None and np.isfinite(rtt):
                delay_buffer.append(float(rtt))
                median_rtt = 0.0
                if len(delay_buffer) > 0:
                    median_rtt = float(np.median(np.asarray(delay_buffer, dtype=float)))
                    pred_delay_steps = int(max(0, round(median_rtt / args.control_dt)))

            # ── 7. AAC: 根据边界跳变更新 k (下次推理生效) ──
            k_selector.update(action_chunk)

            # ── 8. AAC: Bezier 贝塞尔过渡 (chunk 间位置+速度连续) ──
            action_chunk = model.blend_with_prev_chunk_bezier(action_chunk)

            # ── 9. RTC: 更新 prev_chunk ──
            if action_chunk is not None and len(action_chunk) > 0:
                with prev_chunk_lock:
                    rtc_prev_chunk = np.asarray(action_chunk, dtype=float).copy()

                # ── 10. RTC: 推入 StreamActionBuffer ──
                stream_buffer.integrate_new_chunk(
                    np.asarray(action_chunk, dtype=float),
                    max_k=int(args.latency_k),
                    min_m=int(args.min_smooth_steps),
                )

            chunk_round += 1
            debug_print("AAC_RTC",
                        f"chunk_round: {chunk_round} | k: {current_k} | "
                        f"推理耗时: {rtt*1000:.0f}ms | "
                        f"延迟中位数: {median_rtt*1000:.0f}ms | "
                        f"buffer: {stream_buffer.size()}",
                        "INFO")

        except Exception as e:
            debug_print("AAC_RTC", f"推理线程异常: {e}", "WARN")

        # ── 11. 控制推理频率 ──
        elapsed = time.time() - loop_start
        sleep_time = max(0, inference_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


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
    parser = argparse.ArgumentParser(description="AAC + RTC 联合推理")

    # 基础
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

    # AAC
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_init", type=int, default=None)
    parser.add_argument("--delta_high", type=float, default=None)
    parser.add_argument("--delta_low", type=float, default=None)
    parser.add_argument("--k_step_up", type=int, default=None)
    parser.add_argument("--k_step_down", type=int, default=None)
    parser.add_argument("--blend_steps", type=int, default=None)

    # RTC
    parser.add_argument("--inference_rate", type=float, default=None)
    parser.add_argument("--smooth_method", type=str, default=None, choices=["temporal", "raw"])
    parser.add_argument("--min_smooth_steps", type=int, default=None)
    parser.add_argument("--latency_k", type=int, default=None)
    parser.add_argument("--decay_alpha", type=float, default=None)
    parser.add_argument("--enable_rtc", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--mask_prefix_delay", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--max_guidance_weight", type=float, default=None)
    parser.add_argument("--ema_alpha", type=float, default=None)

    args = parser.parse_args()

    # 加载配置
    config_path = (
        Path(args.config) if args.config
        else Path(__file__).with_name("deploy_pi05_real_aac_rtc.yml")
    )
    config = _load_yaml(config_path)

    # 合并: 命令行 > yaml > 默认值
    merged = {}
    merged.update(AAC_DEFAULTS)
    merged.update(RTC_DEFAULTS)
    merged.update(config)
    for key in (
        "model_path", "task_name", "robot_name", "robot_class",
        "episode_num", "max_step", "control_dt", "video",
        # AAC
        "k_min", "k_max", "k_init", "delta_high", "delta_low",
        "k_step_up", "k_step_down", "blend_steps",
        # RTC
        "inference_rate", "smooth_method", "min_smooth_steps",
        "latency_k", "decay_alpha",
        "enable_rtc", "mask_prefix_delay", "max_guidance_weight",
        "ema_alpha",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value

    merged["node"] = bool(args.node or merged.get("node", False))
    for key in list(merged.keys()):
        merged[key] = _normalize_value(merged[key])

    merged.setdefault("episode_num", 1)
    merged.setdefault("max_step", 1000000)
    merged.setdefault("control_dt", 1 / 30)
    merged.setdefault("video", None)

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
    model._ema_alpha = args.ema_alpha
    model._blend_steps = args.blend_steps

    # ── 初始化 AAC 自适应 k-Selector ──
    k_selector = AdaptiveKSelector(
        k_min=args.k_min,
        k_max=args.k_max,
        k_init=args.k_init,
        delta_high=args.delta_high,
        delta_low=args.delta_low,
        k_step_up=args.k_step_up,
        k_step_down=args.k_step_down,
    )

    # ── 初始化 RTC StreamActionBuffer ──
    stream_buffer = StreamActionBuffer(
        decay_alpha=args.decay_alpha,
        state_dim=14,
        smooth_method=args.smooth_method,
    )

    # ── 初始化机器人 ──
    robot_class = _get_class(f"my_robot.{args.robot_name}", args.robot_class)
    if args.node:
        robot_class = build_robot_node(robot_class)
    robot = robot_class()
    robot.set_up()

    shutdown_event = threading.Event()

    for episode in range(args.episode_num):
        step = 0
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()
        k_selector.reset()
        stream_buffer.clear()
        shutdown_event.clear()

        # ── FFmpeg 写入器 ──
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
                print("start to inference (AAC+RTC mode), press ENTER to end...")
            else:
                print("waiting for start command, press ENTER to start...")
                time.sleep(1)

        # ── 启动 AAC+RTC 联合推理线程 ──
        infer_thread = threading.Thread(
            target=inference_thread_aac_rtc,
            args=(args, model, robot, k_selector, stream_buffer, shutdown_event),
            daemon=True,
        )
        infer_thread.start()

        # ═════════════════════════════════════════════════════════════════════
        # 执行主循环 — 从 StreamActionBuffer 逐帧弹出
        # ═════════════════════════════════════════════════════════════════════
        prev_action = None

        while step < args.max_step and is_start:
            # 1. 从缓冲区取动作
            raw_action = stream_buffer.pop_next_action()
            if raw_action is None:
                time.sleep(0.001)
                continue

            # 2. EMA 逐帧平滑
            if prev_action is not None:
                smoothed_action = (
                    args.ema_alpha * raw_action
                    + (1 - args.ema_alpha) * prev_action
                )
            else:
                smoothed_action = raw_action
            prev_action = smoothed_action.copy()

            # 3. 录像
            if video_writers:
                _, cam_data = robot.get()
                for cam_key, writer in video_writers.items():
                    frame = cam_data[cam_key]["color"][:, :, ::-1]
                    writer.write(frame)

            # 4. 日志
            if step % 50 == 0:
                current_k = k_selector.get_k()
                debug_print("main",
                            f"step: {step}/{args.max_step} | "
                            f"k: {current_k} | "
                            f"buffer: {stream_buffer.size()}",
                            "INFO")

            # 5. 执行
            move_data = output_transform(smoothed_action)
            robot.move(move_data)
            step += 1

            time.sleep(args.control_dt)

            # 6. 停止
            if step >= args.max_step or is_enter_pressed():
                debug_print("main", "enter pressed, the episode end", "INFO")
                is_start = False
                break

        # ── 收尾 ──
        shutdown_event.set()
        infer_thread.join(timeout=3)
        for writer in video_writers.values():
            writer.release()
        debug_print("main", f"finish episode {episode}, running steps {step}", "INFO")


if __name__ == "__main__":
    main()
