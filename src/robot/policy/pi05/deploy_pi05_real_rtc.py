#!/usr/bin/env python3
"""
RTC (Real-Time Chunking) 推理脚本.

参考论文: "Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025)
来自 kai0 项目的 train_deploy_alignment 模块.

核心思想:
  - 异步推理: 推理线程非阻塞地将 action chunk 推入 StreamActionBuffer
  - 时间平滑: 缓冲区对相邻 chunk 重叠部分做线性混合 (100% old → 0% new)
  - 上一 chunk 引导 (RTC): 将已执行的 action chunk 前缀发送给模型,
    引导新 chunk 生成时与已执行部分对齐
  - 延迟估计: 基于推理 RTT 中位数预测延迟步数
  - EMA 逐帧平滑: 执行层对每个动作做指数平滑

与 AAC (deploy_pi05_real_aac.py) 的区别:
  - AAC: 自适应 k-selector + Bezier 贝塞尔过渡, 同步推理
  - RTC: 固定 chunk_size + StreamActionBuffer 时间平滑 + 异步推理 + prev_chunk 引导

Usage:
  # 使用默认配置 (deploy_pi05_real_rtc.yml)
  python deploy_pi05_real_rtc.py

  # 使用自定义配置
  python deploy_pi05_real_rtc.py --config my_rtc_config.yml

  # 远程服务器模式 (需要先在 GPU 主机启动 policy server)
  python deploy_pi05_real_rtc.py --remote --host 192.168.1.100 --port 8000
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
# RTC 默认参数
# ═══════════════════════════════════════════════════════════════════════════════
RTC_DEFAULTS = {
    "chunk_size": 50,
    "execute_horizon": 25,
    "inference_rate": 10,
    "smooth_method": "temporal",
    "min_smooth_steps": 30,
    "latency_k": 3,
    "decay_alpha": 0.15,
    "enable_rtc": True,
    "mask_prefix_delay": False,
    "max_guidance_weight": 0.3 ,
    "ema_alpha": 0.3,
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
# StreamActionBuffer — RTC 核心: chunk 队列 + 时间平滑
# ═══════════════════════════════════════════════════════════════════════════════
class StreamActionBuffer:
    """
    Action chunk 缓冲区，支持:
      - latency_k 裁剪: 丢弃 chunk 前 k 步以补偿推理延迟
      - 时间平滑: 对相邻 chunk 重叠部分做线性混合 (100% old → 0% new)
      - 逐帧弹出: pop_next_action() 每次返回一个动作

    设计参考 kai0/train_deploy_alignment/inference/ 中的实现.
    """

    def __init__(
        self,
        decay_alpha: float = 0.25,
        state_dim: int = 14,
        smooth_method: str = "temporal",
    ):
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk: deque = deque()       # 当前待执行的动作队列
        self.k: int = 0                        # 已从当前 chunk 弹出的步数
        self.last_action: np.ndarray | None = None  # 上一个弹出的动作

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int = 0,
        min_m: int = 8,
    ):
        """
        集成新的推理 chunk:
          (1) 裁剪前 max_k 步 (基于当前 k 的延迟补偿)
          (2) 与旧 chunk 重叠部分做线性混合
          (3) 重置 k = 0
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))

            # (1) 延迟裁剪: 丢弃前 drop_n 步
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            # 如果不需要平滑, 直接替换
            if str(self.smooth_method).lower() == "raw":
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return

            # (2) 构建 old_list (上一个 chunk 的剩余部分)
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                # 上一个 chunk 已耗尽, 用 last_action 补齐 min_m 步
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

            # 对齐长度
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            # (3) 线性混合: w_old 从 1→0, w_new 从 0→1
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
        """弹出下一个动作, 递增 k."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# RTC 推理线程 — 异步非阻塞
# ═══════════════════════════════════════════════════════════════════════════════
def inference_thread_rtc(
    args,
    model: PI05_DUAL,
    robot,
    stream_buffer: StreamActionBuffer,
    shutdown_event: threading.Event,
    # 共享状态 (由主线程和推理线程共同访问)
    state_lock: threading.Lock,
    shared_state: dict,
):
    """
    RTC 异步推理线程:
      - 获取最新观测, 构建 RTC payload
      - 调用模型推理, 将结果推入 stream_buffer
      - 更新 prev_chunk 和延迟估计

    与 AAC 的关键区别: payload 中携带 prev_action_chunk + inference_delay,
    让模型感知"已执行了什么", 从而生成更连贯的后续动作.
    """
    chunk_size = args.chunk_size
    exec_h = min(args.execute_horizon, chunk_size)

    # 延迟估计: 滑动窗口记录推理耗时
    delay_buffer: deque = deque(maxlen=20)
    pred_delay_steps = 0

    # RTC prev_chunk 追踪
    rtc_prev_chunk: np.ndarray | None = None
    prev_chunk_lock = threading.Lock()

    inference_interval = 1.0 / max(args.inference_rate, 1)

    while not shutdown_event.is_set():
        loop_start = time.time()

        try:
            # 1. 获取观测
            data = robot.get()
            img_arr, state = input_transform(data)

            # 2. 更新模型观测窗口
            model.update_observation_window(img_arr, state)

            # 3. 设置 chunk 大小
            model.pi0_step = chunk_size

            # 4. 构建 RTC payload (注入到 observation_window 中)
            #    本地模型模式下, 我们通过 observation_window 传递 RTC 参数
            if model.observation_window is not None:
                model.observation_window["execute_horizon"] = exec_h
                model.observation_window["enable_rtc"] = args.enable_rtc

                with prev_chunk_lock:
                    pc = np.array(rtc_prev_chunk) if rtc_prev_chunk is not None else None
                if pc is not None:
                    model.observation_window["prev_action_chunk"] = pc

                model.observation_window["inference_delay"] = int(max(0, pred_delay_steps))
                model.observation_window["mask_prefix_delay"] = args.mask_prefix_delay
                model.observation_window["max_guidance_weight"] = args.max_guidance_weight

            # 5. 推理
            t0 = time.time()
            action_chunk = model.get_action()
            rtt = time.time() - t0

            # 6. 更新延迟估计 & 打印推理耗时
            if rtt is not None and np.isfinite(rtt):
                delay_buffer.append(float(rtt))
                if len(delay_buffer) > 0:
                    median_rtt = float(np.median(np.asarray(delay_buffer, dtype=float)))
                    pred_delay_steps = int(max(0, round(median_rtt / args.control_dt)))
                debug_print("RTC_infer",
                            f"推理耗时: {rtt*1000:.0f}ms | "
                            f"滑动窗口中位数: {median_rtt*1000:.0f}ms | "
                            f"预估延迟: {pred_delay_steps}步",
                            "INFO")

            # 7. 更新 prev_chunk (供下次推理使用)
            if action_chunk is not None and len(action_chunk) > 0:
                with prev_chunk_lock:
                    rtc_prev_chunk = np.asarray(action_chunk, dtype=float).copy()

                # 8. 推入 stream_buffer
                stream_buffer.integrate_new_chunk(
                    np.asarray(action_chunk, dtype=float),
                    max_k=int(args.latency_k),
                    min_m=int(args.min_smooth_steps),
                )

        except Exception as e:
            debug_print("RTC_infer", f"推理线程异常: {e}", "WARN")

        # 9. 控制推理频率
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
    parser = argparse.ArgumentParser(description="RTC (Real-Time Chunking) 推理")

    # 基础参数
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

    # RTC 特有参数
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--execute_horizon", type=int, default=None)
    parser.add_argument("--inference_rate", type=float, default=None)
    parser.add_argument("--smooth_method", type=str, default=None,
                        choices=["temporal", "raw"])
    parser.add_argument("--min_smooth_steps", type=int, default=None)
    parser.add_argument("--latency_k", type=int, default=None)
    parser.add_argument("--decay_alpha", type=float, default=None)
    parser.add_argument("--enable_rtc", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--mask_prefix_delay", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--max_guidance_weight", type=float, default=None)
    parser.add_argument("--ema_alpha", type=float, default=None)

    # 远程服务器模式 (可选, 用于连接 kai0 的 policy server)
    parser.add_argument("--remote", action="store_true",
                        help="使用远程 policy server 而非本地模型")
    parser.add_argument("--host", type=str, default="localhost",
                        help="policy server 地址 (--remote 模式)")
    parser.add_argument("--port", type=int, default=8000,
                        help="policy server 端口 (--remote 模式)")

    args = parser.parse_args()

    # 加载配置文件
    config_path = (
        Path(args.config) if args.config
        else Path(__file__).with_name("deploy_pi05_real_rtc.yml")
    )
    config = _load_yaml(config_path)

    # 合并: 命令行 > yaml > 默认值
    merged = {}
    merged.update(RTC_DEFAULTS)
    merged.update(config)
    for key in (
        "model_path", "task_name", "robot_name", "robot_class",
        "episode_num", "max_step", "control_dt", "video",
        "chunk_size", "execute_horizon", "inference_rate",
        "smooth_method", "min_smooth_steps", "latency_k", "decay_alpha",
        "enable_rtc", "mask_prefix_delay", "max_guidance_weight", "ema_alpha",
        "remote", "host", "port",
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

    if not merged.get("remote"):
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
    if args.remote:
        # 远程模式: 连接 kai0 policy server
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        print(f"[RTC] 已连接远程服务器: {args.host}:{args.port}")
        print(f"  Server metadata: {policy.get_server_metadata()}")

        # 构造一个简单的本地模型包装
        class RemoteModelWrapper:
            def __init__(self):
                self.observation_window = None
                self.pi0_step = args.chunk_size

            def update_observation_window(self, img_arr, state):
                pass  # 远程模式下观测由推理线程直接构建

            def reset_obsrvationwindows(self):
                self.observation_window = None

            def random_set_language(self):
                pass

        model = RemoteModelWrapper()
        use_remote = True
    else:
        model = PI05_DUAL(args.model_path, args.task_name)
        use_remote = False

    # 注入 EMA 平滑参数
    model._ema_alpha = args.ema_alpha

    # ── 初始化 StreamActionBuffer ──
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

    # 共享状态
    shutdown_event = threading.Event()

    for episode in range(args.episode_num):
        step = 0
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()
        stream_buffer.clear()
        shutdown_event.clear()

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
                print("start to inference (RTC mode), press ENTER to end...")
            else:
                print("waiting for start command, press ENTER to start...")
                time.sleep(1)

        # ── 启动 RTC 异步推理线程 ──
        if use_remote:
            # 远程模式: 推理线程直接通过 websocket 调用
            def _remote_infer_loop():
                chunk_size = args.chunk_size
                exec_h = min(args.execute_horizon, chunk_size)
                delay_buffer_remote: deque = deque(maxlen=20)
                pred_delay = 0
                prev_chunk = None
                prev_chunk_lock = threading.Lock()
                inference_interval = 1.0 / max(args.inference_rate, 1)

                # lang prompt
                import json as _json
                root_dir = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                json_path = os.path.join(root_dir, "task_instructions",
                                         f"{args.task_name}.json")
                with open(json_path, "r") as f:
                    instr = _json.load(f)
                lang = instr["instructions"][0]
                print(f"[RTC Remote] lang: {lang}")

                while not shutdown_event.is_set():
                    loop_start = time.time()
                    try:
                        data = robot.get()
                        img_arr, state = input_transform(data)
                        imgs = [
                            np.transpose(img_arr[0], (2, 0, 1)),
                            np.transpose(img_arr[1], (2, 0, 1)),
                            np.transpose(img_arr[2], (2, 0, 1)),
                        ]
                        payload = {
                            "state": state,
                            "images": {
                                "cam_high": imgs[0],
                                "cam_right_wrist": imgs[1],
                                "cam_left_wrist": imgs[2],
                            },
                            "prompt": lang,
                            "execute_horizon": exec_h,
                            "enable_rtc": args.enable_rtc,
                            "mask_prefix_delay": args.mask_prefix_delay,
                            "max_guidance_weight": args.max_guidance_weight,
                        }
                        with prev_chunk_lock:
                            pc = np.array(prev_chunk) if prev_chunk is not None else None
                        if pc is not None:
                            payload["prev_action_chunk"] = pc.tolist()
                        payload["inference_delay"] = int(max(0, pred_delay))

                        t0 = time.time()
                        out = policy.infer(payload)
                        rtt = time.time() - t0

                        if rtt is not None and np.isfinite(rtt):
                            delay_buffer_remote.append(float(rtt))
                            if len(delay_buffer_remote) > 0:
                                med = float(np.median(
                                    np.asarray(delay_buffer_remote, dtype=float)))
                                pred_delay = int(max(0, round(med / args.control_dt)))

                        actions = out.get("actions", None)
                        if actions is not None and len(actions) > 0:
                            with prev_chunk_lock:
                                prev_chunk = np.asarray(actions, dtype=float).copy()
                            stream_buffer.integrate_new_chunk(
                                np.asarray(actions, dtype=float),
                                max_k=int(args.latency_k),
                                min_m=int(args.min_smooth_steps),
                            )
                    except Exception as e:
                        debug_print("RTC_remote", f"远程推理异常: {e}", "WARN")

                    elapsed = time.time() - loop_start
                    sleep_t = max(0, inference_interval - elapsed)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            infer_thread = threading.Thread(target=_remote_infer_loop, daemon=True)
        else:
            infer_thread = threading.Thread(
                target=inference_thread_rtc,
                args=(args, model, robot, stream_buffer, shutdown_event,
                      threading.Lock(), {}),
                daemon=True,
            )
        infer_thread.start()

        # ═════════════════════════════════════════════════════════════════════
        # 主执行循环 — 从 stream_buffer 逐帧弹出并执行
        # ═════════════════════════════════════════════════════════════════════
        prev_action = None  # 用于 EMA 平滑

        while step < args.max_step and is_start:
            # 1. 从缓冲区获取下一个动作 (非阻塞)
            raw_action = stream_buffer.pop_next_action()

            if raw_action is None:
                # 缓冲区空, 等待推理线程产出
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
                buf_size = len(stream_buffer.cur_chunk) if stream_buffer else 0
                debug_print("RTC_main",
                            f"step: {step}/{args.max_step} | buffer: {buf_size}",
                            "INFO")

            # 5. 执行动作
            move_data = output_transform(smoothed_action)
            robot.move(move_data)
            step += 1

            time.sleep(args.control_dt)

            # 6. 停止条件
            if step >= args.max_step or is_enter_pressed():
                debug_print("RTC_main", "enter pressed, the episode end", "INFO")
                is_start = False
                break

        # ── 收尾 ──
        shutdown_event.set()
        infer_thread.join(timeout=3)
        for writer in video_writers.values():
            writer.release()
        debug_print("RTC_main", f"finish episode {episode}, running steps {step}", "INFO")


if __name__ == "__main__":
    main()
