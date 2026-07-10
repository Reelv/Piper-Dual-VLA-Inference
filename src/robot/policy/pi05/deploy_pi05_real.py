import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
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


# ── FFmpeg 管道写入器 ──────────────────────────────────────────────────────────
class FFmpegWriter:
    """
    把 BGR numpy 帧通过 stdin 管道写给 ffmpeg，输出 H.264 mp4。
    比 cv2.VideoWriter(mp4v) 兼容性强得多。
    """

    def __init__(self, path: str, width: int, height: int, fps: int = DEFAULT_FPS):
        self.path = path
        cmd = [
            "ffmpeg",
            "-y",                          # 覆盖已有文件
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",               # 从 stdin 读
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "18",                  # 质量（0=无损, 51=最差）
            "-pix_fmt", "yuv420p",         # 确保播放器兼容
            "-movflags", "+faststart",     # 支持流式播放
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        """frame: H×W×3 BGR uint8 numpy array"""
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass  # ffmpeg 进程已退出，静默忽略

    def release(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        print(f"[FFmpegWriter] saved → {self.path}")


# ── 工具函数 ──────────────────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser()
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


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    os.environ.setdefault("INFO_LEVEL", "INFO")
    args = parse_args()

    model = PI05_DUAL(args.model_path, args.task_name)
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

        # ── 初始化 FFmpeg 写入器 ──────────────────────────────────────────────
        video_writers: dict[str, FFmpegWriter] = {}
        if args.video is not None:
            _, cam_data = robot.get()
            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
            for cam_key in CAMERA_KEYS:
                first_frame = cam_data[cam_key]["color"][:, :, ::-1]  # RGB, H×W×3
                height, width = first_frame.shape[:2]
                video_path = os.path.join(
                    OUTPUT_VIDEO_DIR, f"episode_{episode}_{cam_key}.mp4"
                )
                video_writers[cam_key] = FFmpegWriter(video_path, width, height, DEFAULT_FPS)
                print(f"Video saving enabled: {video_path}")

        # ── 等待启动 ──────────────────────────────────────────────────────────
        print(f"当前推理 Instruction: {model.instruction}")
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("start to inference, press ENTER to end...")
            else:
                print("waiting for start command, press ENTER to start...")
                time.sleep(1)

        # ── 推理主循环 ────────────────────────────────────────────────────────
        while step < args.max_step and is_start:
            data = robot.get()
            img_arr, state = input_transform(data)
            model.update_observation_window(img_arr, state)
            #lzr 6.1 注释
            # action_chunk = model.get_action()
            # print(f"action_chunk[0]: {action_chunk[0]},gripper: action_chunk[0]['arm']['left_arm']['gripper']")
            action_chunk = model.get_action()
            action_chunk = model.blend_with_prev_chunk_bezier(action_chunk)

            # for i, action in enumerate(action_chunk):
            #     smoothed_action = model.smooth_action(action)  # 存到新变量
            for action in action_chunk:
                smoothed_action = model.smooth_action(action)  # 存到新变量
                if video_writers:
                    _, cam_data = robot.get()
                    for cam_key, writer in video_writers.items():
                        frame = cam_data[cam_key]["color"]
                        writer.write(frame)

                if step % 10 == 0:
                    debug_print("main", f"step: {step}/{args.max_step}", "INFO")

                # move_data = output_transform(action)
                # robot.move(move_data)
                # step += 1

                move_data = output_transform(smoothed_action)  # 用平滑后的
                robot.move(move_data)
                step += 1
                time.sleep(args.control_dt)
                if step >= args.max_step or is_enter_pressed():
                    debug_print("main", "enter pressed, the episode end", "INFO")
                    is_start = False
                    break

        # ── 收尾 ──────────────────────────────────────────────────────────────
        for writer in video_writers.values():
            writer.release()
        debug_print("main", f"finish episode {episode}, running steps {step}", "INFO")


if __name__ == "__main__":
    main()
