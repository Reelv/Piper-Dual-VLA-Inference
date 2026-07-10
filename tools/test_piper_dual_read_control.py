import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from my_robot.agilex_piper_dual_base import PiperDual

DEFAULT_QPOS = np.array([0.057, 0.0, 0.216, 0.0, 0.085, 0.0], dtype=np.float32)
DEFAULT_GRIPPER = 0.8


def _decode_if_bytes(img):
    if isinstance(img, bytes):
        jpeg_bytes = np.array(img).tobytes().rstrip(b"\0")
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(nparr, 1), False
    return img, True


def _resize_image(img, target_size):
    width, height = target_size
    if img.shape[0] != height or img.shape[1] != width:
        return cv2.resize(img, (width, height))
    return img


def _prepare_frame(img, label, is_rgb):
    if is_rgb:
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        frame = img

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = 8
    y = 8 + text_h
    cv2.rectangle(
        frame,
        (x - 4, y - text_h - 4),
        (x + text_w + 4, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, label, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return frame


def _stack_images_horiz(images, target_size):
    width, height = target_size
    resized = []
    for img in images:
        if img.shape[0] != height or img.shape[1] != width:
            resized.append(cv2.resize(img, (width, height)))
        else:
            resized.append(img)
    return np.hstack(resized)


def _parse_vec6(value):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated values")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _build_move_data(mode, left, right, gripper):
    move_data = {"arm": {}}
    if left is not None:
        move_data["arm"]["left_arm"] = {mode: left}
    if right is not None:
        move_data["arm"]["right_arm"] = {mode: right}
    if gripper is not None:
        gripper_value = float(np.clip(gripper, 0.0, 1.0))
        if "left_arm" in move_data["arm"]:
            move_data["arm"]["left_arm"]["gripper"] = gripper_value
        if "right_arm" in move_data["arm"]:
            move_data["arm"]["right_arm"]["gripper"] = gripper_value
    return move_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=0, help="0 means run until q when showing")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--mode", choices=["joint", "qpos"], default="qpos")
    parser.add_argument("--left", type=str, default=None)
    parser.add_argument("--right", type=str, default=None)
    parser.add_argument("--gripper", type=float, default=None)
    args = parser.parse_args()

    robot = PiperDual()
    robot.set_up()

    if args.move:
        left = _parse_vec6(args.left) if args.left else DEFAULT_QPOS
        right = _parse_vec6(args.right) if args.right else DEFAULT_QPOS
        gripper = DEFAULT_GRIPPER if args.gripper is None else args.gripper
        move_data = _build_move_data(args.mode, left, right, gripper)
        robot.move(move_data)
        time.sleep(0.5)

    i = 0
    while True:
        if args.loops > 0 and i >= args.loops:
            break
        data = robot.get()
        arm_data, cam_data = data[0], data[1]

        left = arm_data.get("left_arm", {})
        right = arm_data.get("right_arm", {})

        left_joint = np.array(left.get("joint", []))
        right_joint = np.array(right.get("joint", []))
        left_qpos = np.array(left.get("qpos", []))
        right_qpos = np.array(right.get("qpos", []))
        print(
            f"[{i}] left_joint={left_joint} left_qpos={left_qpos} left_gripper={left.get('gripper')} "
            f"right_joint={right_joint} right_qpos={right_qpos} right_gripper={right.get('gripper')}"
        )

        cam_order = ("cam_head", "cam_left_wrist", "cam_right_wrist")
        imgs = []
        for cam_key in cam_order:
            img, is_rgb = _decode_if_bytes(cam_data[cam_key]["color"])
            print(f"  {cam_key}: shape={img.shape}, dtype={img.dtype}")
            imgs.append((cam_key, img, is_rgb))

        if args.show:
            base_h, base_w = imgs[0][1].shape[:2]
            prepared = []
            for cam_key, img, is_rgb in imgs:
                resized = _resize_image(img, (base_w, base_h))
                prepared.append(_prepare_frame(resized, cam_key, is_rgb))
            stacked = _stack_images_horiz(prepared, (base_w, base_h))
            cv2.imshow("piper_dual_cams", stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        time.sleep(args.sleep)
        i += 1

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# python test_piper_dual_read_control.py --show 
# python test_piper_dual_read_control.py --move --show