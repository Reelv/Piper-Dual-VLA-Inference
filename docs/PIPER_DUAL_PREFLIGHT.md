# Piper 双臂前置检查

推理前的硬件检查流程与 `Piper-Dual-Teleop` 保持一致。目标是先确认两台 Piper 可以安全移动，再启动 VLA 推理。

## 1. CAN 接口

插入两个 CAN 适配器，检查接口：

```bash
ip -br link show type can
```

本仓库默认使用：

- 左臂：`can_left`
- 右臂：`can_right`

如果你的机器使用 `can0/can1`，需要修改 `my_robot/agilex_piper_dual_base.py` 中 `set_up()` 里的 CAN 名称，或在系统侧创建稳定命名。

## 2. Piper SDK

如果使用 `Piper-Dual-Teleop` 自带 SDK：

```bash
export PIPER_SDK_ROOT=/path/to/Piper-Dual-Teleop/piper_sdk
```

检查 SDK 能导入：

```bash
python -c "import piper_sdk; print('piper_sdk ok')"
```

## 3. 使能、读状态、基础运动

推荐先在 `Piper-Dual-Teleop` 中跑同一套检查：

```bash
cd /path/to/Piper-Dual-Teleop
python tools/enable_dual.py
python tools/read_states_dual.py
python examples/piper_ctrl_moveP_once_dual.py
python tools/disable_dual.py
```

这些检查通过后，再回到本仓库运行推理。

## 4. RealSense 相机

先读取相机序列号：

```bash
python tools/realsense_serial.py
```

设置环境变量：

```bash
export PIPER_CAM_HEAD_SERIAL=replace_me
export PIPER_CAM_LEFT_WRIST_SERIAL=replace_me
export PIPER_CAM_RIGHT_WRIST_SERIAL=replace_me
```

相机对应关系必须和训练数据一致：

- `cam_head` -> openpi/pi05 观测中的 `cam_high`
- `cam_left_wrist` -> 左腕相机
- `cam_right_wrist` -> 右腕相机

## 5. 推理前安全检查

- 两臂工作空间内无人、无松散线缆。
- 急停可触达。
- 首次测试把 `max_step` 设小，例如 `100`。
- 首次测试建议关闭录像：`video: null`。
- 如果动作方向异常，立即停止并检查 `input_transform()` / `output_transform()` 的左右臂顺序、夹爪尺度和训练数据格式。
