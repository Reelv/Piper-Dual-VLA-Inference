# Piper Dual VLA Inference

面向 AgileX Piper 双臂真机的 pi05 VLA 推理工程。这个仓库基于 `control_your_robot` 的机器人控制/采集框架整理，重点保留 Piper 双臂真机推理入口，并加入三类动作执行改动：

- `baseline`: 原始 pi05 action chunk 推理 + EMA/Bezier 平滑。
- `AAC`: Adaptive Action Chunking，根据相邻 chunk 动作跳变动态调整 chunk 长度。
- `RTC`: Real-Time Chunking，异步推理、流式 action buffer、chunk 重叠区时间平滑。
- `AAC + RTC`: 自适应 chunk 长度与 RTC 异步执行结合。

权重、数据、录像和本机硬件 ID 不随仓库发布。开源前请先看 [开源检查清单](docs/OPEN_SOURCE_CHECKLIST.md)。

## 适用硬件

- 两台 AgileX Piper 从臂，CAN 接口默认名为 `can_left` 和 `can_right`。
- 三路 RealSense RGB 相机：`cam_head`、`cam_left_wrist`、`cam_right_wrist`。
- 推理前置测试流程与 `Piper-Dual-Teleop` 一致：先确认双臂可使能、可读状态、可执行基础运动，再运行 VLA 推理。

详细流程见 [Piper 双臂前置检查](docs/PIPER_DUAL_PREFLIGHT.md)。

## 仓库结构

```text
Piper-Dual-VLA-Inference/
├── my_robot/
│   └── agilex_piper_dual_base.py         # Piper 双臂 + 三相机真机配置
├── src/robot/
│   ├── controller/Piper_controller.py    # Piper CAN 控制器
│   ├── sensor/Realsense_sensor.py        # RealSense 传感器
│   └── policy/pi05/
│       ├── inference_model.py            # pi05 输入/输出变换与平滑工具
│       ├── deploy_pi05_real.py           # baseline 真机推理
│       ├── deploy_pi05_real_aac.py       # AAC 真机推理
│       ├── deploy_pi05_real_rtc.py       # RTC 真机推理
│       └── deploy_pi05_real_aac_rtc.py   # AAC + RTC 真机推理
├── task_instructions/                    # 任务语言指令示例
├── docs/
└── .env.example
```

## 安装

建议 Python 3.10/3.11，真实 Piper 控制还需要系统已配置好 CAN 和 `piper_sdk`。

```bash
cd Piper-Dual-VLA-Inference
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果 `piper_sdk` 没有安装到环境里，设置 SDK 路径：

```bash
export PIPER_SDK_ROOT=/path/to/Piper-Dual-Teleop/piper_sdk
```

RealSense 序列号不要写死进代码，运行前通过环境变量配置：

```bash
export PIPER_CAM_HEAD_SERIAL=replace_me
export PIPER_CAM_LEFT_WRIST_SERIAL=replace_me
export PIPER_CAM_RIGHT_WRIST_SERIAL=replace_me
```

## 权重与任务指令

默认配置里的 `model_path` 是占位示例：

```yaml
model_path: ./src/robot/policy/pi05/checkpoints/pi05_piper_full_base/sort_out_40/20000
task_name: sort_out_40
```

请把自己的 checkpoint 放到被 `.gitignore` 忽略的 `checkpoints/` 路径，或在运行时用 `--model_path` 指定绝对路径。`task_name` 需要和 `task_instructions/<task_name>.json` 对应。

## 推理

先完成前置检查，确认双臂附近安全、急停可用、CAN 接口和相机都正常。

Baseline:

```bash
python src/robot/policy/pi05/deploy_pi05_real.py \
  --config src/robot/policy/pi05/deploy_pi05_real.yml
```

AAC:

```bash
python src/robot/policy/pi05/deploy_pi05_real_aac.py \
  --config src/robot/policy/pi05/deploy_pi05_real.yml
```

RTC:

```bash
python src/robot/policy/pi05/deploy_pi05_real_rtc.py \
  --config src/robot/policy/pi05/deploy_pi05_real_rtc.yml
```

AAC + RTC:

```bash
python src/robot/policy/pi05/deploy_pi05_real_aac_rtc.py \
  --config src/robot/policy/pi05/deploy_pi05_real_aac_rtc.yml
```

脚本启动后会等待回车再开始推理。运行中再次按回车结束当前 episode。

## 开源注意事项

- 不提交 `checkpoints/`、`save/`、`datasets/`、`output_video/`、`.venv/`。
- 不提交真实相机序列号、CAN 适配器 bus-info、本机绝对路径。
- 真机执行默认应先低速、小步、空场景验证。
- 如果发布到 GitHub，建议在 release 或 README 中说明权重获取方式，而不是把权重直接放进 git。

## 上游说明

本项目基于 `control_your_robot` 整理，保留 Apache-2.0 License。正式发布前建议补充你自己的修改说明、论文/项目引用和权重获取方式。
