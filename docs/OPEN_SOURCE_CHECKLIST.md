# 开源检查清单

发布前建议逐项确认。

## 必须删除或忽略

- `.venv/`
- `checkpoints/`
- `src/robot/policy/**/checkpoints/`
- `save/`
- `datasets/`
- `output_video/`
- `*.hdf5`
- `*.mp4`
- `*.npy`
- 真实 RealSense 序列号
- 本机绝对路径，例如 `/home/ths/...`
- 私人微信二维码、内部图片和未授权素材

## 建议保留

- 推理源码：`src/robot/policy/pi05/`
- Piper 双臂机器人配置：`my_robot/agilex_piper_dual_base.py`
- 任务指令 JSON 示例：`task_instructions/`
- 安装、前置检查、推理命令说明
- License、NOTICE 和上游引用

## 发布建议

1. 新建 GitHub 仓库。
2. 在本目录执行 `git init`。
3. 用 `git status --ignored` 检查权重、数据、视频是否被忽略。
4. 如果要发布模型权重，使用 GitHub Release、Hugging Face 或网盘，不要直接提交到 git。
5. README 中写清楚真机风险、硬件版本、相机布局、动作维度和任务指令格式。
