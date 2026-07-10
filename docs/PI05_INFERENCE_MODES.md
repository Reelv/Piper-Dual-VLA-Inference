# pi05 推理模式

## Baseline

入口：`src/robot/policy/pi05/deploy_pi05_real.py`

流程：

1. 获取双臂状态和三路 RGB 图像。
2. `PI05_DUAL.get_action()` 输出 action chunk。
3. 对 chunk 边界做 Bezier 过渡。
4. 执行层逐帧 EMA 平滑。
5. `output_transform()` 转成 `robot.move()` 输入。

## AAC

入口：`src/robot/policy/pi05/deploy_pi05_real_aac.py`

核心参数：

- `k_min`, `k_max`, `k_init`
- `delta_high`, `delta_low`
- `k_step_up`, `k_step_down`
- `ema_alpha`, `blend_steps`

根据新旧 chunk 边界动作差异动态调整下一次执行的 chunk 长度。跳变大时减小 `k`，跳变小时增大 `k`。

## RTC

入口：`src/robot/policy/pi05/deploy_pi05_real_rtc.py`

核心参数：

- `chunk_size`
- `execute_horizon`
- `inference_rate`
- `smooth_method`
- `min_smooth_steps`
- `latency_k`
- `enable_rtc`
- `max_guidance_weight`

推理线程和执行线程分离，新的 action chunk 写入 `StreamActionBuffer`，执行线程持续取下一帧动作。相邻 chunk 的重叠部分会做时间混合。

## AAC + RTC

入口：`src/robot/policy/pi05/deploy_pi05_real_aac_rtc.py`

结合 AAC 的自适应 chunk 长度和 RTC 的异步缓冲执行。建议先分别调通 baseline、AAC、RTC，再运行联合模式。
