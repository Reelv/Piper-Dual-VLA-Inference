# AAC + RTC 联合推理：为什么将自适应动作分块与实时分块执行结合？

## 1. 背景概述

### 1.1 Action Chunking 的起源与优势

**Action Chunking（动作分块）** 是现代机器人模仿学习中的核心技术范式，最早由以下工作系统化提出：

- **Diffusion Policy** (Chi et al., RSS 2023 / IJRR 2024)：首次将扩散模型与 Receding-Horizon Control（滚动时域控制）结合，一次性预测未来多步动作序列（action chunk），而非逐帧单步预测。相比 LSTM-GMM、IBC、BET 等方法，在 12 个任务上平均成功率提升 **46.9%**。
- **ACT (Action Chunking Transformer)** (Zhao et al., 2023)：基于 Transformer 的编码器-解码器架构，直接预测动作块，在 ALOHA 双臂操作中表现优异。

**Action Chunking 的核心优势：**
- **时序一致性**：预测完整轨迹段而非孤立单帧，动作在时间上更连贯
- **多模态处理**：能自然应对同一状态下多种有效动作策略（如避障的左右两条路径）
- **鲁棒性**：对视觉扰动和干扰具有更强抵抗力
- **减少累积误差**：chunk 内动作相互关联，减少逐帧推理的漂移

### 1.2 从 Action Chunking 到 Real-Time 部署的挑战

然而，将 Action Chunking 策略部署到真实机器人上时，面临根本性矛盾：

> **Chunk 越大 → 轨迹越平滑，但响应越滞后**
> **Chunk 越小 → 响应越快，但轨迹越抖动**

具体挑战包括：
1. **固定 chunk size 无法适应动态场景**：精细操作需要高频重规划（小 k），大范围移动可以低频推理（大 k）
2. **Chunk 间跳变**：新 chunk 的第一帧与旧 chunk 的最后一帧可能不连续，造成机械臂抖动
3. **推理延迟与执行频率不匹配**：VLA 模型推理耗时（50-200ms），而控制频率需 30-50Hz
4. **同步阻塞问题**：传统"推理→执行全部→再推理"模式导致执行线程空闲等待

---

## 2. 两个核心组件

### 2.1 AAC：自适应动作分块 (Adaptive Action Chunking)

> 参考论文：*"Adaptive Action Chunking for Real-Time Edge VLA Control"*
> 相关学术工作：StreamingVLA (Shi et al., 2026), Denoising-Variance Adaptive Chunking (Feng et al., RSS 2026)

**AAC 解决的核心问题：** "什么时候该重新规划？chunk 大小该多大？"

**核心机制：**

| 组件 | 功能 |
|------|------|
| **AdaptiveKSelector** | 根据跨 chunk 动作跳变 $\Delta$ 动态调整 k |
| **Bezier 贝塞尔过渡** | 保证相邻 chunk 间位置 + 速度连续（$C^1$ 连续） |

**自适应 k 调节逻辑：**

```
if Δ > δ_high:  k ↓  (动作变化剧烈 → 减小 chunk，提高响应速度)
if Δ < δ_low:   k ↑  (动作趋于稳定 → 增大 chunk，提高平滑度)
else:           k 不变
```

**直觉解释：**
- 当机器人需要急转弯或抓取物体时（大 $\Delta$），快速减小 chunk size 让策略更频繁地重规划
- 当机器人沿直线移动时（小 $\Delta$），增大 chunk size 减少推理开销，保持轨迹平滑

### 2.2 RTC：实时分块执行 (Real-Time Chunking)

> 参考论文：*"Real-Time Execution of Action Chunking Flow Policies"* (Black, Galliker, Levine, NeurIPS 2025)
> 来自 kai0 项目的 `train_deploy_alignment` 模块
> 相关学术工作：ABPolicy (Yang et al., 2026), Legato (Liu et al., RSS 2026), TIDAL (Sun et al., 2026), DiscreteRTC (Wang et al., 2026)

**RTC 解决的核心问题：** "如何让异步推理与同步执行无缝衔接？"

**核心机制：**

| 组件 | 功能 |
|------|------|
| **StreamActionBuffer** | 相邻 chunk 重叠部分线性混合（100% old → 0% new） |
| **异步推理线程** | 推理线程与执行线程完全分离，互不阻塞 |
| **prev_chunk 引导** | RTC payload 携带已执行前缀，确保新 chunk 感知历史 |
| **延迟估计** | 实时统计推理延迟中位数，自适应补偿 |

**时间平滑（Temporal Smoothing）原理：**

```
重叠区域:
  blended[i] = w_old[i] × old_chunk[i] + w_new[i] × new_chunk[i]
  w_old: 1.0 → 0.0  (线性衰减)
  w_new: 0.0 → 1.0  (线性增长)

非重叠区域:
  combined = smoothed_overlap + new_chunk[overlap_len:]
```

---

## 3. AAC + RTC 结合的七大优势

### 优势 1：动态响应能力 × 时序连续性 — 鱼与熊掌兼得

这是两者结合最核心的价值。

| 维度 | 单独 Action Chunking | AAC + RTC 结合 |
|------|---------------------|----------------|
| 精细操作时 | 固定 k 过大会导致"过冲" | AAC 自动减小 k → 高频重规划 |
| 平稳移动时 | 固定 k 过小会浪费推理资源 | AAC 自动增大 k → 低频高效推理 |
| Chunk 衔接处 | 可能产生跳变抖动 | Bezier + RTC 时间混合 → $C^1$ 连续 |

**本质：** AAC 负责 **"何时规划"**（temporal adaptation），RTC 负责 **"如何执行"**（spatial smoothing），两者在时间和空间两个维度上互补。

### 优势 2：异步并行 — 推理不再阻塞执行

传统同步模式：
```
[推理 80ms] → [执行 chunk (k×dt)] → [推理 80ms] → ...
               ↑ 执行期间推理线程空闲，推理期间机器人等待
```

AAC + RTC 异步模式：
```
推理线程: [推理1] [推理2] [推理3] ...
执行线程: [执行中...不断从 buffer 取帧...]
           ↑ 两个线程完全解耦，CPU/GPU 利用率最大化
```

**实际收益：**
- 推理频率可达 4-10 Hz（独立于控制频率 30-50 Hz）
- 即使单次推理耗时 100ms+，执行仍不受影响
- 通过 `latency_k` 裁剪 + 延迟估计补偿通信开销

### 优势 3：Chunk 间无缝过渡 — 消除机械臂抖动

单独使用 action chunking 时，新 chunk 的第一帧可能与正在执行的最后一帧有较大差异，导致机械臂产生不连续的"顿挫感"。

AAC + RTC 提供 **双层平滑保障**：

```
第一层（AAC Bezier）: 位置 + 速度连续
  P(t) = (1-t)³P₀ + 3(1-t)²t·P₁ + 3(1-t)t²·P₂ + t³P₃
  保证 chunk 边界的 C¹ 连续

第二层（RTC Temporal）: 重叠区域线性混合
  blended[i] = (1-α_i) × chunk_old[i] + α_i × chunk_new[i]
  平滑过渡，消除瞬时跳变
```

### 优势 4：VLA 大模型友好 — 适应可变推理延迟

π0.5 等 VLA 模型在真实场景中推理延迟波动较大（50ms ~ 300ms+），受网络、GPU 负载等因素影响。

RTC 的延迟估计机制：
```
pred_delay_steps = round(median(rtt_list) / control_dt)
```
将延迟转化为步数补偿，确保动作时序对齐。

AAC 的自适应 k 则进一步优化：
- 延迟高时 → 自动增大的 k 倾向（因为跨 chunk 跳变减小）
- 延迟低时 → 可灵活选择小 k 提高响应

两者配合使系统能 **自适应用推理延迟的波动**。

### 优势 5：EMA 逐帧平滑 — 执行层面的最后保障

除了 chunk 级别的 Bezier 和 RTC 混合，执行线程中还应用 EMA（指数移动平均）：

```python
smoothed_action = ema_alpha × raw_action + (1 - ema_alpha) × prev_action
```

三层平滑体系：
```
AAC Bezier (chunk 间 C¹ 连续)
    ↓
RTC Temporal (chunk 重叠线性混合)
    ↓
EMA (逐帧指数平滑)
    ↓
机器人执行
```

每一层解决不同粒度的问题，层层递进，确保最终动作极致平滑。

### 优势 6：学术界共识 — 多项同期工作验证此方向

2025-2026 年间，多个顶级会议/期刊的独立工作均指向同一结论：**自适应分块 + 实时平滑执行是 VLA 落地的关键路径**。

| 工作 | 出处 | 核心思路 |
|------|------|---------|
| **RTC** (Black et al.) | NeurIPS 2025 | 重叠混合 + 异步执行 |
| **Legato** (Liu et al.) | RSS 2026 | 学习原生 continuation 实现平滑 |
| **ABPolicy** (Yang et al.) | arXiv 2026.02 | 异步 B-Spline 流策略 |
| **TIDAL** (Sun et al.) | arXiv 2026.01 | 时间交错扩散与动作循环 |
| **Denoising-Variance Adaptive Chunking** (Feng et al.) | arXiv 2026.06 | 用去噪方差决定何时重规划 |
| **DiscreteRTC** (Wang et al.) | arXiv 2026.04 | 离散扩散作为天然异步执行器 |
| **StreamingVLA** (Shi et al.) | arXiv 2026.03 | 自适应提前观测 + 流式 VLA |
| **FASTER** (Lu et al.) | arXiv 2026.03 | 重新思考实时流 VLA |

这些工作的共同主题：**打破同步阻塞、自适应调整规划频率、保证动作时空连续性**。

### 优势 7：工程实用性 — 参数可调、配置灵活

AAC + RTC 的结合在工程实现上高度模块化和可配置：

```yaml
# AAC 参数
k_min: 30          # 最小 chunk size（精细操作）
k_max: 100         # 最大 chunk size（平稳移动）
k_init: 50         # 初始 chunk size
delta_high: 0.03   # 动作跳变上限阈值
delta_low: 0.005   # 动作跳变下限阈值

# RTC 参数
inference_rate: 4      # 推理频率 (Hz)
smooth_method: temporal # 平滑方式
min_smooth_steps: 8    # 最小平滑步数
latency_k: 0           # 延迟补偿步数
decay_alpha: 0.25      # 混合衰减系数
ema_alpha: 0.4         # 执行层 EMA 系数
```

每个参数都有明确的物理含义，可根据不同机器人和任务场景灵活调优。

---

## 4. 架构全景图

```
┌──────────────────────────────────────────────────────────────┐
│                    推理线程 (异步, ~inference_rate Hz)          │
│                                                              │
│  观测 → AdaptiveKSelector.get_k() → 动态 k                    │
│      → model.get_action(rtc_payload) → action_chunk [k步]     │
│      → AdaptiveKSelector.update() → 更新 k                   │
│      → Bezier 贝塞尔过渡 (chunk 间 C¹ 连续)                   │
│      → StreamActionBuffer.integrate() → RTC 时间混合          │
│      → 更新 prev_chunk + 延迟估计                              │
└──────────────────────┬───────────────────────────────────────┘
                       │ StreamActionBuffer (线程安全队列)
┌──────────────────────▼───────────────────────────────────────┐
│                    执行线程 (频率 ~1/control_dt Hz)             │
│                                                              │
│  buf.pop_next_action() → EMA 逐帧平滑 → robot.move()          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 总结

**AAC + RTC 结合的本质是：**

| 问题 | AAC 的贡献 | RTC 的贡献 |
|------|-----------|-----------|
| **何时规划？** | ✅ 自适应 k 调节 | — |
| **如何衔接？** | ✅ Bezier C¹ 连续 | ✅ 重叠线性混合 |
| **如何执行？** | — | ✅ 异步双线程 + 延迟补偿 |
| **如何平滑？** | 贝塞尔过渡 | 时间混合 + EMA |

> **一句话总结：AAC 让模型"聪明地决定什么时候想"，RTC 让机器人"平滑地执行想出来的动作"。两者结合，实现了从 VLA 模型到真实机器人部署的完整闭环。**

---

## 参考资料

1. Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023 / IJRR 2024.
2. Black, K., Galliker, M. Y., Levine, S. "Real-Time Execution of Action Chunking Flow Policies." NeurIPS 2025.
3. Zhao, T. Z., et al. "Action Chunking Transformers (ACT)." 2023.
4. Shi, Y., et al. "StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation." arXiv:2603.28565, 2026.
5. Feng, X., et al. "Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies." arXiv:2606.03847, 2026.
6. Liu, Y., et al. "Learning Native Continuation for Action Chunking Flow Policies (Legato)." RSS 2026.
7. Yang, F., et al. "ABPolicy: Asynchronous B-Spline Flow Policy for Real-Time and Smooth Robotic Manipulation." arXiv:2602.23901, 2026.
8. Sun, Y., et al. "TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control." arXiv:2601.14945, 2026.
9. Wang, P., et al. "DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors." arXiv:2604.25050, 2026.
10. Lu, Y., et al. "FASTER: Rethinking Real-Time Flow VLAs." arXiv:2603.19199, 2026.
11. Physical Intelligence. "π0.5: a VLA with Open-World Generalization." 2025.
