# AAC + RTC 参数详解与调试指南

> 本文档以理论和实践双重视角，系统讲解 AAC + RTC 联合推理中每个参数的含义、作用机制及调试思路。

---

## 目录

1. [系统架构速览](#1-系统架构速览)
2. [参数全景图](#2-参数全景图)
3. [AAC 参数详解](#3-aac-参数详解)
   - [3.1 k_min / k_max / k_init](#31-k_min--k_max--k_init)
   - [3.2 delta_high / delta_low](#32-delta_high--delta_low)
   - [3.3 k_step_up / k_step_down](#33-k_step_up--k_step_down)
   - [3.4 blend_steps](#34-blend_steps)
4. [RTC 参数详解](#4-rtc-参数详解)
   - [4.1 inference_rate](#41-inference_rate)
   - [4.2 min_smooth_steps](#42-min_smooth_steps)
   - [4.3 latency_k](#43-latency_k)
   - [4.4 decay_alpha](#44-decay_alpha)
   - [4.5 enable_rtc / mask_prefix_delay / max_guidance_weight](#45-enable_rtc--mask_prefix_delay--max_guidance_weight)
5. [执行层参数详解](#5-执行层参数详解)
   - [5.1 ema_alpha](#51-ema_alpha)
   - [5.2 control_dt](#52-control_dt)
6. [调试方法论](#6-调试方法论)
   - [6.1 调试优先级与流程](#61-调试优先级与流程)
   - [6.2 常见问题与对策表](#62-常见问题与对策表)
   - [6.3 逐参数调参顺序](#63-逐参数调参顺序)
   - [6.4 诊断信号解读](#64-诊断信号解读)
7. [理论背景补充](#7-理论背景补充)

---

## 1. 系统架构速览

```
┌──────────────────────────────────────────────────────────────────┐
│                   推理线程 (异步, ~inference_rate Hz)              │
│                                                                  │
│  Step 1: 获取观测 (img + state)                                   │
│  Step 2: AdaptiveKSelector.get_k() → 动态 k                      │
│  Step 3: model.get_action(rtc_payload) → action_chunk [k 步]      │
│  Step 4: k_selector.update(chunk) → 根据跨chunk跳变更新k          │
│  Step 5: Bezier 贝塞尔过渡 → chunk间 C¹ 连续                      │
│  Step 6: StreamActionBuffer.integrate() → RTC 重叠混合            │
│  Step 7: 更新 prev_chunk + 延迟估计                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │ StreamActionBuffer (线程安全队列)
┌──────────────────────────▼───────────────────────────────────────┐
│                   执行线程 (频率 = 1/control_dt Hz)                │
│                                                                  │
│  buf.pop_next_action() → EMA 逐帧平滑 → robot.move()              │
└──────────────────────────────────────────────────────────────────┘
```

**关键认知：** 推理线程和执行线程完全解耦。推理线程持续生产 action chunk 并推入缓冲区，执行线程以固定频率（如 25Hz）逐帧消费。缓冲区充当"蓄水池"。

---

## 2. 参数全景图

| 类别 | 参数 | 默认值 | 物理含义 | 调参敏感度 |
|------|------|--------|---------|-----------|
| **AAC** | `k_min` | 10 | chunk 最小步数 | ★★★ |
| **AAC** | `k_max` | 80 | chunk 最大步数 | ★★★ |
| **AAC** | `k_init` | 50 | chunk 初始步数 | ★★ |
| **AAC** | `delta_high` | 0.03 | 动作跳变高阈值 | ★★★★★ |
| **AAC** | `delta_low` | 0.005 | 动作跳变低阈值 | ★★★★★ |
| **AAC** | `k_step_up` | 5 | k 增大步长 | ★★ |
| **AAC** | `k_step_down` | 10 | k 减小步长 | ★★ |
| **AAC** | `blend_steps` | 20 | Bezier过渡步数 | ★★★ |
| **RTC** | `inference_rate` | 8 | 推理频率(Hz) | ★★★★ |
| **RTC** | `min_smooth_steps` | 20 | 最小平滑步数 | ★★★ |
| **RTC** | `latency_k` | 4 | 延迟裁剪步数 | ★★★★ |
| **RTC** | `decay_alpha` | 0.25 | 指数衰减系数 | ★★ |
| **RTC** | `enable_rtc` | true | RTC引导开关 | ★★★ |
| **RTC** | `mask_prefix_delay` | false | 前缀延迟掩码 | ★ |
| **RTC** | `max_guidance_weight` | 0.5 | RTC guidance 权重 | ★★ |
| **执行** | `ema_alpha` | 0.3 | 逐帧EMA平滑 | ★★★★ |
| **执行** | `control_dt` | 0.04 | 控制周期(秒) | ★★★ |

---

## 3. AAC 参数详解

### 3.1 k_min / k_max / k_init

**理论含义：**

`k` 是每次推理产生的动作块（action chunk）的步数。在 action chunking 范式中，模型一次预测未来 $k$ 步的动作序列：

$$\text{action\_chunk} = [a_0, a_1, a_2, ..., a_{k-1}]$$

- **$k$ 越大** → 轨迹段越长 → 动作更平滑、推理开销更低 → 但环境突变时响应滞后
- **$k$ 越小** → 轨迹段越短 → 响应更快、重规划更频繁 → 但可能引入高频抖动、推理负载更高

**物理对应：**

在 `control_dt = 0.04s`（25Hz 控制频率）下：
- `k=10` → 预测 0.4 秒的未来轨迹
- `k=80` → 预测 3.2 秒的未来轨迹

**实践建议：**

| 场景 | k_min | k_max | k_init |
|------|-------|-------|--------|
| 精细操作（插拔、抓取） | 5~10 | 40~60 | 20~30 |
| 大范围移动（搬运） | 15~30 | 80~120 | 50~60 |
| 动态避障 | 3~8 | 30~50 | 15~25 |

---

### 3.2 delta_high / delta_low

**这是整个 AAC 系统最核心的两个参数。**

**理论含义：**

`delta` 定义为上一个 chunk 的最后一帧与本 chunk 的第一帧之间的欧氏距离（作用于归一化后的动作空间）：

$$\Delta = \| a_{\text{new}}[0] - a_{\text{old}}[-1] \|_2$$

自适应调节逻辑：

```
if Δ > delta_high:   k ← max(k_min, k - k_step_down)   # 动作突变 → 减小k
elif Δ < delta_low:  k ← min(k_max, k + k_step_up)     # 动作平稳 → 增大k
else:                k 保持不变                          # 过渡区间
```

**物理直觉：**

$\Delta$ 本质上衡量的是"策略在当前时刻有多大改变"：
- **大 $\Delta$**：模型认为之前的动作方向不再适用（如遇到障碍物、到达目标附近需要精细操作），需要更频繁地重规划 → 减小 k
- **小 $\Delta$**：模型认为当前轨迹稳定，可以大胆预测更远的未来 → 增大 k

**实践建议：**

$\Delta$ 的大小取决于动作空间的数值尺度。在当前的 Pi0.5 实现中，动作已被归一化（joint positions 等）。

调试步骤：
1. **先跑一次，记录 $\Delta$ 值**：在 `AdaptiveKSelector.update()` 中已经打印了 `Δ=` 值，观察其典型范围
2. **设置 delta_low**：取平稳移动时 $\Delta$ 的 75 分位数
3. **设置 delta_high**：取方向切换/精细操作时 $\Delta$ 的 25 分位数
4. **经验法则**：两者之间至少保持 3~5 倍差距，形成明显的"迟滞带"

| 问题现象 | 可能原因 | 调整方向 |
|---------|---------|---------|
| k 一直停在 k_min | delta_low 太高，$\Delta$ 永远降不下来 | ↓ 降低 delta_low |
| k 一直停在 k_max | delta_high 太低，$\Delta$ 永远触不到高阈值 | ↑ 提高 delta_high |
| k 频繁抖动 | delta_low 和 delta_high 间距太小 | ↑ 拉大两者差距 |
| k 几乎不变 | 阈值设置过极端 | 参考 $\Delta$ 实际分布重新设置 |

---

### 3.3 k_step_up / k_step_down

**理论含义：**

控制 k 每次变化的幅度。通常 `k_step_down > k_step_up`（减小比增大更快），体现的是"安全优先"原则——遇到突变快速反应，恢复平稳时缓慢增加。

**实践建议：**

- 典型比值 `k_step_down : k_step_up ≈ 2:1`
- 步长过大会导致 k 在极端值之间跳跃，失去"自适应"的平滑性
- 步长过小会导致 k 调整太慢，跟不上场景变化

| k_max - k_min | 建议 k_step_up | 建议 k_step_down |
|---------------|----------------|------------------|
| 20~30 | 3~5 | 5~8 |
| 40~60 | 5~8 | 10~15 |
| 70~100 | 8~10 | 15~20 |

---

### 3.4 blend_steps

**理论含义：**

`blend_steps` 控制 Bezier 贝塞尔过渡覆盖的步数。Bezier 过渡在 chunk 边界处对前 $n$ 步进行重映射，保证位置和速度的 $C^1$ 连续性。

三次贝塞尔公式：

$$P(t) = (1-t)^3 P_0 + 3(1-t)^2 t \cdot P_1 + 3(1-t) t^2 \cdot P_2 + t^3 P_3$$

其中 $P_0$ 是上一 chunk 最后一帧，$P_3$ 是本 chunk 第 n 帧，$P_1, P_2$ 由各自速度方向决定：

$$P_1 = P_0 + \frac{n}{3} \cdot v_0, \quad P_2 = P_3 - \frac{n}{3} \cdot v_3$$

**物理含义：**

`blend_steps` 决定了"过渡带"的宽度。在这个区间内，旧 chunk 的轨迹被贝塞尔曲线逐渐替换为新 chunk 的轨迹。

**实践建议：**

| blend_steps | 效果 |
|-------------|------|
| 太小 (5~10) | 过渡急促，可能仍有轻微顿挫感 |
| 适中 (15~30) | 平滑且不显著偏离模型意图 |
| 太大 (>40) | 过度平滑，可能"抹掉"模型的有意义动作，响应变迟钝 |

**与 RTC 的关系：** Bezier 过渡运行在推理线程中（chunk 级别），RTC 的时间混合运行在 StreamActionBuffer 中（帧级别），两者叠加形成双层平滑。通常 `blend_steps` 设置得比 `min_smooth_steps` 稍小一些，让 Bezier 处理 chunk 边界的粗粒度衔接，RTC 处理细粒度的帧间混合。

---

## 4. RTC 参数详解

### 4.1 inference_rate

**理论含义：**

推理线程每秒触发的推理次数。这是独立于控制频率的元参数：
- 执行线程以 `1/control_dt` Hz 运行（如 25Hz）
- 推理线程以 `inference_rate` Hz 运行（如 8Hz）

**供需关系分析：**

每次推理产生 $k$ 步动作，推入缓冲区。执行线程以 $1/\text{control\_dt}$ Hz 消费。

$$\text{生产速率} = k \times \text{inference\_rate} \quad \text{步/秒}$$
$$\text{消费速率} = 1 / \text{control\_dt} \quad \text{步/秒}$$

**实践建议：**

为保证缓冲区不枯竭，需满足：

$$k_{\text{avg}} \times \text{inference\_rate} \geq 1 / \text{control\_dt}$$

例如：`control_dt=0.04`（25Hz），`k_avg≈50`，则 `inference_rate ≥ 0.5Hz` 即可。但实际中还需考虑：
1. 推理延迟：每次推理可能耗时 50~200ms，`inference_rate` 不能超过 `1/推理延迟`
2. GPU 负载：过高频率浪费算力

| control_dt | k 范围 | 建议 inference_rate |
|------------|--------|-------------------|
| 0.033 (30Hz) | 30~60 | 3~6 |
| 0.04 (25Hz) | 30~80 | 4~8 |
| 0.05 (20Hz) | 40~100 | 3~6 |

---

### 4.2 min_smooth_steps

**理论含义：**

StreamActionBuffer 在整合新旧 chunk 时，确保重叠平滑区域至少有 `min_smooth_steps` 步。如果当前缓冲区剩余步数不足 `min_smooth_steps`，会自动用最后一帧补足。

**平滑机制：**

```
重叠区:
  blended[i] = w_old[i] × old_chunk[i] + w_new[i] × new_chunk[i]
  w_old[i] = 1.0 → 0.0 (线性衰减)
  w_new[i] = 0.0 → 1.0 (线性增长)
```

**物理含义：**

在重叠区，旧 chunk 的"权重"从 100% 线性降到 0%，新 chunk 的权重从 0% 线性升到 100%。`min_smooth_steps` 越大 → 过渡越平缓，但也意味着模型意图的生效滞后越大。

**实践建议：**

- 对精细操作：`min_smooth_steps = 10~20`（快速响应优先）
- 对大范围移动：`min_smooth_steps = 20~40`（平滑优先）
- 典型值：约为 `k_init` 的 30%~50%

---

### 4.3 latency_k

**理论含义：**

推理延迟导致"时间错位"——当新 chunk 到达时，机器人已经执行了若干步旧动作。`latency_k` 从新 chunk 开头丢弃前 k 步，同时旧 chunk 中已执行的步数自然被消费，使得新旧 chunk 在时间轴上对齐。

**自动延迟估计模式（当前实现）：**

```python
median_rtt = np.median(rtt_list)          # 中位数推理延迟
pred_delay_steps = round(median_rtt / control_dt)  # 转换为步数
```

然后将 `pred_delay_steps` 传给 RTC payload，模型侧可用此信息做内部分块对齐。同时 `latency_k` 参数作为 `max_k` 传入 `integrate_new_chunk()`，控制物理丢弃步数上限。

**实践建议：**

- 观察日志中的 `推理耗时` 和 `延迟中位数`，将 `latency_k` 设置为 `median_rtt / control_dt` 的 1.0~1.5 倍
- 如果 `latency_k` 太大，会浪费模型预测的有效动作
- 如果 `latency_k` 太小，新旧 chunk 在时间上错位，可能导致抖动

| 典型推理延迟 | control_dt | 建议 latency_k |
|-------------|-----------|---------------|
| 50~80ms | 0.04 | 2~3 |
| 80~150ms | 0.04 | 3~5 |
| 150~250ms | 0.04 | 5~8 |

---

### 4.4 decay_alpha

**理论含义：**

StreamActionBuffer 内部保留的指数衰减系数，控制平滑过渡时对历史动作的"记忆"强度。值越大，新 chunk 的影响越快占主导。

目前 `decay_alpha` 在 `StreamActionBuffer.__init__()` 中保存，但实际的平滑逻辑使用的是线性权重 `np.linspace(1.0, 0.0, overlap_len)`。`decay_alpha` 主要用于兼容性保留和未来扩展（如指数加权变体）。

**当前实践：** 该参数对行为影响较小，保持默认值 `0.25` 即可。

---

### 4.5 enable_rtc / mask_prefix_delay / max_guidance_weight

**理论含义：**

这是 RTC 论文中的核心机制——将已执行的动作前缀发回给模型，让模型在新一轮推理时"知道"机器人已经做了什么。

- **`enable_rtc`**：总开关。设为 `false` 时退化为普通 action chunking（无 prev_chunk 引导）
- **`mask_prefix_delay`**：是否对已执行前缀做延迟掩码。当推理延迟较大时，前缀可能已经"过期"
- **`max_guidance_weight`**：RTC guidance 的最大权重（0~1）。控制模型在生成新 chunk 时受已执行前缀"约束"的程度

**物理直觉：**

RTC 引导类似于告诉模型："你上次说要做 [a_0, a_1, ..., a_{k-1}]，机器人已经执行到了 a_m，请在此基础上继续规划。"这显著减少了 chunk 之间的不一致。

**实践建议：**

| 参数 | 保守设置 | 激进设置 |
|------|---------|---------|
| `enable_rtc` | `true`（推荐始终开启） | — |
| `mask_prefix_delay` | `false` | `true`（高延迟场景） |
| `max_guidance_weight` | 0.3~0.5 | 0.6~0.8 |

---

## 5. 执行层参数详解

### 5.1 ema_alpha

**理论含义：**

指数移动平均（EMA）在逐帧执行中平滑动作：

$$a_{\text{smoothed}} = \alpha \cdot a_{\text{raw}} + (1-\alpha) \cdot a_{\text{prev}}$$

- $\alpha = 1.0$：不平滑，直接使用原始动作（响应最快但可能抖动）
- $\alpha = 0.0$：完全平滑，动作永远不变（无意义）
- $\alpha = 0.3$：70% 权重给上一帧，30% 给当前帧（偏平滑）

**这是三层平滑体系的最底层**，直接作用在发送给机器人的每个动作上：

```
AAC Bezier (chunk 间 C¹ 连续) → RTC Temporal (重叠线性混合) → EMA (逐帧指数平滑) → 机器人
```

**实践建议：**

| 场景 | 建议 ema_alpha | 原因 |
|------|---------------|------|
| 精细操作 | 0.5~0.7 | 需要快速响应，不能太"肉" |
| 大范围移动 | 0.2~0.4 | 轨迹平滑优先 |
| 默认 | 0.3~0.4 | 折中 |

**信号解读：** 如果观察到机器人动作"慢半拍"或"跟不上"，增大 `ema_alpha`；如果关节抖动明显，减小 `ema_alpha`。

---

### 5.2 control_dt

**理论含义：**

机器人控制周期（秒）。$\text{control\_dt} = 0.04s$ 意味着每秒发送 25 个动作给机器人。

**与其他参数的关系：**

- `k` 的实际时间跨度 = $k \times \text{control\_dt}$ 秒
- `inference_rate` 上限受限于 $1/\text{control\_dt}$
- `latency_k` 的计算依赖 $\text{control\_dt}$

**实践建议：**

- 大多数机器人控制器支持 20~50Hz（control_dt = 0.02~0.05）
- 太小（如 0.01）会增加通信负载且机器人未必能跟上
- 太大（如 0.1）会导致动作阶梯感明显
- `0.033~0.04`（25~30Hz）是最常用的区间

---

## 6. 调试方法论

### 6.1 调试优先级与流程

```
Layer 0: 基础连通性
  ├── 模型能正常加载推理
  ├── 机器人能正常 move
  └── 摄像头数据正常

Layer 1: 固定 k 基准 (关闭 AAC + RTC, k=固定值)
  ├── 确认"裸"推理的轨迹质量
  └── 记录推理延迟、动作数值范围

Layer 2: 仅 RTC (固定 k, 开启 RTC)
  ├── 调整 inference_rate, latency_k, min_smooth_steps
  └── 目标: chunk 间平滑过渡

Layer 3: 仅 AAC (关闭 RTC, 开启 AAC)
  ├── 观察 Δ 分布 → 设定 delta_high/delta_low
  └── 目标: k 能随场景自适应变化

Layer 4: AAC + RTC 联合
  ├── 微调 ema_alpha, blend_steps
  └── 目标: 端到端最佳体验

Layer 5: 极限调优
  ├── 针对特定任务微调所有参数
  └── 在多个任务上验证泛化性
```

---

### 6.2 常见问题与对策表

| # | 问题现象 | 根本原因分析 | 调试步骤 | 参数调整方向 |
|---|---------|------------|---------|------------|
| 1 | **机器人动作抖动/震颤** | 噪声未经充分平滑 | ① 减小 ema_alpha (0.3→0.2) ② 增大 blend_steps ③ 增大 min_smooth_steps | `ema_alpha↓`, `blend_steps↑`, `min_smooth_steps↑` |
| 2 | **机器人动作"肉"/滞后** | 平滑过度，响应迟钝 | ① 增大 ema_alpha (0.3→0.5) ② 减小 blend_steps ③ 增大 inference_rate | `ema_alpha↑`, `blend_steps↓`, `inference_rate↑` |
| 3 | **Chunk 切换时顿挫** | 新旧 chunk 边界不连续 | ① 确认 Bezier blend 生效 ② 增大 blend_steps ③ 确认 RTC smooth_method="temporal" | `blend_steps↑`, 确认 `smooth_method=temporal` |
| 4 | **缓冲区频繁枯竭（动作停顿）** | 推理速度跟不上消费速度 | ① 观察日志中 buffer size ② 降低 control_dt 或增大 inference_rate ③ 增大 k_min | `inference_rate↑`, `k_min↑`, `control_dt↑` |
| 5 | **k 一直卡在 k_min** | delta_low 太高，Δ 降不下来 | ① 观察日志中的 Δ 值 ② 降低 delta_low ③ 检查是否动作空间尺度问题 | `delta_low↓` |
| 6 | **k 一直卡在 k_max** | delta_high 太低，Δ 触不到高阈值 | ① 观察日志 Δ 值范围 ② 提高 delta_high | `delta_high↑` |
| 7 | **到目标附近"刹不住车"** | k 太大导致"过冲"，来不及重规划 | ① 降低 k_max ② 降低 delta_low（更容易触发 k 减小） ③ 增大 k_step_down | `k_max↓`, `delta_low↓`, `k_step_down↑` |
| 8 | **动作"一步一卡"** | 推理延迟过大，latency_k 设置不当 | ① 观察推理耗时中位数 ② 调整 latency_k ③ 如延迟>200ms，考虑模型优化 | `latency_k` 设为 `median_rtt/control_dt` |
| 9 | **ema_alpha 调了没效果** | 可能有缓冲或 EMA 位置不对 | ① 确认 ema_alpha 赋值到 model._ema_alpha ② 检查 smooth_action() 是否被调用 | 检查代码流程 |
| 10 | **不同任务表现差异大** | AAC 阈值不适配特定任务 | ① 分别在每个任务上记录 Δ 分布 ② 为每个任务设置独立配置 | 按任务定制 delta_high/delta_low |

---

### 6.3 逐参数调参顺序

按照"从粗到细、从关键到次要"的原则：

```
Step 1: control_dt → 固定 (0.04)
Step 2: k_init → 固定 (40~60)
Step 3: ema_alpha → 先粗调 (0.2 vs 0.5 感受差异, 定方向)
Step 4: delta_high/delta_low → 核心调参 (基于 Δ 分布)
Step 5: inference_rate → 基于推理延迟设定
Step 6: latency_k → 基于延迟中位数设定
Step 7: blend_steps → 精细调整过渡平滑度
Step 8: min_smooth_steps → 精细调整 RTC 平滑
Step 9: k_min/k_max → 设定安全边界
Step 10: k_step_up/k_step_down → 微调 k 变化速率
Step 11: max_guidance_weight → 微调 RTC 引导强度
```

---

### 6.4 诊断信号解读

运行时，日志会输出以下关键信息：

```
[AAC_RTC] chunk_round: 42 | k: 35 | 推理耗时: 87ms | 延迟中位数: 82ms | buffer: 18
[AAC] Δ=0.0123 ∈ [0.005, 0.03] → k = 35 (hold)
```

| 信号 | 健康范围 | 预警信号 | 含义 |
|------|---------|---------|------|
| `buffer size` | 5~30 | < 3（即将枯竭） | 推理生产跟不上消费 |
| `buffer size` | 5~30 | > 50（堆积严重） | 推理过度，浪费算力 |
| `推理耗时` | 50~150ms | > 200ms | 模型推理太慢，可能影响实时性 |
| `Δ 值` | 0.001~0.1 | 波动极小(<0.001) | 可能是 dead band 或动作空间太小 |
| `Δ 值` | 0.001~0.1 | 波动极大(>0.5) | 可能是噪声或策略不稳定 |
| `k 变化频率` | 每 3~10 轮变一次 | 每轮都变 | 阈值设置过于敏感 |
| `k 变化频率` | 每 3~10 轮变一次 | 一直不变 | 阈值设置过于极端 |

---

## 7. 理论背景补充

### 7.1 为什么需要三层平滑？

| 层级 | 作用域 | 解决的问题 | 方法 |
|------|--------|-----------|------|
| L1: Bezier | Chunk 间 | 新 chunk 开头与旧 chunk 结尾的硬跳变 | 三次贝塞尔插值（C¹ 连续） |
| L2: RTC Temporal | Chunk 重叠区 | 新旧 chunk 在时间维度的权重过渡 | 线性权重混合 |
| L3: EMA | 逐帧 | 传感器噪声、模型推理的随机波动 | 指数移动平均 |

三层平滑作用于不同的时间粒度，互补而非冗余。去掉任何一层都会在相应粒度上引入不连续性。

### 7.2 AAC 调节逻辑的数学本质

AAC 本质上是一个**带迟滞的 bang-bang 控制器**，以 $\Delta$ 为反馈信号，k 为被控量：

$$k_{t+1} = \begin{cases} \max(k_{\min}, k_t - k_{\text{step\_down}}) & \text{if } \Delta > \delta_{\text{high}} \\ \min(k_{\max}, k_t + k_{\text{step\_up}}) & \text{if } \Delta < \delta_{\text{low}} \\ k_t & \text{otherwise} \end{cases}$$

其中 $\delta_{\text{low}} < \delta_{\text{high}}$ 形成迟滞带，避免 k 在临界值附近振荡。

### 7.3 RTC 异步架构的理论优势

从排队论角度，异步架构将"推理"和"执行"解耦为两个独立的队列：
- 传统同步：$\text{吞吐量} = \min(\text{推理速率}, \text{执行速率})$
- RTC 异步：$\text{吞吐量} = \text{执行速率}$（推理速率只需满足不低于消费速率即可）

这意味着推理延迟的波动只影响 buffer 深度，不影响执行连续性。

---

## 参考资料

- 本文档中引用的所有理论、公式与实现均基于本仓库 `src/robot/policy/pi05/` 中的源码
- AAC 源码: `deploy_pi05_real_aac.py`, `deploy_pi05_real_aac_rtc.py`
- RTC 源码: `deploy_pi05_real_rtc.py`, `deploy_pi05_real_aac_rtc.py`
- 核心模型实现: `inference_model.py` (含 Bezier blend、EMA 平滑)
- 配置文件: `deploy_pi05_real_aac_rtc.yml`, `deploy_pi05_real_rtc.yml`
