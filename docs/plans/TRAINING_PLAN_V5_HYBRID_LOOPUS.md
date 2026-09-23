# V5 训练草案：LoopUS × Qwen3.5 Hybrid（边界实测、块边界主线）

> 版本：V5，2026-09-19 起草，2026-09-20 定稿
> 状态：规划文档，**只完成了底模核查；未启动 V5 训练或测量**。
> 前置：`PROJECT_REVIEW.md`、`LOOPUS_NOTES.md`、`TRAINING_PLAN_V4_HYBRID.md`。
> 目标：以 Qwen3.5-4B 为最小验证模型，为 Qwen3.8-27B 训练循环深度提供可靠配方。

---

## 0. 本版相对 V4 的修订

V4 的方向仍然保留，但本版修正了三个容易混淆的层面：

1. **模块边界不是动力学边界**：3:1 模块只用于代码组织、缓存分类和实验描述，
   不直接决定 encoder/reasoning/decoder 的切分。切分必须由修正后的表征动力学测量决定。
2. **LoopUS 官方门控作为主线**：LoopUS 源码的 `ReasoningBlock.forward()` 在整个推理块
   完成后执行一次选择门（`loopus/models/modeling_lds.py:411–435`）。因此 V5 的主案是
   **块边界单门**；用户草案提出的“每个注意力层之后门控”保留为 hybrid 专用对照，不再
   事先宣称它更优。
3. **旧测量不能直接当验收结果**：旧 `s1a_dynamics.py` 存在末层双重 norm、层转移分组
   整体错位等问题；旧 `eval_loopus_g0.py` 的 ARC continuation logits 位置也错位。
   这些旧数字只作为诊断线索，必须修正代码后重测。

---

## 1. 目标与可验证问题

### 1.1 终极目标

把 Qwen3.8-27B（同类 hybrid 架构）训练成可稳定循环推理的模型：循环深度增加时，
推理质量不下降且在 reasoning 任务上获得测试时计算增益。

### 1.2 本机阶段只验证机制，不承诺达到 27B 绝对分数

Qwen3.5-4B 本机实验的成功标准不是“超过原模型所有指标”，而是回答：

> 在同一批样本、同一层块、同一评分协议下，经过 LoopUS 式训练后，
> CE(d) 是否随循环深度 d 下降或保持稳定？最终严格 EM(d) 是否至少有一个
> reasoning 基准呈现可重复的正增益？

### 1.3 三个核心问题

| 编号 | 问题 | 证据 |
|---|---|---|
| Q1 | 表征动力学边界在哪里？3:1 架构周期是否与动力学峰锁相？ | 修正后的 S1a |
| Q2 | LoopUS 块边界单门是否足以稳定 hybrid 循环？ | A2 主线 vs A4 无门 |
| Q3 | hybrid 专用的“只在全注意力层后门控”是否优于块边界门？ | A3 对照 |

---

## 2. 已知底模事实与范围

`Qwen/Qwen3.5-4B-Base` 已完成配置核查：

| 项 | 已核实事实 |
|---|---|
| 文本层数 | 32 |
| hidden size | 2560 |
| `layer_types` | 24 `linear_attention` + 8 `full_attention` |
| 架构周期 | 每 4 层严格为 3 linear + 1 full |
| 模块数量 | 8 个 4 层模块 |
| 模型封装 | 多模态 `Qwen3_5ForConditionalGeneration`，文本主干在 `language_model` |
| transformers | `loopus_env` 的 5.3.0 原生支持 `qwen3_5_text` |
| 权重 | 已下载 8.68 GiB；本机加载 spike 已通过，显存需另测 |

注意：架构周期是**设计事实**，不是动力学周期。后文所有边界结论必须由 S1a
数据产生。

### 2.1 Qwen3.8-27B 架构核实（2026-09-20 提前完成，迁移清单第一项）

官方 config.json（HF `Qwen/Qwen3.8-27B`，存档 `q38_27b_config.json`）：

| 项 | Qwen3.5-4B（本机代理） | Qwen3.8-27B（迁移目标） | 同构性 |
|---|---|---|---|
| 模型类型 | `qwen3_5` / `Qwen3_5ForConditionalGeneration` | **完全相同** | ✅ |
| 层数 | 32 | 64 | 2× |
| 层类型 | 24 linear + 8 full，严格 3:1 | 48 linear + 16 full，严格 3:1 | ✅ 同周期 |
| full_attention_interval | 4 | 4 | ✅ |
| hidden size | 2560 | 5120 | 2× |
| 原生上下文 | — | 262,144（256K） | — |
| GQA | — | 24 heads / 4 KV heads | — |
| **loop/recurrent-depth 原生块** | 无 | **无**（config 无任何 loop/recur/depth/iter 键） | ✅ 同为 retrofit 场景 |

**结论：27B 与 4B 代理是同架构家族（同 model_type、同 3:1 周期、同层类型），
“27B 本身有循环块”的假设不成立**——它有的只是和 4B 同样的 DeltaNet 序列维递归，
深度循环块同样需要人造 + A2 门训练。这使 4B 实验的迁移强度远高于 LoopUS 的
标准→混合跨越：方法、门控设计、B/K 配方、评分协议全部可直接移植，**唯一必须
重新测量的是切分位置**（64 层的平台区不会在数值上等于 32 层的 L8-19；按相对深度
外推的候选 ~L16-39 仅为假设，须实测）。门控参数随 hidden×2 增至约 3.3M，仍可忽略。

**相关先例（不同路线，同一问题）**：社区模型 `Jackrong/Qwopus3.8-27B-Flash`
（HF，2026-09）针对 27B 的“停不下来的思考”用 **SFT（1.5M 教师样本取 top 10%）+
NeMo-RL/GSPO 强化**压短可见思维链：输出字符 −9.9%、解码 +12.8%，代价
MMLU-Pro −1.45pp（92.73→91.28，作者自报本地结果，E1 级）。它走的是“缩短显式
CoT”路线，与本项目“隐空间循环替代显式 CoT”是同一问题的两种解法——若 L0 验证
循环可行，其价值在于用隐式计算换 token 而不付显式截断的质量税。

---

## 3. 架构方案

### 3.1 三段结构：边界数据驱动

定义：

- Encoder `E`：只执行一次；
- Reasoning block `M`：循环执行 B 次；
- Decoder `D`：循环结束执行一次。

形式：

```text
h0 = E(x)
for b in 1..B:
    h_b = Gate(M(h_{b-1}), h_{b-1})
logits = D(h_B)
```

候选块不能由“前 4 个模块/中间 2 个模块”直接指定。S1a 先测：

- 每个真实 layer 的 post-layer hidden state；
- 相邻真实 layer 的 cosine distance；
- 各层 logit-lens CE（明确排除末层 final norm 的重复计算）；
- 实际 `layer_types[i]` 对应的转移类别。

S1a 通过拟合或稳健的分段变化检测给出边界候选；S1b 对候选块做冻结循环轨迹测试，
最后才定案。模块只能作为候选的描述单位。

### 3.2 门控主线与对照

#### A2：LoopUS 忠实主线（默认主案）

在整个 `M` 完成一次前向后施加一枚选择门：

```text
proposal = M(h_prev)
h_next = alpha(h_prev, proposal) * proposal
        + (1 - alpha(h_prev, proposal)) * h_prev
```

门控采用 LoopUS 的 Mamba/S4D 风格：

```text
delta = softplus(W_delta(W_in(proposal - h_prev)) + bias)
A      = -exp(A_log)
alpha  = exp(delta * A)        # 每 token/每 channel，0 < alpha < 1
```

官方 LoopUS 实现的 `SelectiveGate` 参数预算以 `dt_rank=ceil(d/16)` 为默认；d=2560、
`dt_rank=160` 时，一枚 gate 约 824,320 个参数，另加 confidence head 约 7,681 个。

#### A3：hybrid 专用对照（用户草案主意）

只在推理块内部真实 `layer_types[i] == "full_attention"` 的层之后施加选择门；
`linear_attention` 层不加外门。每轮门的数量等于候选 `M` 中实际 full-attention 层数；
在 4B 的草案候选 16–23 中才是约 2 枚，不能把 2 枚写成架构不变的常数。它的依据仅是
层类型与 P2 测量，**不是先验真理**。A3 若优于 A2，才把它作为 Qwen3.8-27B 的 hybrid
特化方案。

#### A4：无外部门控

直接循环 `M`，用于测量 DeltaNet 内置状态机制是否足以稳定深度递推。A4 只在 S1b
轨迹没有发散、或作为明确失败对照时运行。

### 3.3 KV / recurrent state 语义

不能把 DeltaNet 简化成“没有 cache”：Qwen3.5 的动态缓存包含：

- full attention：`key_cache/value_cache`，沿序列长度增长；
- linear attention：定形的 `conv_states/recurrent_states`，不是 full-attention KV，
  但仍是必须正确初始化、更新和隔离的递归状态。

深度循环时同一 token 的 hidden state 已改变，不能跨 depth 盲目复用上一轮的状态。
训练阶段不使用 generation-style KV cache；每轮对完整训练序列前向，状态生命周期由该轮
前向结束决定。推理生成时，prefix cache 应按“生成步 × recursion step”正确隔离；
LoopUS 的 KV cache 加速结果需要在 hybrid 上单独复现，不能直接继承。

---

## 4. 损失与训练

每个被随机采样的监督深度 b 计算：

```text
L_LM   = CE(logits_b, next_token_labels)
L_mono = SiLU(L_LM(b) - stop_grad(L_LM(b-1)))
L_Q    = BCEWithLogits(q_b, token_accuracy_b)
L      = L_LM + beta * L_mono + L_Q
```

训练循环遵循 LoopUS 官方逻辑：

- 每个 batch 展开 B 个 depth iterations；
- 随机采样 K 个 iteration 回传梯度；
- 非监督 iteration 使用 `no_grad()` 并 detach；
- 默认先用官方的 B=20、K=5、ctx=1024 作为服务器配方；本机由 S2 吞吐实测后降级。

### 4.1 本机训练级别

| 级别 | 底模 | 可训练参数 | 本机目的 |
|---|---|---|---|
| L0 | 4B 4bit | 只训 A2/A3 gate + q head | 首个可行性 spike，先看 CE(d) 是否分化 |
| L1 | 4B 4bit | L0 + reasoning block LoRA | 方向性训练，token 预算按 S2 实测定（预计 20–50M） |
| L2 | 更小的 2B/0.8B Base（若需要） | 可增加 LoRA / 全参小头 | 本机训练吞吐/稳定性验证 |
| Server | Qwen3.8-27B | 全 LoopUS/Hybrid 方案 | 3B token 级正式训练 |

4B 全参 AdamW 不属于本机方案：权重、梯度和优化器状态已超出 8GB。任何“本机成功”
必须标注为 gate/head-only 或 LoRA 结果，不能冒充 LoopUS 论文的全量训练。

---

## 5. 执行阶段与结论回填

### S0：底模与环境核查 — 已完成

config、下载、transformers 5.3 兼容、bf16 加载此前已通过；4-bit spike 于 2026-09-20 完成。

> **结论（2026-09-20 回填，`results/s0_hybrid_spike.json`，verdict=PASS）**：
> - NF4（bnb 0.50.2，double-quant）在 transformers 5.3.0 下加载正常，返回 `Qwen3_5ForCausalLM`
>   文本主干；32 层、24 linear_attention + 8 full_attention，模块层类型与 config 交叉验证一致；
> - ctx 512 前向峰值 **3.0 GiB**，训练态反向峰值 **3.78 GiB**——8GB 卡余量充足；
> - 梯度成功穿过冻结 4-bit 主干进入 fp32 probe（grad absmax 0.021，有限非零），基座可训练
>   参数为 0 → **门控/LoRA 训练的先决条件成立，L0/L1 路线可行**；
> - 环境注意：DeltaNet 层当前走 torch 慢速路径（未装 flash-linear-attention / causal-conv1d），
>   512 ctx 单次前向约 3.6–8.7s。测量可接受；**S3 训练前建议安装这两个包**，否则吞吐受限。

### S1a：正确的表征动力学测量 — 已完成（v2 定案）

必须修正旧测量的四个问题：

1. `hidden_states` 的 index 要对应真实 layer；embedding 不能混入普通 layer；
2. 末层 final norm 不得被重复施加；末层距离单独标为“含 final norm 的出口点”；
3. `layer_types[i]` 按真实 config 分组，不得用 `i % 4` 硬编码标准模型；
4. 记录每条文本而不只是跨文本平均；P2 的统计独立单位是文本/transition，不能把
   文本均值后的小点数冒充独立样本。

S1a 输出：

- `results/s1a_dynamics_v2.json`：逐文本、逐层 distance/CE/raw layer type；
- P1：周期性/锁相是否真实；
- P2：linear→linear、linear→full、full→linear 三类距离及效应量/置信区间；若目标配置出现
  full→full，则单独保留该组，不能并入“含注意力”总组；
- P3：注意力峰与 logit-lens CE 的宏观变化位置；
- fit：稳健分段模型的边界候选，而非只取二阶差分 top-k。

**采集与统计约定**：优先用显式逐层 forward/hook 记录 `h_emb`、每层 post-layer
`h_i` 和最终 `final_norm(h_L)`，不依赖不同 transformers 版本对 `output_hidden_states`
末项语义的猜测。余弦距离只在 raw `h_i → h_{i+1}` 上计算；final norm 作为独立出口点，
不得再次作用于已经 normalized 的张量。logit-lens 必须在 JSON 中写明是“每层统一应用一次
同一 norm”还是“原始 hidden 的线性探针”，全程只能选一种比较口径。

每条文本是一个统计簇：先在文本内汇总各 transition，再以文本为 bootstrap 单位（或使用
带文本随机效应的模型）给出 P2 的效应量和区间。不能把同一文本的多个层转移当作独立样本，
也不能先跨文本平均成一条曲线后再进行 Mann–Whitney 检验。P3 的宏观边界用 1/2/3 段
piecewise fit 比较 AIC/BIC 或 held-out error，并报告边界 bootstrap 区间；二阶差分 top-k
只能作为可视化线索。

> **S1a-v2 结论（2026-09-20 回填，`results/s1a_dynamics_v2.json`，脚本 `s1a_dynamics_v2.py`；旧 v1 结果文件未覆盖）**
>
> 测量口径：hooks 采集原始逐层隐状态（h_0=embeddings，h_{i+1}=layer i 输出）；final norm 仅在
> logit-lens 中每阶段施加一次；层类型取自模块属性，与 config 交叉验证一致；8 条 WikiText 文本
> （ctx 512）；全部统计以文本为 bootstrap 簇。
>
> **P1（周期与锁相）：否定。** 全曲线仅 3 个峰（t_2、t_6、t_19），间距 [4,13]，lag-4 自相关
> 0.027≈0；逐文本锁相率 ≤0.33。3:1 机制留下的是局部锯齿（见 P2），不构成周期性峰列——
> H2′ 在修正代码后仍被否定，"模块边界=动力学边界"没有峰结构支持。
>
> **P2（问题一）：成立且统计显著，图像比预期更精细。** linear→linear 0.0886 [0.0873,0.0899]
> < 含注意力合并 0.1026 [0.1002,0.1051]；逐文本差 −0.0139，CI [−0.0159,−0.0124]，8/8 文本
> 同向，sign p=0.0078。分解：linear→full **0.1262**（锐跳在注意力输出处）、full→linear
> **0.0789**（三类中最小——注意力后软着陆）、linear→linear 0.0886。"3 渐进 + 1 锐跳"成立。
> 对照模型（全注意力）层间 0.0843，与 hybrid 的 ll/fl 同量级——差异来自注意力插入的跳变，
> 不是 DeltaNet 天生更平。
>
> **P3（问题二）：不重合。** 宏观边界（CE 三段拟合，BIC；bootstrap 200/200 选中 3 段）在
> transition [1, 24]；注意力跳变峰在 t_3/t_7/t_19（0.126–0.139）与出口 t_31（0.315）。
> 边界 t_24 处 t_23/t_24 均为小值（0.075/0.079）——注意力峰不是阶段闸门；宏观结构由
> 嵌入适应（t_0=0.865 悬崖）和输出锐化坡决定。
>
> **平台区（切分依据，修正代码后确认）**：hybrid CE 在 L2–L21 平坦（核心 L4–L21 波动 0.65，
> 除 L3=13.79 小凸起），L22 起下降、L24 起锐化至 2.35；对照 Qwen3-1.7B 全程单调下降
> （中段波动 3.27，为 hybrid 的 5 倍）——**"hybrid 有真平台、标准模型没有"在修正代码后
> 复现成立**，V1 在标准 Qwen3 上循环失败获得表征级解释。
>
> **切分候选（交 S1b 仲裁）**：平台 L2–L21 + 完整周期约束 → M ∈ {8–19（3 模块，平台中心，
> v4 遗留）、16–23（草案，注意 L22–23 已在缓降段）、4–19（4 模块，更宽平台覆盖）、
> 9–20（旋转对照：完整周期但不对齐网格，检验网格对齐是否有超出周期完整性的额外收益）}。
> 四个候选均通过接缝验证（末层类型/相位 = 首层原始前驱）。证据等级 E1（冻结权重描述性），
> 按判定门 G1a 的"三段式拟合定切分"路线执行。
>
> 可视化：`results/figures/s1a_hybrid.png`（距离曲线 + CE 拟合 + 自相关 + P2 分组）、
> `results/figures/s1a_control_and_comparison.png`（对照曲线 + 平台区对比 + P3 不重合图）；
> 绘图脚本 `plot_s1a.py`，数据源 `results/s1a_dynamics_v2.json`。

### S1b：冻结候选块循环轨迹 — 已完成（M=L8-19 定案）

对 S1a 产出的每个候选块（至少含 measured-primary、草案 16–23、一个整模块对照）分别测：

- 20 次循环的 hidden drift；
- 每一轮完整 `E → M(loop) → D` 的 next-token CE；
- decoder 必须包含在 CE 中，不能只算循环块 logit-lens；
- 每个样本开始前清空 full-attention KV 和 linear `conv/recurrent_states`；循环轮次之间
  只传递设计中明确的 hidden state，不隐式复用上一轮 generation cache；
- 文本独立保存曲线，不能把不同文本的首轮和末轮混为一条轨迹。

> **S1b-v2 结论（2026-09-20 回填，`results/s1b_trace_v2.json`，脚本 `s1b_trace_v2.py`；旧 `s1_trace.py` 缺陷不再使用）**
>
> 协议：4 候选 + 对照 × 8 文本 × 20 轮冻结循环，每轮**完整 E→M(loop)→D 后**测 next-token CE
> （修复 v1 缺 decoder 尾部的缺陷）；attention_mask=None（LoopUS 无 padding 官方语义）；
> use_cache=False 全程，结构上无跨轮状态泄漏。守门：手动深度-1 CE 与模型前向等价
> diff=0.00000 PASS。hybrid 用 NF4（bf16 触发 ~9× 共享内存回退）；bf16 spot（L8-19，
> 2 文本 × 4 轮）最大 CE 偏移 **0.30**。
>
> **H1 判定：否定。** 所有候选与对照的 CE 都随深度单调上升，无一收缩：
> L8-19 2.39→3.03(b5)→4.41(b10)→8.32(b20)（近线性 ~0.28/轮）；L16-23（草案）加速爆炸至
> 12.50；L4-19 9.31；L9-20（旋转）11.82；对照 10–17 8.98（增速饱和）。按判定门 G1：
> **A4（无门）从训练矩阵剔除，A2 块边界门主线地位确认。**
>
> **切分定案：L8-19。** 全深度最优（每个深度点上 CE 最低）、跳块代价最低（ce0=3.50）、
> 退化近线性；草案 L16-23 正式否决（最差且加速爆炸）；L4-19 次优但每轮多 4 层开销，
> 与 L8-19 的差距（b5 差 0.14）在 NF4 噪声（±0.3）内，按循环效率取 L8-19；旋转对照
> L9-20 明显劣于 L8-19（b20 11.82 vs 8.32）→ **"周期完整"必要但不充分，模块对齐的
> F→L 接缝（块末层=注意力层）实测更优**——回答了"网格对齐是否有超出周期完整性的额外
> 收益"：有。
>
> **机制发现（对 S2/S3 直接有用）**：漂移呈"收缩—逃逸"两段结构——首轮大跳（1-cos≈0.45），
> 2–5 轮内方向漂移收敛到 ~0.001（方向锁定），但 L2 范数从低点 6.4 重新膨胀（b20 回到 8.4），
> CE 沿固定方向近线性恶化。循环动力学是**被坏吸引子支配的收缩映射，不是混沌**。A2 门
> （A_bar 阻尼插值）+ L_mono 的修复对象因此非常明确：把吸引子拉回流形，而非稳定混沌。
> 对照模型呈同样的 U 形漂移（L2 绝对量级跨模型不可比），V1"深度越深越差"获得动力学级解释。
>
> 逐文本一致性：L8-19 的 ce[5] 范围 2.24–3.36（其中 1 条文本 b5 低于深度-1，说明逐文本上
> 少量循环可以有益），ce[20] 7.84–8.85——均值趋势不被单文本驱动。证据等级 E1（冻结轨迹）。
> **S2 按 M=L8-19 + A2 主线搭建。**

### S2：训练基建 spike — 已完成

- 接入 Qwen3.5 text backbone 的 `layer_types`、linear recurrent state 和 full-attention mask；
- 实现 A2（块边界门）与 A3（full-attention 层后门）两个开关；
- 实现 LoopUS 的 B/K 随机深监督、L_mono、q head；
- 4bit 加载后，先跑 batch=1、ctx=256/512 的单步 forward/backward；
- 记录峰值显存、tok/s、显存泄漏、linear state 是否跨 batch 残留；
- 加入 20 条评分器对抗效度门。

> **S2 结论（2026-09-20 回填，`results/s2_spike.json` + `results/s2_grad_check.json` +
> `results/s2_scorer_gate.json`）**
>
> - **机制全通**：A2 块边界门（LoopUS SelectiveGate 原式，fp32 参数）+ q head +
>   每监督步 `optimizer.step`（LoopUS 忠实）在冻结 NF4 hybrid 上端到端跑通，切分
>   M=L8-19 直接可用。可训练参数 832,001（门 824,320 + 头 7,681）与预算一致；基座
>   零解冻；8/8 可训练参数梯度有限非零（q_head 梯度经 q_loss 路径，单独验证）；隔离
>   测试输出差 **0.0**（无跨 batch/跨轮状态泄漏）。
> - **吞吐与预算（ctx512，官方 B=20/K=5）**：44.3 s/batch，**57.8 监督 token/s ≈
>   0.21M/h ≈ 理论 5M/天**（现实 3–4M）。训练态峰值 8.71 GiB → WDDM 共享内存回退，
>   吞吐受限。S3 前优化项：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、
>   CE 分块、必要时减 B。**L0 预算改为 ~5M 监督 token（约 1.5–2 天）先验证 CE(d)
>   分化信号，有信号再加注**；原 20–50M 估算按实测吞吐需 10–25 天，不作为第一档。
> - **诊断注意**：spike 的 lm_first/lm_last 不可当训练进度（不同监督深度 b 的 CE
>   天然不同——S1b 结论的直接体现）；S3 必须按固定深度 d∈{1,2,4,8} 分列记录 dev CE。
>   ctx256 慢于 ctx512（48.2 vs 44.3 s/batch）为笔记本持续负载热节流伪影，不作结论。
>   A_bar 初始化均值 0.084、逐通道双峰（小 |A| 通道放行块信号、大 |A| 通道关断）——
>   LoopUS 官方初始化的实际含义，记录备查。
> - **评分器效度门：20/20 PASS**。GSM8K 仅认"完成 + ####"；ProsQA 仅认 #### 锚定
>   陈述或末句完整独立陈述（封闭话语标记白名单 therefore/thus/so/hence）；全文本
>   扫描 "X is a Y" 已禁。S3/S4 自由生成评测一律调用 `scorer_strict.py`。
>
> **S3 冒烟（2026-09-20，`results/s3_smoke_resume.json`）**：断点续训全链路 PASS——
> 中断-恢复（2+2 batch）与不中断（4 batch）dev CE 四深度**逐位一致**，RNG/数据顺序/
> AdamW 状态恢复正确；检查点 <15 MB 原子写、滚动双份。正式运行默认 ckpt-every=25
> batch，断电最多损失该间隔工作量。用户已确认接受慢速+分布式存档方案。

### S3：L0/L1 本机训练 — L0 已完成（5M token 跑满）；L1 未开始

**运行状态（2026-09-20 22:15）**：L0 已启动并在 **step 575 / 9766（监督 token 1.472M / 5M，29.4%）按用户要求于存档点暂停**，检查点验收 12/12 PASS（ckpt step=575 与 history 零偏差）。暂停时成绩（dev CE，vs 训练前基线）：

| 深度 | 基线 | step 100 | 200 | 300 | 400 | **500** |
|---|---|---|---|---|---|---|
| d1 | 2.4427 | 2.4658 | 2.3385 | 2.2466 | 2.1620 | **2.1483** |
| d2 | 2.4604 | 2.2886 | 2.1909 | 2.1337 | 2.0883 | **2.0805** |
| d4 | 2.7869 | 2.1880 | 2.1240 | 2.0890 | 2.0663 | **2.0540** |

深度倒置稳固（d4 < d1），对照 S1b 冻结曲线 d4 −0.745；d4 降幅收窄（−0.012/百步）趋平台。健康度：A_bar 0.144、grad_norm ≤18.5。

**监控已补齐（2026-09-20，暂停期间实施，`s3_train_l0.py` / `s2_spike.py` 已更新）**：
- 每个监督步记录 **q_hat**（退出门置信度，未经训练≈0.5）；
- 每次评估记录**块后隐状态 L2 范数 vs 深度**——首次测量即复现 S1b 机制签名：冻结态 d1=10.4 → d8=25.8（每轮约 +40% 范数膨胀），这是门要压制的核心对象；
- 每次评估记录 2 条贪心生成样本（depth=2，12 token）；
- `train_watch.py` / `checkpoint_accept.py` / `dashboard.py` 均已兼容新字段；验收输出新增范数与样本打印；
- 以上经一次性测试目录验证（batch 日志含 q_hat、评估含 monitor、watch 摘要行带 ‖h‖），测试目录已清理；
- **续跑时自动生效**（`--resume auto` 从 step 575 无损继续），范数曲线将从续跑后第一个评估点开始积累。

每个配置独立目录、独立 JSON，不覆盖：

- `A2_gate_only`：LoopUS 忠实块边界门；
- `A3_gate_only`：用户方案的 attention-layer 差异化门；
- `A2_lora`：A2 + reasoning block LoRA；
- `A3_lora`：A3 + reasoning block LoRA。

每 100 个 optimizer steps 保存：

- train/eval CE；
- d=1/2/4/8 的完整曲线；
- 严格评测结果；
- 3 条完整生成文本；
- gate alpha 分布、q confidence、linear recurrent state norm。

> 结论：待补（每个配置一个 `runs/` + `results/` 目录）。

### S4：消融与统计验收 — 未开始（tier-1 脚本已就绪，待跑）

| 配置 | 用途 |
|---|---|
| A1 | 原始 Qwen3.5-4B，不循环 |
| A2 | LoopUS 忠实：推理块边界单门（主线） |
| A3 | hybrid 特化：仅 full-attention 层后门 |
| A4 | 无外部门控 |
| A5（可选） | A3 的门移到块边界，隔离“门位置”与“门覆盖范围” |

必测指标：WikiText/LAMBADA PPL、ARC-C/OBQA/PIQA/HellaSwag、深度外推 B=40/80/100、
平均早停步数、KV/recurrent-state 显存与 tok/s。

- ARC continuation log-likelihood 必须校验 prefix/continuation token 对齐；
- 原始 lm-eval 结果和自定义评测分开存储；
- 主结论 n≥1000 或明确标注方向性结果；禁止用旧错位 ARC 结果宣称通过。

> 结论：待补（`results/s4_ablation.json`）。

### 评测协议冻结（S3/S4 及后续所有评测必须遵守）

V1 的自由生成正结果曾被答案抽取器污染；V5 不允许训练脚本自行定义一套宽松评分。
评测分为两条互补但不可混用的证据链：

| 协议 | 允许回答的问题 | 不能回答的问题 |
|---|---|---|
| teacher-forced scored | 给定可见 scaffold 后，循环隐状态是否能降低答案 CE/提高答案 token EM | 模型能否从自己的生成前缀进入答案段 |
| free-running | 部署时从问题开始是否真的输出最终答案，以及深度变化是否带来收益 | 不能用截断文本的最后一个数字/候选句替代最终答案 |

强制规则：

1. ProsQA 自由生成只有在文本明确出现 `####` 后的最终陈述，或完整、独立的一行
   最终陈述时才计分；不能从题目复述中匹配 `<Name> is a <thing>`。
2. GSM8K 自由生成必须优先接受 `####` 后的答案；达到 token budget 但没有 EOS 的文本
   只能记为未完成，不能用最后一个中间数字兜底。
3. `train_prosqa.py` 的 `score_with_gold()` 和 `eval_scored.py` 只产生 teacher-forced
   CE/EM；不能把其结果写成自由生成准确率。
4. `eval_fast.py`、`extract_prosqa_answer()` 和历史 `eval_ouro*` 中的宽松 fallback
   只能用于诊断，不能进入主表。正式 scorer 必须把原文、预测、是否含 `####`、EOS 和
   token 数一并写入 JSON。
5. 每个新评分器先通过至少 20 条对抗样本：答案出现在问题复述、推理中间句、截断末尾、
   `####` 前后和错误候选句时，均不能误记为正确。
6. 深度比较使用同一题目、同一生成预算、同一随机顺序；报告逐题配对结果和 Wilson
   区间，n<1000 时只能称为方向性结果，不得宣称统计显著提升。

旧文件的处理：`results/s1a_dynamics.json`、旧 G0 ARC 数字和宽松 ProsQA JSON 保留作
审计记录，但必须在新报告中标为 provisional/invalid；重跑前不能覆盖原文件。

### S5：回填与迁移 — 未开始

把每个 S 的“结论”写入本文末尾记录表，并更新：

- `docs/reports/PROJECT_REVIEW.md` 附录；
- `docs/reports/RESEARCH_SUMMARY.md`；
- `docs/reports/DELIVERABLES.md`；
- 服务器计划（主线 A2，A3 为 hybrid 专项对照）。

---

### S6：关闭门（halting gate）训练 — 待商榷（执行细则已细化）

背景：L0 的 q_head 与主训练联合训练未收敛（BCE 全程贴 ln2≈0.693 无下降），
AUC 诊断 0.534≈随机——q 只学到"循环到第几圈"（mean q 单调降 0.25→−0.09），
不编码答案质量；根因是监督目标（整段 token 准确率）在深度维度上是平线（0.56）。
"循环块已训好、缺关闭门"这一处境的文献结论：

- arXiv 2607.20519（Ouro-1.4B/2.6B + 合成任务）：失败源于**联合门训练诱导的轨迹**，
  而非门表达力；**冻结轨迹上事后拟合门可恢复强早退行为**（低平均深度近满深度准确率），
  简单置信读出（零训练）经常匹配或超过学习门；
- LoopUS 附录 E 自带零训练收敛退出规则（‖h̄_{t+1}−h̄_t‖₂ < ε，不用 confidence head），
  其阈值规则给出最锐利退出分布；Huginn 的 KL<5e-4 退出属同类；
- RecurTrace（arXiv 2609.03379）：halting 头用"再深一圈 loss 是否下降"的 oracle 监督
  （0.6B–8B 有效）；ACT/PonderNet 联合训练在 LM 上塌缩到 1 圈——联合重训路线排除。

候选路线（成本升序）：

| 路线 | 做法 | 成本 | 状态 |
|---|---|---|---|
| R0 零训练阈值 | 附录 E 收敛规则 ‖Δh‖<ε + softmax 置信读出基线，测 exit-depth/CE Pareto | ~1 小时，纯推理 | 待商榷 |
| R1 冻结轨迹 post-hoc | 前向收集 1–2k 块 × d=1..8 逐圈特征 + Δloss oracle 标签，拟合小头（7.7k–832k）；验收 AUC≥0.7 且 Pareto 优于 R0 | 一晚（特征 2–4h 纯前向 + 分钟级拟合） | 待商榷 |
| R2 联合重训 | ACT/PonderNet/RL-Halting 式 | 昂贵且文献实测有害（塌圈/稳定次优） | 排除 |

待商榷点：

1. R0/R1 验收口径：早停 Pareto（平均退出深度 vs CE/任务分）与固定深度基线的算力对齐方式；
2. R1 标签粒度：逐 token Δloss oracle（RecurTrace 用法，贵）vs 序列级；
3. 与 L1（β 消融 + FineWeb-Edu 预算阶梯）先后：R0 免训练可先行，R1 在 L1 前完成或与 L1b 并行；
4. 与 KV cache 工程的集成顺序（退出要兑现算力节省，依赖 KV cache / 早退传播）。

**执行细则（文献复核 + traj_alive 轨迹判别后细化，待批准执行）**

前置事实（`results/traj_alive_v5_l0.json` 已标定）：轨迹活性认证（速度几何衰减
1.28→0.007 不归零、KL 平台 0.0005、翻转率 1–2.5%）；CE 拐点 **d=8**（d13 后微升）
= oracle 标签自然翻转点；速度阈值 ‖Δh‖/‖h‖ < 0.01 落在 d≈8–10。

R0 零训练基线（~1 小时，纯推理，16 dev 块 + 短生成，四条规则各测退出 Pareto）：

1. 速度阈值：‖Δh‖/‖h‖ < 0.01（traj_alive 标定）；
2. KL 退出：KL(p_{d-1}, p_d) < 5e-4（Huginn）；
3. patience-2：连续 2 圈 top-1 不变才退（Zhou et al. NeurIPS 2020）；
4. 附录 E 收敛规则 ‖h̄_{d+1}−h̄_d‖₂ < ε（LoopUS 原文，ε 网格扫描）。

产出 `results/s6_r0_pareto.json`：平均退出深度 × CE/acc，对照固定深度 d=4/8/16。

R1 冻结轨迹 post-hoc 门（一晚，M + SelectiveGate + LM 头全程 requires_grad_(False)）：

1. 特征收集（纯前向 2–4h，traj_alive 收集器扩展）：1–2k WikiText 块 × d=1..16，
   逐圈逐 token 记录 [delta_pre 范数（**不加 LN**）、A_bar 均值、块后 h 范数、‖Δh‖、当圈 Δloss]；
2. oracle 标签（RecurTrace 口径）：y = 1[loss(h_{d+1}) < loss(h_d)]，逐 token；
3. 双头对照：线性 + 2 层 MLP（2607.20519：头容量不敏感），裸增量范数特征保留
   （AUC 诊断教训：LN 抹掉范数信号）；
4. 训练：只训头，AdamW 1e-3，约 5 epoch，文档级 train/val 切分防泄漏；
5. 评估：val AUC≥0.7；早停 Pareto 压过 R0 全部基线；退出深度分布"易浅难深"
   而非常数；可选 ICML24 风险控制校准（无标签 Consistency Risk 口径）。

产出 `results/s6_r1_gate.json` + `runs/v5_l0/exit_head.pt`（独立文件，不动原存档）。

验收总线（回写结论表）：R1 门 AUC≥0.7 且 Pareto 支配 R0 → S6"关闭门可用"；
否则记录失败并回炉标签/特征设计。

待商榷点 → 讨论后的建议结论：①验收口径=等算力 Pareto（退出深度折算 FLOPs）+固定深度对照；
②标签粒度=逐 token（RecurTrace 0.6B–8B 普适性）；③顺序=tier-1(2h) → R0(1h) → R1(一晚) → L1；
④KV cache=R1 并行的工程前提，不阻塞 R1 训练。

## 6. 证据强度分级

| 级别 | 含义 |
|---|---|
| E0 | 架构/配置事实，来自 config 或源码 |
| E1 | 预实验描述性曲线，尚无训练因果结论 |
| E2 | 同批样本、校正评分、带对照的训练结果 |
| E3 | 多数据集、独立复现、统计检验通过的结果 |

当前状态：Qwen3.5 架构为 E0；旧 S1a 曲线仅 E1 且存在实现问题；旧 G0 ARC 数字不作为证据；
LoopUS 官方论文结果是外部 E3 参考，不是本项目复现结论。

## 7. 风险与止损

- **Hybrid 缓存误解**：linear attention 有固定形状 conv/recurrent states，不等于“无状态”；
  每个 depth iteration 必须隔离状态，优先写单元测试。
- **4bit 自定义模型不兼容**：30 分钟 spike 失败就切 L0 bf16 小 batch 或 2B 模型，不私自改包版本。
- **S1a 旧数据误导**：不再引用旧 P1/P2/P3 数字；修正脚本后重测。
- **训练分支爆炸**：先 A2 gate-only；若 S2 CE(d) 没有正信号，停止 L1，不跑四分支。
- **资源约束**：本机只证明方向；3B token 与 27B 全训属于服务器。

## 8. 结论记录表（append-only）

| 日期 | 步骤 | 结论 | 证据 |
|---|---|---|---|
| 2026-09-19 | D1–D6 | 块边界不再按 3:1 直接决定；A2 块边界门为主线，A3 层类型门为对照；S1a 需校正后重测 | 本文档 |
| 2026-09-19 | S0 | config/权重/transformers 5.3 加载通过；4bit 未完成 | `q35cfg.json`、`logs_q35_download.txt` |
| 2026-09-20 | S0.4 | **PASS**：NF4 加载正常（前向 3.0 GiB / 反向 3.78 GiB）；梯度可穿过冻结 4-bit 主干（grad absmax 0.021），基座零解冻 → L0/L1 可行。注：DeltaNet 慢速路径，训练前装 fla+causal-conv1d | `results/s0_hybrid_spike.json` |
| 2026-09-20 | 文档定稿 | V5 定稿为现行方案；`docs/README.md` 索引切换，V3/V4 标为历史版本；历史报告中的"60B–7700B token"统一限定为 V1 旧配方估算（LoopUS 3B 先例不适用该下界）；评测协议冻结小节并入；全部 S 阶段保持暂停，恢复时从 S0.4（4-bit spike）与 S1a-v2 重测开始 | `docs/README.md`、本文档 |
|  | S1a-v1 | 旧测量存在末层双 norm、转移分组错位、标准模型硬编码等问题；不作定案 | `s1a_dynamics.py`、`results/s1a_dynamics.json` |
| 2026-09-20 | S1a-v2 | P1 周期否定（3 峰 t=[2,6,19]，lag-4 自相关 0.027，逐文本锁相 ≤1/3）；P2 成立且显著（ll 0.0886 < 含注意力 0.1026，8/8 文本同向 p=0.0078；锐跳在注意力输出 0.126、其后软着陆 0.079）；P3 不重合（宏观边界 t=[1,24]，注意力峰 t=3/7/19/31）；hybrid 平台 L2–L21（波动 0.65）vs 对照单调下降（波动 3.27）。候选 M：8–19 / 16–23 / 4–19 / 旋转 9–20，接缝验证全部通过，交 S1b 仲裁 | `results/s1a_dynamics_v2.json`、`results/figures/s1a_*.png` |
| 2026-09-20 | S1b | **H1 否定**（4 候选 + 对照 CE 全部随深度单调上升）→ A4 剔除、A2 主线确认；**切分定案 M=L8-19**（全深度最优、近线性退化；草案 L16-23 否决；旋转对照 L9-20 落败 → 模块对齐 F→L 接缝有超出周期完整性的实测收益）；机制：方向漂移 5 轮内收敛 ~0.001 但范数再膨胀，坏吸引子主导；等价守门 diff=0，NF4 偏移 ≤0.30 | `results/s1b_trace_v2.json` |
| 2026-09-20 | S2 | 训练步全通：门+头 832,001 参数、8/8 梯度有限非零、基座零解冻、隔离 0.0；官方配方 ctx512/B20/K5 = 44.3 s/batch ≈ 57.8 监督 tok/s（峰值 8.71 GiB 触发共享内存回退）；**L0 预算改为 ~5M token 先验证信号**；评分器对抗效度门 20/20 PASS | `results/s2_spike.json`、`results/s2_grad_check.json`、`results/s2_scorer_gate.json` |
| 2026-09-20 | S3 冒烟 | 断点续训 PASS：中断恢复 vs 不中断 dev CE 四深度逐位一致；检查点 <15 MB 原子写+滚动双份；用户接受慢速+分布式存档 | `results/s3_smoke_resume.json` |
| 2026-09-20 | S3 L0 | 正式启动：只训门+头 832k 参数，M=L8-19，B=20/K=5/ctx512，目标 5M 监督 token，ckpt-every=25、eval-every=100 | `runs/v5_l0/` |
| 2026-09-20 | S3 暂停 | **step 575（29.4%）存档点暂停**（用户要求），验收 12/12 PASS、零进度损失；暂停时 d1/d2/d4 = 2.148/2.081/2.054（基线 2.443/2.460/2.787），深度倒置稳固、对照冻结 d4 −0.745，降幅收窄趋平台 | `runs/v5_l0/`、`results/checkpoint_accept_v5_l0.json` |
| 2026-09-20 | S3 监控补齐 | q_hat、块后隐状态范数（复现 S1b 机制：冻结态每轮 +40% 范数膨胀）、2 条生成样本纳入每次评估；watch/accept/dashboard 全链路兼容；一次性目录验证后清理；续跑自动生效 | `s3_train_l0.py`、`s2_spike.py` |
| 2026-09-20 | 看板 | 量程拆分修复（冻结曲线曾把纵轴拉到 8.6 压扁主曲线）+ 范数监控图 + JS 容错；五图浏览器实测通过；局域网 10.200.96.105:8741 | `dashboard.py` |
| 2026-09-21 | S3 step1000 全检 | **PASSED（质量检测全绿）**：① 深度外推 **FLAT**——16 个新 dev 块上 d=1/2/4/8/16/32 的 CE = 2.119/2.096/2.084/**2.081**/2.091/2.118（全距仅 0.038），**d=32（训练深度的 1.6 倍）无发散**，对照 S1b 冻结态 b=20 爆到 8.3——门+训练彻底修复漂移；② 12 条 32-token 贪心生成（3 提示 × 4 深度含 d=8）全部连贯、无重复循环、无退化；③ 门高度通道选择性（37.7% 关断 / 4.5% 全开 / std 0.35）；④ step1000 评估 d4=**1.9909 首破 2.0**，范数膨胀率收敛至 ~+10%/轮（冻结 +70%）。训练暂停于 step 1000（51.2%），待用户决定续训 | `results/quality_check_v5_l0_step1000.json` |
| 2026-09-21 | S3 step1300 全检 | **PASSED，较 step1000 全面改善**：① 深度外推 FLAT 复现且更平——d=1/2/4/8/16/32 = 2.098/2.078/2.066/**2.062**/2.067/2.091（全距 0.036，最低点移到 d8，d8 比 d1 低 0.036）；② 12 条生成全部连贯，且**四个深度的输出几乎逐字一致**（深度对生成分布近似透明=稳定性达成）；③ 门选择性保持（36.2% 关断/10.1% 全开）；④ 训练评估 d4=1.9689（−0.818），范数膨胀率收窄至 **+6.1%/轮**（step600 +17.4% → 持续单调下降）。暂停于 step 1300（67.9%），待用户决定续训（剩 ~652 步） | `results/quality_check_v5_l0_step1300.json` |
| 2026-09-20 | 27B 架构核实 | config.json 实测：27B 与 4B 同家族（qwen3_5、64 层严格 3:1、hidden 5120、256K 上下文），**无原生循环块**——与 4B 同为 retrofit 场景，方法/门控/配方可移植、切分须重测；先例 Qwopus3.8-27B-Flash 走 SFT+RL 缩短可见 CoT（−1.45pp MMLU-Pro） | `q38_27b_config.json`、HF `Jackrong/Qwopus3.8-27B-Flash` |
| 2026-09-22 | S3 L0 完成 | **5M 监督 token 跑满（step 1954，5,002,240）**，终态审计 12/12 PASS；final 深度扫描 d=1/2/4/8 = 1.9786/1.9539/1.9377/**1.9297** 单调改善，范数 +2%/轮；机制认证点 2.56M token（step1000 质检），饱和点 ~3.33M token（斜率塌缩 20×） | `runs/v5_l0/`、`results/checkpoint_accept_v5_l0.json` |
| 2026-09-22 | q_head AUC 诊断 | **AUC=0.534≈随机**：q 仅学到深度计数（mean q 0.25→−0.09 单调降），不编码答案质量；目标（整段 acc）在深度上是平线（0.56）为主因 → 换 Δloss 方向目标 + 绕过 LN | `results/q_head_auc_v5_l0.json` |
| 2026-09-22 | S6 立项 | 关闭门训练节新增（**待商榷**）：R0 零训练阈值 / R1 冻结轨迹 post-hoc oracle 头（arXiv 2607.20519 背书）/ R2 联合重训排除；四个待商榷点见 S6 | 本文档 |
| 2026-09-22 | tier-1 就绪 | `s4_tier1_eval.py` 修复完成并干跑验证（本地 gsm8k 改写版格式适配、ctx 768、自适应 shots）；按用户指令暂停，明日跑完整版（含 free-running 严格评分） | `s4_tier1_eval.py` |
| 2026-09-22 | S6 R0 | **零训练退出基线完成**（16 块 × d=1..20）：最优前沿 ruleE ε=0.1 平均退出 5.25 圈 / CE +0.0021、velocity 0.02 退出 5.81 / +0.0018（vs 固定 d8 最优 CE 2.0327）——**省 ~35% 算力近零损失**；Huginn KL5e-4 退出 9.3 圈被支配、patience 系 CE 代价大；固定深度 d4/8/16 极平（差 0.005）→ 序列级关闭问题基本被阈值解决，R1 验收线抬高为逐 token 自适应支配该前沿 | `results/s6_r0_pareto.json` |
| 2026-09-23 | tier-1 A 链 | **能力验收 teacher-forced 口径完成**（GSM8K 300 题配对）：base=0.7453 最优；trained d1..d8=0.814/0.815/0.819/0.827（vs base +0.068~+0.082，**循环化结构代价 ~9%**，d1 即存在=门控稀释，非训练损伤）；frozen d1/2/4/8=1.228/1.172/1.457/**2.553**——任务深度漂移通病在原生循环动力学上剧烈存在，**门训练压制 97%**（trained 深度斜率 +0.014 vs frozen +1.33）；trained−frozen = −0.36~−1.73（CI 全排 0）。自由生成口径（B 链）待重设计（96 预算协议性全 None） | `results/s4_tier1_v5_l0.json` |
| 2026-09-23 | tier-1 B 链 | **自由生成口径完成（n=4/256预算/1-shot）**：主表严格分三配置全 0/4——**协议-模型错配**：基座模型贪心续写不自终止，12/12 预算尽无 EOS（规则2 判未完成）；协议许可的诊断读数（规则4）：base 2/4 对、**trained d1/d4 均 0/4 且出现重复退化**（"invests 1000..."式复读、生成游离出新假例题）→ 自由生成能力损伤实锤（比 A 链 +0.068 严重），深度无正收益。**分水岭判定：L1 v2 必要**，优先级=修生成退化（h₀ 再注入）> 深度收益（β=1.0+块训练）> 起步税（恒等首圈）。待用户决策：finished 定义是否修订（EOS→自然停止边界） | `results/s4_tier1_b_freegen.json` |
|  | S1b | 待测：含 decoder 的完整循环轨迹 | `results/s1b_trace.json` |
|  | S2 | 待测 | `results/s2_spike.json` |
|  | S3 | 待测 | `runs/v5_*` |
|  | S4 | 待测 | `results/s4_ablation.json` |
|  | S5 | 待测 | PROJECT_REVIEW 附录 |

## 参考

- Park et al., **LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models**, arXiv:2605.11011。
- `loopus/models/modeling_lds.py`：官方 SelectiveGate、ReasoningBlock、q head。
- `loopus/training_runtime.py`：官方随机深监督、L_mono、L_Q、checkpoint 逻辑。
- `q35cfg.json`：Qwen3.5-4B-Base 文本架构配置。
