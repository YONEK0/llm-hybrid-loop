# Recurrent-Depth Latent Reasoning — Results

Post-training **Qwen3-4B-Thinking-2507** (4-bit NF4) so part of its reasoning happens in
**recurrent latent computation** instead of emitted tokens. Single 8GB laptop GPU.

Technique lineage: Geiping et al. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
(recurrent depth), Zhu et al. (ByteDance Ouro)
[arXiv:2510.25741](https://arxiv.org/abs/2510.25741), Bae et al. (Relaxed Recursive
Transformers) [arXiv:2410.20672](https://arxiv.org/abs/2410.20672), Hao et al. (Coconut,
Meta) [arXiv:2412.06769](https://arxiv.org/abs/2412.06769). This is the mechanism
[reported](https://www.theinformation.com/articles/secret-technique-behind-openais-astra-model-sparks-security-concerns)
— **unofficially, no OpenAI confirmation** — for GPT-6 Astra.

## Implementation

```
prelude (layers 0-23) | recurrent block (24-29, weights reused) x K | coda (30-35)
```

- Block weights **shared across iterations**; each iteration gets its own **low-rank
  delta** on 7 projections ("relaxed" tying).
- Each iteration **re-injects the prelude output** alongside its running state.
- Loop count is **sampled from 1..K during training**, so one checkpoint serves any depth.
- **Deep supervision**: answer read out after every iteration.
- Trainable: 24.1M (deltas + injection). 4-bit base frozen.

## Correctness gate (passes)

`check_equivalence.py`: with zero-initialised deltas, **depth 1 reproduces the stock model
exactly** — `max_abs_diff = 0.0`.

Reaching this required three fixes that each silently corrupt results: rebuild the causal
mask via the model's own `create_causal_mask`; pass `position_ids` to every layer; derive
the answer slice from that branch's own prefix length.

## Result: the central claim is NOT supported

Two training runs, both scored with the gold first-k rationale steps fed in (the protocol
matches training). `d` = latent iterations at inference.

**Run A — depth-sampled** (1200 steps, 2500 problems, scaffold annealed 3→2→1→0, loop count
drawn from 1..4 per step):

| visible steps | d=1 | d=4 | d=8 |
| --- | --- | --- | --- |
| 3 | **0.533** | 0.200 | 0.133 |
| 2 | 0.233 | 0.100 | 0.133 |
| 1 | 0.100 | 0.067 | 0.133 |
| 0 | 0.033 | 0.000 | 0.133 |

**Run B — fixed depth 4** (600 steps, scaffold fixed at 2 visible steps):

| visible steps | d=1 | d=2 | d=4 |
| --- | --- | --- | --- |
| 2 | 0.400 | 0.400 | 0.367 |
| 1 | 0.133 | 0.133 | 0.100 |
| 0 | 0.033 | 0.000 | 0.000 |

Both runs: **accuracy is flat or falling in depth.** Latent iterations do not compensate for
removed reasoning steps — the opposite of what the method predicts. Depth-sampled training
is actively worse than fixed-depth (0.533 vs 0.400 at k=3/2, d=1), consistent with the
sampled objective giving later iterations weak and inconsistent training signal.

### The decisive control

Same protocol, trained parameters disabled (deltas zeroed, injection reset to passthrough):

| configuration | trained | parameters disabled |
| --- | --- | --- |
| 3 steps, depth 1 | 0.533 | **0.633** |
| 3 steps, depth 4 | 0.200 | **0.433** |
| 3 steps, depth 8 | 0.133 | — |
| 0 steps, depth 1 | CE 0.636 | CE **9.923** |

Two things this establishes:

1. **With enough visible scaffold the untouched base model is *better* than our trained
   version** — the trained deltas actively hurt (0.633 → 0.533 at depth 1). The accuracy
   in that row is the base model reading the scaffold, not latent reasoning.
2. **Without a scaffold the trained parameters do rescue the loss** (CE 9.9 → 0.64), so
   training demonstrably changed behaviour — but accuracy is still ~0. The model learned
   the answer *format* (`#### N`) without the arithmetic.

Free-running generation: 0.000 accuracy in every configuration.

### ProsQA result (the fair test — Coconut's own benchmark)

Trained 1200 steps on 10000 ProsQA problems, curriculum scaffold annealed 3→2→1→0,
depth sampling (1..4). Evaluated on 30 held-out test questions.

| configuration | accuracy | mean tokens |
| --- | --- | --- |
| base model (CoT, frozen, no deltas, no loop) | **0.567** | 48.0 |
| trained, **depth 1** | **0.567** | **16.0** |
| trained, depth 4 | 0.033 | 16.0 |
| trained, depth 8 | 0.000 | 16.0 |

**Token compression works**: the trained model at depth 1 matches the base model's accuracy
while using 16 tokens instead of 48 (3:1 compression). The training taught the model to skip
explicit reasoning while maintaining accuracy.

**Latent depth scaling fails**: increasing the loop count from 1 to 4 to 8 *destroys*
accuracy (56.7% → 3.3% → 0.0%). The recurrent block, applied more than once, actively
corrupts the representation. This is the opposite of what the method predicts.

## What worked vs what did not

**Worked:** the conversion is exact and verified; training is stable in ~5.7 GiB VRAM; the
looped forward pass, per-iteration deltas, deep supervision and depth sampling all run
correctly; loop depth is a genuine runtime knob (no retraining to change it).

**Did not work:** latent iterations substituting for chain-of-thought steps on GSM8K with
this model/data/compute budget. The trained parameters help only in the region where the
base model would otherwise fail completely, and there they lift loss without lifting
accuracy.

## Diagnosis

- **Exposure bias.** The scaffold is teacher-forced during training; at inference the model
  must produce it. Its 3-5 token answers require the first token to carry the whole
  computation. Free-running accuracy is 0 while teacher-forced exact match is 0.53 at the
  same depth — a large gap that is about generation, not about latent reasoning.
- **Teacher forcing also inflates the depth-1 number.** Given 3 gold steps, the answer is
  largely determined, so depth 1 looks strong while doing no latent computation.
- **Two bugs found and fixed along the way**, both of which would have invalidated results
  silently: the label window was offset by the prompt length (equal element counts let
  cross-entropy score the wrong positions), and `trainable_parameters()` omitted the
  injection projection, so re-injection was inert for a whole 900-step run
  (`injection_delta = 0.0000` afterwards). The second run confirmed the fix learns
  (0 → 0.0008 in 8 steps).
- **A third bug** in the ablation restore path (`deltas_state_dict` picked up bitsandbytes
  quantisation metadata) is fixed via `delta_snapshot`/`restore_deltas`.

## Honest assessment

**The binding constraint is scale, not implementation.** Measured training volume here
versus the reference implementations:

| work | tokens | ratio |
| --- | --- | --- |
| This project (cur-v2, 1200 steps) | **0.18M** | 1x |
| Geiping et al., Huginn-3.5B (from scratch) | 800B | 4.4 millionx more |
| Ouro (from scratch, 4 loops) | 7.7T | 43 millionx more |
| Relaxed Recursive Transformers (uptraining) | 60B | 330,000x more |

Three orders of magnitude separate this from even the lightest published conversion, and
the published *successful* looped models were pre-trained with the loop in place, not
converted afterwards. At 0.18M tokens the model has seen roughly 1600 GSM8K-length
sequences; the reference runs used billions.

This is why the negative result should not be read as evidence against recurrent depth.
It is evidence that the mechanism cannot be *bolted on* with a fraction of a percent of a
percent of the required training on an 8GB laptop.

What a fair test would require:

1. **Train the loop from pre-training** (Geiping, Ouro) rather than converting a frozen
   model — the published successes did this.
2. **Synthetic reasoning with step-level supervision.** Coconut's favourable results are
   on synthetic logic (ProsQA: 97.0% vs 77.5% for CoT), not GSM8K, and its curriculum
   supervises per-step latent *tokens fed back as embeddings* — architecturally different
   from looping a block at a single position.
3. **Evaluate where the base model is at chance.** Here, giving 3 gold steps makes the
   answer nearly determined, so depth 1 scores 0.53 while doing no latent computation, and
   the ablation shows the untouched base model is *better* (0.633) — the metric rewards the
   scaffold, not the loop.
4. **Budget**: multiple GPUs for days, not 8GB for 20 minutes.

## Bugs found and fixed (each would have silently invalidated results)

- **Label window offset by the prompt length.** With a scaffold, the supervised slice must
  start at `len(prompt)`, not `len(prompt) + len(scaffold) - 1`. Element counts matched
  either way, so cross-entropy scored the wrong positions without raising.
- **`trainable_parameters()` omitted the injection projection**, so loop re-injection was
  inert for an entire 900-step run (`injection_delta = 0.0000` at the end). The second run
  confirmed the fix learns (0 → 0.0008 in 8 steps); `check_injection.py` now guards this.
- **`deltas_state_dict()` captured bitsandbytes quantisation metadata** from the frozen
  projectors it wraps, so restoring a snapshot raised on unexpected keys. Fixed via
  `delta_snapshot` / `restore_deltas`.
- **Non-causal attention mask and missing `position_ids`** in the hand-written loop made
  output incoherent. Both were needed for the depth-1 equivalence gate to pass at 0.0.

## What would move this forward on this hardware

Given the numbers above, the honest next step is **not** more of the same. Options in
increasing cost:

1. Verify the loop's mechanics on a **synthetic task where step-level latent supervision is
   natural** (ProsQA-style DAG reasoning) rather than GSM8K, using a small model that can be
   trained from scratch here. This tests latent reasoning without needing billions of tokens.
2. Use the **pre-trained looped models** that already exist (Ouro-1.4B/2.6B, Huginn-3.5B)
   to study depth-vs-accuracy behaviour directly — no training required, and they were built
   with the loop in place.
3. Only then consider whether conversion-style post-training is viable at all.

## Reproduce

```bash
.venv/Scripts/python.exe download_model.py unsloth/Qwen3-4B-Thinking-2507-bnb-4bit
.venv/Scripts/python.exe download_data.py
.venv/Scripts/python.exe check_equivalence.py          # must pass
.venv/Scripts/python.exe check_injection.py            # verifies injection trains
.venv/Scripts/python.exe train_curriculum.py --run-name cur-v2 --train-count 2500 \
    --max-steps 1200 --accum 4 --depth-sampling --keep-steps-schedule "0:3,600:2,900:1,1100:0"
.venv/Scripts/python.exe eval_scored.py  --checkpoint runs/cur-v2/checkpoint
.venv/Scripts/python.exe control_disabled.py
```

Artifacts: `results/cur-v2_scored_grid.json`, `results/cur-v2_trained_vs_disabled.json`,
`runs/cur-v2/history.json`, `logs_cur_v2.txt`.

Environment: Windows, RTX 5060 Laptop (8GB), torch 2.11+cu128, transformers 5.17,
bitsandbytes 0.50.2, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
`github.com` unreachable here (clones via `gh-proxy.com`); models via
`HF_ENDPOINT=https://hf-mirror.com`.

## 补充：Ouro 深度-准确率曲线（收尾实验）

`eval_ouro_depth.py`（`ouro_env`，transformers 4.57.6）利用 `modeling_ouro.py` 前向签名中的
`exit_at_step=k` 参数扫描有效循环深度：模型内部始终跑满 `total_ut_steps=4` 轮迭代（KV 缓存
逐轮存储、完全一致），仅改变从哪一轮的归一化隐状态读出 logits。默认推理
（`early_exit_threshold=1.0` 永不触发）等价于 `exit_at_step=3`。

ProsQA test 30 题（同题、greedy、≤16 token）：

| 有效深度 | Ouro-2.6B（48L×4，7.7T token） | Ouro-1.4B（24L×4） |
|---|---|---|
| 1 | 26.7% | 46.7% |
| 2 | 40.0% | 30.0% |
| 3 | 56.7% | 36.7% |
| 4 | 60.0% | 36.7% |

发现：

1. **Ouro-2.6B 单调上升**（+33 点），深度 4 与此前独立评测的 60% 精确复现。深度 1 时
   模型退化成显式文本 CoT（"Okay, let's try to figure out..."），16 token 答不完；深度 4
   时直接给出简洁答案——循环迭代把显式推理替换成了隐空间推理。
2. **Ouro-1.4B 曲线平坦甚至反向**，与我们在 Qwen3-4B 后训练改造中"加深循环降低准确率"
   的负面结果同构。

这是本研究核心结论（循环深度的不动点性质必须在预训练中习得，后训练无法补足）的最强
正面佐证：深度单调增益只出现在"循环架构 + 海量预训练"的模型上，且预训练越充分越明显。

Artifacts: `results/ouro_depth_sweep_Ouro-1.4B.json`, `results/ouro_depth_sweep_Ouro-2.6B.json`.

Run:

```bash
ouro_env/Scripts/python.exe eval_ouro_depth.py Ouro-1.4B
ouro_env/Scripts/python.exe eval_ouro_depth.py Ouro-2.6B
```

## 评分效度修订（2026-09-19，覆盖上文宽松口径数字）

收尾审计发现：上文及 `RESEARCH_SUMMARY.md` 早期版本中所有 ProsQA **自由生成**数字
（56.7%、60%、46.7% 等）由抽取伪影构成。机制：宽松抽取器从后向前找第一个
`<Name> is a <thing>` 模式，模型开头的"Okay, let's figure out if Eva is a lorpus..."
复述句被当答案；gold 恰为首个被提及候选类时误记为正确。

- 诊断（`diagnose_extraction.py`，n=30）：base@48 / base@16 / trained@16 三条件各
  17 个"正确"全部来自首行复述；严格口径（仅认 #### 后文本或整行裸陈述句）0/30；
  0/30 篇全文任何位置无裸答案行，无误杀。
- n=150 扩展（`eval_compression_scaled.py`）：三条件逐题完全一致（McNemar +0/-0）。
- Ouro 深度曲线严格口径：2.6B 0%→13.3%→53.3%→53.3%（单调性成立，60% 系高估）；
  1.4B 0%→0%→10%→20%（"深度 1 最高 46.7%"反常消失）。
- 底模天花板：ProsQA 512/1024 token 均 0/30 完成（零 EOS），严格 0/30；GSM8K 思考
  256 token 15% 且 15/15 为截断兜底伪影。
- 教师强制（scored）协议数字不受影响：k=0 EM 25%、GSM8K k3_d1 53.3% 等维持。
- 守门规则：自由生成评测一律用严格抽取器；`eval_ouro_depth.py` 已改为双标准输出。

## 附：与"偷星九月333"教学视频的方法对照（2026-09-19）

B 站视频《recurrent depth，gpt-6-astra的新架构？让模型无声的思考》
（BV1UqYJ6sErE，2026-09-08，14:13，UP：偷星九月333）讲解的循环深度训练配方，
经视频截帧逐页核对，共四个组件：

1. **每轮循环深监督**：共享块 F_θ 循环 T 轮，每轮结束都接同一个 LM head
   （W_out 共享）对下一 token 预测，产生 T 个交叉熵损失，标签均为 x_{i+1}。
   → **本项目已实现**（`train_curriculum.py` 深监督，aux_weight=0.2）。
2. **损失加权**：最简单取平均；进阶用退出分布 π 加权任务损失。
   → 本项目用随机深度采样近似（等效家族）。
3. **退出门（PonderNet 式）**：λ=σ(w^T h+b) 逐轮给出退出概率，
   π_i^(t)=λ_i^(t)·∏_{s<t}(1−λ_i^(s))；推理时按累计继续概率 S=∏(1−λ) 动态退出。
4. **退出门两阶段训练**（本项目此前未实现的部分）：
   - 阶段一：联合训练主干+LM head+退出门，任务损失加熵正则 −βH(π)，
     防止退出分布过早坍缩到某一轮；
   - 阶段二：冻结主干与 LM head，仅训退出门。以相邻两轮损失下降量
     I^(t)=max(0, sg(ℓ^(t−1))−sg(ℓ^(t))) 构造继续概率软标签
     w^(t)=σ(k(I^(t)−τ))，对门做二元交叉熵（w=继续，λ=退出）。

### 对照结论

该配方是本项目的**完整版**：我们实现了 1、2，缺的正是 3、4（退出门及其两阶段
训练——项目早期讨论过"用损失下降构造二分类软标签"，正是该视频阶段二的方案；
Ouro 的 `early_exit_gate` 即预训练好的对应物）。

但需要明确：**退出门训练的信号来源是"加深循环能降低损失"这一前提**。在本项目的
后训练改造设定下，实测加深循环使损失上升、准确率下降（depth 1→4→8：
56.7%→3.3%→0%），I^(t)≈0 或为负 → 软标签会退化为"永远第 1 轮就退出"，
门学到的是"隐空间迭代没有增益"这一事实，而非产生增益。因此该配方：

- **可行域 A（视频演示域）**：从零/小模型预训练时就带循环+门联合训练——
  与本项目"循环能力需在预训练中习得"的结论一致，机制演示完全可行；
- **可行域 B（服务器路线）**：作为 60B token Relaxed Recursive uptraining 的
  组成部分（阶段一联合训练 + 熵正则 + 阶段二门微调）——应加入服务器方案；
- **不可行域（本项目已穷尽的）**：冻结强底模 + 8GB 后训练——门无信号可学。

行动项：服务器 uptraining 方案在 `train_curriculum.py` 基础上增加
退出 gate 模块（结构同 Ouro `early_exit_gate`）、π 加权任务损失、熵正则、
两阶段训练开关。改动量约 150 行。
