# LoopUS 源码解剖笔记（Phase 1 产出）

> 源码：`loopus/`（github.com/Thrillcrazyer/LoopUS，clone 于 2026-09-19）
> 核心文件：`models/modeling_lds.py`（1611 行）、`training_runtime.py`、`utils/inference.py`

## 1. 架构（LDSForCausalLM = encoder + reasoning×N + decoder）

- **三段切分**：底模层按余弦距离分段动态切成 EncoderBlock / ReasoningBlock /
  DecoderBlock，只有中间 ReasoningBlock 被循环 N 次（默认 N=8 推理 / B=20 训练）。
  加载时也可用 `--encoder-layers/--decoder-layers` 手工指定。
- **SelectiveGate**（`modeling_lds.py:308`）——Mamba 风格 ZOH 更新，逐 token/通道：
  ```
  dt_low = W_in·(h_new − h_old)            # 低秩投影（dt_rank = hidden/16）
  delta  = softplus(W·dt_low + b)          # >0，Mamba S4D 初始化（dt∈[0.001,0.1]）
  A_bar  = exp(delta · A)，A = −exp(A_log) < 0   # ∈(0,1)
  out    = A_bar·h_new + (1−A_bar)·h_old   # 阻尼插值，初始 A_bar≈1（近恒等）
  ```
  关键点：门控信号来自**新旧隐状态的差**（我们 V1 的 LatentInjection 用的是
  固定 h₀，没有新旧差、没有阻尼——这正是漂移的解药）。
- **ConfidenceHead**：linear(hidden,1)，q=σ(logit)，推理时 q≥q_th（默认 0.6）停。

## 2. 训练目标（`training_runtime.py:492-500`，与论文一致）

```python
loss_mono = SiLU()(lm_loss - lm_loss_prev)     # 越循环越差才受罚
q_loss    = BCEWithLogits(q_logit, target_q)   # target_q = 该轮 token 准确率(0/1)
loss      = lm_loss + cfg.beta * loss_mono + q_loss
```
- **随机深监督**：B 轮展开，仅采样 K 个深度回传梯度（其余 no_grad+detach），
  规避全 BPTT。
- 训练数据：FineWeb-Edu 流式（`create_streaming_dataloaders`），ctx 1024。

## 3. 与我们 recurrent_qwen.py 的差异表（移植要点）

| 项 | LoopUS | 我们 V1 | V1 失败归因 |
|---|---|---|---|
| 循环块选层 | 余弦距离分段动态 | 拍脑袋 24–29 | 可能选错块 |
| 迭代间过渡 | 选择性门（差驱动阻尼插值） | h₀ 重注入（LatentInjection） | 无阻尼→漂移 |
| "越循环越差" | L_mono 显式惩罚 | 无 | 增益为负无人管 |
| 深监督 | 每轮 LM loss + 随机 K 深度回传 | 全深度回传（显存大、梯度耦合） | 训练不稳 |
| 自适应深度 | 置信头 q≥0.6 停 | 无（固定深度） | 无 Pareto |
| KV cache | 循环模型可用（论文 Fig.3） | 不可用 | 部署慢 |

## 4. 运行方式备忘

- 依赖：torch≥2.10 / transformers≥5.0（主 venv 2.11/5.17 兼容，无需新环境；
  deepspeed/flashinfer 仅 Linux，Windows 自动排除）
- 官方权重：HF `Thrillcrazyer/Qwen3_1.7B_LoopUS`，经 `load_lds_model(
  decomposed_model=...)` 整模型加载（`LDSForCausalLM.from_pretrained`）
- 推理 CLI：`generate.py --model-name Qwen/Qwen3-1.7B --decomposed-model <path>
  --n-recursion 8 --max-new-tokens 100`
- 训练 CLI：`train.py`（accelerate；TrainConfig 参数见 training_cli.py）

## 5. 待验证问题（Phase 0-4 / G0）

1. 官方 1.7B 循环化模型在**我们严格评分协议**下是否可测出 Δ(d)>0？
2. 深度外推（N>训练 B）是否如论文所述稳定？
3. 其 tokenizer/格式：底模是 Qwen3-1.7B **Base**（无 chat 模板假设）？
   —— 影响 27B 方案的底模选择论证。

---

## 附录补充：论文全文核对（2026-09-21，arXiv HTML v1）

### 1. 论文没有报告训练饱和点
- 主实验：FineWeb-Edu **3B token 固定预算、1 epoch**（§4.1/附录 A.1），公开 1.7B 参考模板为 1.5B token。
- 发布脚本 `eval_interval=-1`（附录 A.2 明确"periodic validation is disabled"）；仅每 5000 优化步存档时跑 wikitext lm-eval（LIMIT=200）→ 全程仅 2–3 个评测点，**无训练进度曲线，无饱和点数据**。
- 论文的"饱和"是**推理深度维度**的（Figure 5：大部分收益在前几次迭代，d=40/80/100 外推稳定，自适应早停平均 3.39/8）——与我们的 CE(d) 曲线同类，与"训练步数饱和"无关。

### 2. 官方配方确认（B=20/K=5 与我们一致）
- 主实验统一 B=20、K=5、ctx=1024、AdamW 5e-5、cosine+warmup300、bf16、FlashAttention-2、1 epoch。
- 切分（附录 A.1）：1.7B: E=0..1/D=27；4B: E=0..1/D=35；8B: E=0..5/D=35；Phi-4: E=0..5/D=39；TinyLlama: E=0/D=21——**推理块覆盖 ~85–90% 网络层**（与我们 37.5% 的动力学选块哲学不同）。
- 推理 KV cache 方案（附录 A.4）：**每个循环深度独立缓存**（enc + {rea,b} + dec），prefill 一次后增量解码；1.64×/2.31×/2.49× 提速（1.7B/4B/8B @1024 tok）。

### 3. 重大发现：附录 D 已含 Qwen3.5-27B/35B-A3B 的表征动力学图
- Figure 16(c)(d)/17(c)(d)：PCA 轨迹 + 逐层距离 profile 覆盖 **Qwen3.5-27B 和 35B-A3B**。
- 意义：我们计划中的 S1a-27B 测量可直接与论文 Figure 17c **交叉验证**（他们确认了三段式结构在 Qwen3.5 上存在，但**没有做接缝规则、切分候选、循环轨迹验证**——那些是我们的增量）。

### 4. 重大发现：附录 F 明确把混合架构列为未解决的未来工作
原文（Limitations → Heterogeneous and hybrid model architectures）：
> "In such models, the middle-layer region may no longer behave as a uniform reusable block, and the optimal recursion policy may depend on the operator type, layer role, or token state. Future work should study architecture-aware recursion policies..."
- **论文作者亲口声明 LoopUS 不覆盖混合架构**（Gated DeltaNet、稀疏注意力等），并指出"middle-layer region may no longer behave as a uniform reusable block"。
- 我们的壁垒 1（接缝规则、相位门控 A3、动力学选块）正是对他们声明的开放问题的直接回答——差异化定位获得论文原文背书。

### 5. 机制理论（附录 B）与我们的范数发现互补
- 论文把门控形式化为"对角预条件的松弛不动点迭代"（Eq.21），配合单调性损失 = 任务对齐代理能量的下降过程。
- 我们的范数监控（膨胀率 +70%→+6%）为这个理论提供了他们没给的**定量证据**——可以写进后续报告作为独立贡献点。
