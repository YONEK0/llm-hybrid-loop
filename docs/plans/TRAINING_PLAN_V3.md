# 训练规划 V3：LoopUS 路线（Qwen 系列循环化验证 → 27B 铺路）

> **历史版本，已由 `TRAINING_PLAN_V5_HYBRID_LOOPUS.md` 取代。** V3 的 G0/评测流程保留作
> 过程记录；ARC continuation 对齐、d1 PPL 异常和自由生成评分必须按 V5 的协议复核。
>
> 日期：2026-09-19。取代 `TRAINING_PLAN_V2.md`（V2′ Ouro 路线降级为备选分支，见 §8）。
> 依据：LoopUS 先例发现（`../reports/RESEARCH_SUMMARY.md` 末节）+ `../reports/PROJECT_REVIEW.md` 全部教训。

## 一、目标与定位

**终极目标**（用户定义）：Qwen3.8-27B 训练出循环深度（recurrent depth），
获得推理向的测试时算力增益。

**本规划范围**：本机 8GB 上的完整验证链，为 27B 服务器训练提供配方、
代码与风险结论。核心判定问题：

> LoopUS 配方（选择性门 + 单调性损失 + 随机深监督 + 分段选块）在**我们的
> 硬件、我们的底模、我们的数据**上，能否复现"深度增益 Δ(d)>0 且不掉点"？

## 二、路线依据（为什么是 LoopUS 而不是自研）

| 方案 | 训练量 | 结果 | 来源 |
|---|---|---|---|
| 本项目 V1（自研课程） | 0.18M | 深度增益为负，自由生成 0% | 本地实验 |
| McLeish retrofitting | 52B | **−2.9pp**（掉点） | arXiv:2511.07384 |
| Bae（Relaxed Recursive） | 60B | +3.5pp | ICLR'25 |
| **LoopUS** | **3B** | **+6.3pp**（TinyLlama 对比组） | arXiv:2605.11011 |

LoopUS 的底模矩阵含 Qwen3-1.7B/4B/8B（Qwen3-4B 与我们 V1 同款），代码与
循环化后的 Qwen3-1.7B 权重均已开源。**不重造轮子**：以官方 `train.py` 为主干，
我们的 `recurrent_qwen.py` 仅作对照实现与消融基座。

关键复现指标（论文报告值，作为我们的验收锚点）：
- Qwen3-4B：平均 +1.8pp（ARC-C +3.6 / OBQA +5.0），WikiText ppl 16.4→13.9
- 深度外推稳定（B=40/80/100 不崩），自适应平均深度 3.39/8

## 三、资源盘点

| 资源 | 状态 |
|---|---|
| GPU | RTX 5060 Laptop 8GB（训练峰值预算 ≤7 GiB） |
| LoopUS 代码 | github.com/Thrillcrazyer/LoopUS（经 gh-proxy.com 可达，已验证目录完整） |
| 循环化模型 | HF `Thrillcrazyer/Qwen3_1.7B_LoopUS`（待下载，约 3.4GB bf16） |
| 底模 | Qwen3-4B-Thinking（已有，注意：**LoopUS 用的可能是 Qwen3-4B-Base**，需核对） |
| 数据 | FineWeb-Edu 子集（论文同款；本地裁剪 0.1–0.5B token 规模） |
| 环境 | 新建 `loopus_env`（按其 pyproject/uv.lock 锁版本；主 venv 与 ouro_env 不动） |

## 四、阶段设计（每阶段有判定门，失败即停或降级）

### Phase 0：复现评测（零训练，~半天）

1. clone LoopUS（gh-proxy）+ 下载 `Qwen3_1.7B_LoopUS`；
2. 环境 spike：按 `pyproject.toml` 建 `loopus_env`，跑通 `test_model.py` /
   `generate.py`（8GB 推理 1.7B bf16 无压力，预计 3.5–4 GiB）；
3. **深度曲线复现**：用其 `evaluate.py`（或我们的严格评分协议）测
   Qwen3_1.7B_LoopUS vs 原版 Qwen3-1.7B 的 ARC-C/OBQA 子集，d∈{1,2,4,8}；
4. 检查其 KV cache 与置信头推理路径（论文称循环模型可用 KV cache，与我们
   V1 实现不同，需理解其机制——这是 27B 部署效率的关键）。

**判定门 G0**：本地复现出 Δ(d)>0（哪怕只有 +2pp）→ 进 Phase 1；
复现失败（增益为零/为负）→ 先排查评测差异，两次失败则停，结论为
"LoopUS 增益不可独立复现"，27B 计划风险升级。

### Phase 1：源码解剖与移植评估（1 天，纯 CPU/读码）

对照阅读 LoopUS `models/`（selective gate、confidence head 的实现细节）与
我们的 `recurrent_qwen.py`，产出：
1. `../notes/LOOPUS_NOTES.md`：四组件的精确实现（张量形状、初始化、超参默认值：
   B=20、K=5、A<0 的通道衰减、q_th 阈值）；
2. 选块分析复刻：对我们已下载的 Qwen3-4B 跑隐状态余弦距离分段分析，
   验证论文的"三段式"并得出我们自己的循环块边界（对照 V1 拍脑袋的 24–29）；
3. 决策：直接用官方 train.py（省力、保真）vs 移植组件进 recurrent_qwen.py
   （可控、可消融）。**默认前者**，后者仅当官方代码有硬性环境障碍时启用。

### Phase 2：本机训练验证（2–3 天，GPU 主战场）

**目标**：在 8GB 上跑通小规模 LoopUS 训练，验证"配方在我们手里也出增益"。

规模阶梯（逐级升，OOM 即退回上一级）：

| 级 | 底模 | 精度/方法 | 数据 | 预计可行性 |
|---|---|---|---|---|
| L1 | Qwen3-1.7B | bf16 + 随机深监督（其原生省内存设计） | 0.1B token | 峰值约 6–7 GiB，需 ctx≤1024、batch 1–2 + 累积 |
| L2 | Qwen3-1.7B | 4bit(NF4) + 仅训门/头/LoRA | 0.3B token | 显存充裕，但 4bit×自定义循环结构兼容性风险 |
| L3 | Qwen3-4B | 4bit + LoRA | 0.5B token | 与 27B 最近的本地代理 |

- 训练目标照抄论文：L_LM + L_mono + L_Q，AdamW + cosine，bf16；
- 数据 FineWeb-Edu 流式子集（论文同源，避免"我们数据特殊"的混淆变量）；
- **训练内监控（继承 V1 纪律）**：每 100 步存 dev CE(d) 曲线 + 严格 EM +
  3 条生成文本；固定对照组 {原版底模, LoopUS 官方 1.7B 检查点}；
- 所有自由生成评测用严格抽取器；评分器先过 20 条对抗样本效度门
  （`../reports/PROJECT_REVIEW.md` 错误一的纪律，不重蹈覆辙）。

**判定门 G2**（L1–L3 任一级达到即过）：
- 核心：训练后 Δ(d=4 vs 1) ≥ +2pp（ARC-C 或 OBQA 子集，n≥100）；
- 加分：CE(d) 单调下降、L_mono 训练中归零、自适应深度非平凡；
- 失败处置：若 L1 完整复刻论文设置仍无增益 → 结论"LoopUS 增益依赖未被
  论文披露的细节"，转 §8 备选分支并直接给 27B 计划打高风险标签。

### Phase 3：27B 服务器方案定稿（0.5 天，文档）

G2 通过后产出 `SERVER_PLAN_27B.md`：
- 配置：Qwen3.8-27B（Base，非 Thinking——Thinking 的 <think> 格式会与
  隐空间推理目标冲突，此为默认假设，Phase 1 时以论文底模选择佐证）；
- 训练：LoopUS 配方全参，3B token（FineWeb-Edu），4×A100-80G，
  按 L1 实测吞吐外推总时长（预估 1–2 天）；
- 评测：ARC-C/OBQA/GSM8K/ProsQA + 深度外推 B=40/80/100 + 自适应深度 Pareto；
- 回滚保护：单调性损失天然抑制"越循环越差"；保留置信头阈值扫描。

## 五、时间表（本机部分）

| 天 | 内容 |
|---|---|
| D1 上午 | Phase 0：clone + 下载 + 环境 spike |
| D1 下午 | Phase 0：深度曲线复现（G0） |
| D2 | Phase 1：源码解剖 + 选块分析 |
| D3–D4 | Phase 2：L1 训练 + 评测（G2） |
| D5 | L2/L3 或失败分析；Phase 3 文档 |

## 六、风险清单

| 风险 | 概率 | 缓解 |
|---|---|---|
| LoopUS 复现不出增益 | 中 | G0 早停；两次失败升级为"不可复现"结论 |
| 8GB 训练 OOM（L1） | 中 | ctx 降到 512、K 从 5 降到 2、梯度检查点 |
| 4bit × 循环结构不兼容（L2/L3） | 中高 | 退回 L1 结论外推；或 bf16+深度梯度检查点 |
| 论文底模是 Base 而我们只有 Thinking | 低 | Phase 0 核对其 HF 配置；必要时下载 Qwen3-4B-Base（4bit 约 2.5GB） |
| 自定义建模代码 × transformers 版本坑（V1 教训） | 中 | 严格用 loopus_env 锁定版本，不与主 venv 混用 |
| 增益幅度小于论文 | 高（常态） | 验收锚点设 +2pp 而非 +6.3pp（那是 TinyLlama 小底模的数字） |

## 七、与旧资产的关系

- `recurrent_qwen.py`：降级为对照实现/消融基座（其 LatentInjection+DeltaSet
  与 LoopUS 选择性门是同族不同种，Phase 1 产出精确差异表）；
- Ouro-2.6B：转为"预训练循环模型"参照系，用于校准我们的深度曲线评测协议；
- V1 全部评测纪律（严格评分、三对照组、对抗效度门、数字溯源）原样继承。

## 八、备选分支（G0/G2 失败时）

1. **V2′ Ouro 路线**（`TRAINING_PLAN_V2.md` 旧版内容）：对预训练循环模型
   做门校准/SFT/蒸馏——验证"循环性质保持"而非"从无到有"；
2. **McLeish 开源代码**（github.com/mcleish7/retrofitting-recurrence，
   HF 有 Recurrent-Llama-3.2 检查点）：其 52B token 课程法虽掉点，但代码
   可作为第二个独立参照，交叉定位"哪些组件是增益的必要条件"。

## 九、成功定义（本机阶段）

| 等级 | 标准 |
|---|---|
| 完整成功 | G0 + G2(L3) 双过：4B 级底模上 Δ(d)≥+2pp，27B 服务器方案定稿 |
| 基本成功 | G0 + G2(L1) 过：1.7B 上复现增益，27B 方案带风险标签定稿 |
| 部分成功 | 仅 G0 过：官方检查点本地可跑可信，配方移植完成但训练未出增益 |
| 失败 | G0 不过：LoopUS 不可复现，27B 计划搁置，转备选分支 |

任何等级的结论均写入 `../reports/PROJECT_REVIEW.md` 附录并归档（V1 教训：负结果
也要有协议、有对照、可溯源）。
