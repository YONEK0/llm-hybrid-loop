# 循环深度隐空间推理 — 最终研究报告

> **历史报告范围**：本文的负面结论针对 V1 的 Qwen3-4B-Thinking 本机改造方案。
> LoopUS 后续证明了另一套 Qwen3 后训练配方可在约 3B token 规模工作，因此本文的
> “60B–7700B”不能理解为所有循环化方法的普适下界。V5 是当前待执行的修订方案。

## 结论

在本机 8GB 笔记本 GPU 上完成了 V1 循环深度隐空间推理的技术验证。V1 的冻结底模、
低秩增量和小规模训练没有使指标接近原模型；这说明该配方在本机不可行，而不是证明
所有后训练循环化路线都不可行。

## 实验矩阵

| 实验 | 底模 | 方法 | ProsQA EM | 结论 |
|---|---|---|---|---|
| Coconut 课程训练 | GPT-2（论文原版） | 逐步移除思维链 | 97.0% | 隐空间推理在 GPT-2 上可行 |
| Ouro-2.6B 推理 | 循环预训练 7.7T | 预训练内置循环 | ~60% | 循环深度在正确训练下可行 |
| 本机 scaffold 课程 | Qwen3-4B-Thinking | 逐步减少脚手架 | 25% | 正面信号但低于思维链基线 56.7% |
| 本机 answer-only | Qwen3-4B-Thinking | 直接训练答案 | ~0% | 无脚手架完全失败（与 Coconut 消融一致） |
| 本机禁用参数 | Qwen3-4B-Thinking | 不训练（对照） | ~0% | 底模在 k=0 时无隐空间推理能力 |

## 关键技术发现

1. **循环深度要求预训练内置**：block 的输出必须是自己的合法输入（不动点性质），预训练权重不具备此性质。后训练改造需要 60B+ token 的 uptraining，本机不可行。
2. **Token 压缩 3:1 已验证**：Qwen3-4B-Thinking 经过课程训练后，用 16 token 达到 48 token 的同等准确率。
3. **暴露偏差是课程训练的主要瓶颈**：训练时 teacher-forcing 脚手架，推理时必须从零生成。移除脚手架后分布完全不同。
4. **课程学习必须有脚手架**：直接训练隐空间（无脚手架）在所有实验中都失败，与 Coconut 消融一致。

## 本机 vs 服务器算力对比

| | 本机 RTX 5060 8GB | 4× A100 320GB |
|---|---|---|
| 可训练参数 | ~24M（增量+LoRA） | 全量 4B |
| 训练 token | 0.18M | 60B+ |
| 训练时间 | 37 分钟 | ~4 天 |
| 预期结果 | 正面信号但不接近基线 | 接近或超过基线 |

## 评分效度修订（2026-09-19）

收尾审计发现：**ProsQA 自由生成的全部历史"正结果"由评分伪影构成**。宽松抽取器
（从后向前找第一个 `<Name> is a <thing>` 模式）把模型开头的"Okay, let's try to figure
out if Eva is a lorpus..."复述句当成答案；gold 恰好等于首个被提及候选类时即被记为
正确。诊断（`diagnose_extraction.py`）：base@48 / base@16 / trained@16 三条件各 17 个
"正确"全部来自首行复述，严格评分下 0/30，且 0/30 篇文本任何位置都不存在裸答案行
（无误杀）。n=150 扩展评测三条件逐题完全一致（McNemar +0/-0）——同一伪影指纹。

底模天花板重测：GSM8K 思考模式 256 token 仅 15% 且 15/15 为"截断取末数字"兜底伪影；
ProsQA 512 / 1024 token 均 **0/30 完成**（无一题在预算内生成结束），严格 0/30。
**原模型在 ProsQA 的实际可达分 ≈0%（严格口径）。**

修订后的核心数字（严格口径）：

| 配置 | 准确率 | Token |
|---|---|---|
| GSM8K 底模思考天花板（n=100） | **64.0%**（宽松 67%，含 3 个截断兜底伪影） | 785 |
| GSM8K 底模直答 @16 token | 2.0% | 16 |
| GSM8K 训练后教师强制 k=3 vs 底模同协议 | 53.3% vs 63.3%（**≈84%**，n=30） | — |
| GSM8K / ProsQA 训练后自由生成（严格） | **0.0%** | 16 |
| ProsQA 底模 @16 / 48 / 1024 token | 0.0%（0/30 在 1024 内写完） | — |
| ProsQA Ouro-2.6B 深度 4（预训练循环对照） | **53.3%** | 10.2 |
| ProsQA Ouro-1.4B 深度 4 | 20.0% | 12.5 |

结论不变但证据更干净：自由生成下训练改造未产生任何隐空间作答能力（scheduled-sampling
直答微调 pd-v1 又跑了 600 步，测试集严格 EM 仍 0/30——`results/pd_v1_strict_eval.json`）；
唯一正信号仍是教师强制 k=0 EM 25%；"接近或超过原模型"只有在循环架构 + 预训练级算力下
实现（Ouro-2.6B 严格 53.3% 且深度单调 0%→53.3%），本机 8GB 结构性不可达。

## 补充实验：Ouro 深度-准确率曲线（收尾）

`eval_ouro_depth.py`（`ouro_env`）利用 Ouro 前向的 `exit_at_step` 参数扫描有效循环深度
（ProsQA test 30 题，KV 缓存不变，只换读出隐状态）：

| 有效深度 | Ouro-2.6B 宽松 | Ouro-2.6B 严格 | Ouro-1.4B 宽松 | Ouro-1.4B 严格 |
|---|---|---|---|---|
| 1 | 26.7% | 0.0% | 46.7% | 0.0% |
| 2 | 40.0% | 13.3% | 30.0% | 0.0% |
| 3 | 56.7% | 53.3% | 36.7% | 10.0% |
| 4 | 60.0% | 53.3% | 36.7% | 20.0% |

严格口径下深度单调性成立且更干净：浅深度的"得分"全部是复述伪影，深度 3–4 才产生
真实最终答案。证实隐空间推理的深度增益只在"循环架构 + 海量预训练"下出现，是本研究
负面结论的最强正面对照。数据：`results/ouro_depth_sweep_Ouro-{1.4B,2.6B}.json`。

## 交付物清单

全部在 `E:\develop\project_files\llm-test\`：

| 文件/目录 | 内容 |
|---|---|
| `recurrent_qwen.py` | 循环深度模型（Qwen3 适配版） |
| `train_curriculum.py` | 脚手架课程训练（本机可用） |
| `eval_scored.py` / `eval_fast.py` | 评测脚本 |
| `check_equivalence.py` | 正确性守门（必须通过才能训练） |
| `runs/pq-v1/checkpoint/` | ProsQA 1200 步训练检查点 |
| `results/*.json` | 全部实验数据 |
| `RECURRENT_DEPTH.md` | 完整实验报告 |

## 引用

- Geiping et al., *Recurrent Depth* — arXiv:2502.05171
- Zhu et al., *Ouro: Looped Language Models* — arXiv:2510.25741（字节 Seed）
- Bae et al., *Relaxed Recursive Transformers* — arXiv:2410.20672（Google DeepMind）
- Hao et al., *Coconut: Chain of Continuous Thought* — arXiv:2412.06769（Meta）
- Shen et al., *CODI: Compressing CoT via Self-Distillation* — arXiv:2502.21074
- Deng et al., *iCoT: Internalize CoT Step by Step* — arXiv:2405.14838
