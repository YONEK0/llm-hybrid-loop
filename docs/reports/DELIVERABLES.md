# 循环深度隐空间推理 — 最终交付物清单

项目路径：`E:\develop\project_files\llm-test\`

## 完成的实验

> **评分口径说明（2026-09-19）**：ProsQA 自由生成的宽松评分（旧 #1/#2 数字）被证实为
> 抽取伪影，已全面按严格口径重评并修订。教师强制（scored）协议数字不受影响。

| # | 实验 | 结果 | 数据文件 |
|---|---|---|---|
| 1 | Ouro 深度-准确率扫描（宽松+严格双口径） | 2.6B 严格单调 0%→13.3%→53.3%→53.3%；1.4B 0%→20% | `results/ouro_depth_sweep_Ouro-{1.4B,2.6B}.json` |
| 2 | Qwen3-4B + 课程训练（ProsQA，1200步） | 教师强制 k=0 EM 0%→25%；自由生成严格 0%（暴露偏差） | `results/pq_v1_fast_eval.json`（宽松，伪影样本） |
| 3 | Qwen3-4B + 课程训练（GSM8K，900步） | 教师强制 k3_d1 EM 53.3%；自由生成 0% | `results/cur-v2_scored_grid.json` |
| 4 | 评分伪影诊断（宽松 vs 严格） | 三条件 17/17"正确"全为复述伪影，严格 0/30，无误杀 | `results/extraction_artifact_diagnostic.json` |
| 5 | n=150 压缩扩展评测（含公平对照+McNemar） | 三条件逐题一致（+0/-0），伪影指纹；严格 0% | `results/compression_scaled_prosqa.json` |
| 6 | ProsQA 底模天花板（512/1024 token） | 0/30 完成（零 EOS），严格 0/30——原模型实际可达分 ≈0% | `results/prosqa_base_ceiling{,_512tok}.json` |
| 7 | GSM8K 底模基线（256 / 1024 token，n=100） | 256token：think 15%（15/15 为截断兜底伪影）；1024token：think 67%，其中 **64/100 EOS 内写完且全部正确**（严格天花板 64%，另 3 个为截断兜底伪影）；direct 2% | `results/base_model_baseline.json`、`results/base_model_baseline_256tok.json` |
| 8 | 消融：禁用训练参数 vs 训练后（GSM8K 教师强制） | 底模对照 63.3% > 训练后 53.3% | 控制台日志（JSON 缺失，见 RESEARCH_SUMMARY） |
| 9 | fixed-d4 固定深度4训练 | depth=1 40%，depth=4 36.7%（宽松口径） | `results/fixed-d4_scored_grid.json` |
| 10 | scheduled-sampling 直答微调（pd-v1，600 步） | **严格自由生成 EM 0/30（深度 1 与 4）**——暴露偏差 400+ 步未修复，本机训练杠杆穷尽 | `results/pd_v1_strict_eval.json`、`runs/pd-v1/` |

## 核心发现

> 本节是 V1 历史实验摘要；当前路线以 V5 为准。

1. **V1 采用的循环化配方需要预训练/大规模 uptraining 级算力**（旧估算 60B–7700B token），8GB 笔记本无法提供；这不是所有 LoopUS 式 retrofit 的普适下界
2. **V1 课程学习必须有显式脚手架**，直接训练隐空间全部失败
3. **评分伪影教训**：宽松文本抽取器会把"开头复述"记为正确——全部自由生成数字须用
   严格抽取器（仅认 #### 后文本或整行裸陈述句）复评
4. **原模型在 ProsQA 上的实际可达分 ≈0%**（1024 token 内写不完任何一题）
5. **深度增加在未正确训练的模型上会破坏表征**；Ouro-2.6B 严格深度曲线（0%→53.3%）
   证明正确循环训练能产生深度增益，但不能据此断言只有从零预训练才可行；LoopUS 先例
   是当前 V5 继续验证后训练路线的依据
6. ~~Token 压缩 3:1 可行~~（撤回：评分伪影）

## 可运行代码

| 脚本 | 用途 | 环境 |
|---|---|---|
| `eval_ouro.py` | Ouro-2.6B ProsQA 推理 | `ouro_env`（transformers 4.57.6） |
| `eval_ouro_depth.py` | Ouro 有效深度扫描（`exit_at_step`） | `ouro_env`（transformers 4.57.6） |
| `train_curriculum.py` | 脚手架课程训练 | 主 venv（transformers 5.17） |
| `eval_scored.py` | 训练一致协议评测 | 主 venv |
| `check_equivalence.py` | depth=1 等价性验证 | 主 venv |
| `eval_compression_scaled.py` | n=150 压缩评测（公平对照+Wilson CI+McNemar；宽松口径，严格结论见 #4） | 主 venv |
| `eval_prosqa_ceiling.py` | ProsQA 底模天花板（可调预算） | 主 venv |
| `diagnose_extraction.py` | 评分伪影诊断（宽松 vs 严格） | 主 venv |
| `train_direct.py` | scheduled-sampling 直答微调（修暴露偏差） | 主 venv |

## 下一步（V5，训练暂缓）

旧的 60B token / Qwen3-4B-Thinking 层 24–29 路线保留为历史交付记录，不是当前执行方案。
当前入口是 [`TRAINING_PLAN_V5_HYBRID_LOOPUS.md`](../plans/TRAINING_PLAN_V5_HYBRID_LOOPUS.md)：

1. 先完成 Qwen3.5-4B Hybrid 的 4-bit spike 和修正后的 S1a 表征动力学测量；
2. 用真实动力学边界确定 Encoder/Reasoning/Decoder，不把 3:1 架构边界直接当作切分；
3. 以 LoopUS 忠实的块边界单门为 A2 主线，以 full-attention 层后门控为 A3 对照；
4. 冻结严格评分协议，分别报告 teacher-forced scored 与 free-running；
5. 只有 S1/S2 通过后，才安排本机 gate-only/LoRA 方向性训练，随后再设计 Qwen3.8-27B 服务器训练。

本轮训练保持暂停，任何新运行必须先回填 V5 的前置结论。
