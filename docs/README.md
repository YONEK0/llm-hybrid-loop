# 循环深度隐空间推理项目 — 文档索引

> 项目根目录：`E:\develop\project_files\llm-test\`
> 本目录（`docs/`）收纳全部自研文档；根目录 `README_SIMCOT_UPSTREAM.md` 属于参考代码库
> SIM-CoT（原 README 改名保留），勿与本项目文档混淆。**项目入口文档是本文件。**

## 报告（docs/reports/）

| 文件 | 内容 |
|---|---|
| [RESEARCH_SUMMARY.md](reports/RESEARCH_SUMMARY.md) | **主报告**：研究总结 + 全部结果矩阵 + LoopUS 先例发现（从这里开始读） |
| [PROJECT_REVIEW.md](reports/PROJECT_REVIEW.md) | V1 训练过程与结果复盘（六类错误总结，重点文档） |
| [FINAL_REPORT.md](reports/FINAL_REPORT.md) | V1 最终报告（含 Ouro 深度曲线、评分效度修订） |
| [DELIVERABLES.md](reports/DELIVERABLES.md) | 交付物清单（实验 × 结果 × 数据文件对照表） |
| [RECURRENT_DEPTH.md](reports/RECURRENT_DEPTH.md) | 完整实验报告（含教学视频方法对照附录） |

## 规划（docs/plans/）

| 文件 | 内容 | 状态 |
|---|---|---|
| [TRAINING_PLAN_V5_HYBRID_LOOPUS.md](plans/TRAINING_PLAN_V5_HYBRID_LOOPUS.md) | **现行训练方案**：修正后的 LoopUS × Qwen3.5 Hybrid；先测动力学边界，再训练块边界门与 attention-layer 门控对照 | L0 训练进行中（step 1000+） |
| [DIFFERENTIATION_PLAN.md](plans/DIFFERENTIATION_PLAN.md) | **差异化壁垒计划（V6 前置）**：超越 LoopUS 的四条可证伪路线（方法学/成本/昇腾/部署），含实验判定门与结论槽 | 规划定稿，实验排队 |
| [TRAINING_PLAN_V4_HYBRID.md](plans/TRAINING_PLAN_V4_HYBRID.md) | V4 历史草案；保留讨论过程和旧实验设计 | 已被 V5 取代，旧 S1a/G0 表述不可作验收依据 |
| [TRAINING_PLAN_V3.md](plans/TRAINING_PLAN_V3.md) | LoopUS 纯复现路线（历史版本） | 复现部分需按 V5 的评分/对齐修订复核 |
| [TRAINING_PLAN_V2.md](plans/TRAINING_PLAN_V2.md) | （已废弃）Ouro 底模强化路线；备选分支保留 | 存档 |

## 笔记（docs/notes/）

| 文件 | 内容 |
|---|---|
| [LOOPUS_NOTES.md](notes/LOOPUS_NOTES.md) | LoopUS 源码解剖（SelectiveGate 实现、训练损失、与我们 V1 的差异表） |

## 相关目录速查

| 目录 | 内容 |
|---|---|
| `results/` | 全部实验数据 JSON（结论必须溯源到这里） |
| `runs/` | 训练检查点（pq-v1 / cur-v2 / pd-v1 等） |
| `models/` | 底模与循环化模型（Qwen3-4B、Ouro-1.4B/2.6B、LoopUS-Qwen3-1.7B） |
| `loopus/` | LoopUS 官方代码（clone） |
| `loopus_env` / `.venv` / `ouro_env` | 三个隔离环境（transformers 5.3.0 锁定 / 主 venv 5.17 / 4.57.6） |
| `Coconut/` `CODI/` | 参考代码库（根 `README_SIMCOT_UPSTREAM.md` 属于 SIM-CoT 参考库） |
