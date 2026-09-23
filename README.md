# llm-hybrid-loop

**Recurrent-depth (looped transformer) post-training for hybrid-architecture LLMs** —
混合架构（线性注意力 + 全注意力 3:1）大模型的深度循环改造：动力学选块、混合门训练、
关闭门（halting gate）与预算阶梯评测。Qwen3.5-4B 为首个实验载体，Qwen3.8-27B 为下一目标。

> 方法谱系：[LoopUS](https://arxiv.org/abs/2605.11011)（循环化后训练配方）；
> 本项目增量 = 混合架构原生的块定位与接缝规则（S1a/S1b）、门控修复、退出基线与三层完成判据——
> LoopUS 论文附录 F 明确声明混合架构未覆盖，为本文档所填的开放问题。

## 当前状态（2026-09）

| 阶段 | 结果 |
|---|---|
| S1a/S1b 块定位 | 表征动力学平台 L2–L21 + 模块对齐 F→L 接缝 → **M=L8-19 定案** |
| S3 L0 训练 | 只训门+头 832k 参数 × 5M token：深度反转 0.26M、全项认证 2.56M、饱和 ~3M（比论文预算小三个数量级的机制成本） |
| tier-1 能力验收 | A 链：base 0.745 最优，trained 起步税 +0.068、深度斜率 +0.014，frozen 爆炸（门训练压制 97% 漂移）；B 链：自由生成复读退化（L1 v2 主攻方向） |
| S6 R0 退出基线 | 零训练阈值规则：5.25 圈退出 / CE +0.002 —— 省 ~35% 算力近零损失 |
| q_head 诊断 | 联合训练失败（AUC 0.53，只学会数圈数）→ S6/R1 冻结轨迹 post-hoc 重训方案已定 |

## 目录结构

```
docs/plans/    执行计划（V5 主计划、差异化壁垒、完成判据调研）
docs/notes/    论文核对、方法笔记
docs/reports/  阶段报告
s0..s6_*.py    管线脚本（spike→动力学→切分→训练→验收→退出基线）
s3_train_l0.py 存档式训练器（原子检查点/滚动双份/断点续训位一致）
dashboard.py   训练监控网站（独立进程，0.0.0.0:8741）
scorer_strict.py 冻结协议严格评分器（20 条对抗样本门）
runs/          训练历史（history.jsonl 随仓库；*.pt 存档不入库）
results/       全部实验 JSON 与图
data/          gsm8k（MIT）/prosqa 本地副本与来源清单
```

## 环境

`loopus_env`（transformers 5.3.0，唯一原生支持 qwen3_5_text 的版本线）——
完整依赖见 `requirements-loopus-env.txt`。硬件基线：RTX 5060 8GB（NF4 冻结底座）。

## 关键文档

- `docs/plans/TRAINING_PLAN_V5_HYBRID_LOOPUS.md` — 主计划与 append-only 结论表
- `docs/plans/DIFFERENTIATION_PLAN.md` — 差异化壁垒与证伪实验
- `docs/notes/DEPTH_LOOP_COMPLETION_METRICS.md` — 三层完成判据（收敛/机制/能力）

## License

Apache-2.0（继承上游 LoopUS 仓库协议）。
