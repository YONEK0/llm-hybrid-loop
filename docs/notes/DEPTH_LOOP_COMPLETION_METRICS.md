# 深度循环训练"完成"判据 — 文献调研汇总（2026-09-22）

问题：深度循环（looped/recurrent-depth）训练用什么指标判断"训完了"？
来源：Huginn (arXiv 2502.05171)、Kuo et al. (arXiv 2606.29983)、LoopUS (arXiv 2605.11011)、
Universal Transformer/ACT/PonderNet/DEQ 传统、以及本项目 v5_l0 实测。

## 0. 先说结论：业界没有单一"完成"指标，实际是三层判据

| 层 | 判据 | 谁在用 | 我们现状(v5_l0 step1300) |
|---|---|---|---|
| A 通用收敛 | 预算耗尽 或 留出 CE 斜率塌缩 | Huginn(800B预算)、LoopUS(3B预算) | ✓ 3M tok 后斜率塌缩20× |
| B 循环机制健康 | 深度曲线族 + 外推稳定 + 轨迹收敛 | Huginn Fig5/6、LoopUS Fig5、Kuo | ✓ 主要项全过，缺2项 |
| C 能力保持 | 基准≥同预算非循环对照 | Huginn Table4(非循环孪生)、LoopUS lm-eval | ✗ 未测（tier-1 待做） |

注意：Huginn 明说训练末期"语言建模收益在放缓，但数学/代码基准仍在稳定上涨"
（Fig 8）——**能力曲线常常到预算用完都没饱和**。所以论文们全是"预算完成制"，
把"饱和"判据留给机制层（B），这正是深度循环区别于普通 LM 的地方。

## 1. 通用收敛层（A）— 与普通 LM 训练相同

- 训练 ℒLM + 留出 CE/ppl 曲线，看**斜率**（每 N token 降幅 < ε）而非绝对值。
- Huginn 监控 loss spikes / 中断（750B tok "without notable interruptions or loss spikes"）。
- 训练/留出背离（过拟合）作反向指标。
- 我们实测：dev CE 均值每 10 万 token 降幅 0.068(快段) → 0.003(2M tok 后)；
  ℒLM 250k 分桶 3.53→2.45 且 1M tok 后基本走平。

## 2. 循环机制层（B）— 判断"循环真的被训出来了"，文献共 6 项

**B1 深度曲线族全程改善**（Huginn Fig 6 右图：val ppl @ r=1/4/8/16/32/64 随训练全深度改善）
   —— 我们的 dev CE(d=1/2/4) 每 25 步评测即同类。✓

**B2 深度增益持续存在（防"循环坍缩"）**——Huginn 的 Bad Run 2 判据：
   "val ppl 在 r=1 与 r=32 相同" = 模型学会忽略输入状态 s = 循环白训。
   正确信号是 CE(r_deep) < CE(r_shallow) 且差距稳定。✓（d4−d1=−0.043 稳定）

**B3 表征坍缩监控**——Huginn Bad Run 1：hidden state 的 token 维相关系数→1.0
   （每个 token 预测同一隐状态）= 坍缩。同族指标：每圈隐状态范数膨胀率。
   我们实测 +70%/圈(冻结) → +6%/圈(step1300)。✓

**B4 深度外推稳定**（LoopUS Fig 5：d=40/80/100 不崩；训练只见 d≤B，完成判据要外推到 d≫B）
   —— 我们 CE(d=1..32) FLAT（quality_check step1000/1300）。✓
   Kuo et al. 补充度量：**prediction flip rate** Pr[ŷ^{K+1}≠ŷ^K]（相邻深度预测翻转率，
   低=轨迹稳定）；以及 oracle-over-iterations acc vs 实际停止规则 acc 的差距。
   我们只有生成样本的定性"深度透明"，flip rate 未定量。△ 可补

**B5 轨迹收敛/不动点**（Huginn 自适应退出判据：相邻迭代输出 KL < 5e-4 即收敛退出；
   DEQ 的 fixed-point residual；Anil et al. 2022 path-independence）
   —— 判断循环算子是否学到"收缩到稳态"。我们 S1b 冻结侧证过"收缩–逃逸"机制，
   训练后未按 KL 阈值测过迭代收敛速度。✗ 可补（成本低：对 dev 块逐迭代测 KL）

**B6 单调性**（LoopUS SiLU 损失直接优化此项）：ℒ(b)−ℒ(b−1) ≤ 0 占比。
   我们实测 lm−lm_prev 自 step200 起持续为负。✓

## 3. 能力保持层（C）— 证明"没把模型练坏/真的更强"

- **同预算非循环对照**（Huginn Table 4：180B recurrent vs 180B fixed-depth 孪生）：
  这是"循环有没有用"的黄金判据 → 对应我们 DIFFERENTIATION_PLAN 的 E-STD。
- **基准套件**（lm-eval：ARC/HellaSwag/MMLU/OBQA/PiQA/SciQ/WinoGrande/GSM8K…）
  在选定部署深度下 ≥ 基座。LoopUS 用 wikitext lm-eval 存档点快检。
- **任务-深度饱和点**（Huginn Fig 7/9）：简单任务 8~12 圈饱和，难任务吃满 32+ 圈
  —— 判断"该分配多少深度"而非"是否训完"，但部署验收要用。
- **自适应退出校准**（LoopUS q_head / Huginn KL-exit 直方图）：
  早停 Pareto 好 = 训练把"何时可停"也学进去了。我们 q BCE 0.68~0.87 未收敛，✗ 待 E-SEL。
- **run-to-run 方差**（Kuo：循环模型 OOD 标准差 0.087~0.122 vs 普通 TF 0.005~0.009）：
  循环训练对种子/数据序敏感，"完成"结论至少要多种子或稳定性证据。✗ 单 run。

## 4. 对本项目的落地：v5_l0 三层完成判定

- **A 通用收敛**：判定"是"——3M tok 后 CE 斜率塌缩 20×；ℒLM 1M tok 后走平。
- **B 机制健康**：B1/B2/B3/B6 已过，B4 定性过（FLAT + 生成透明），B5/Kuo-flip-rate 未测
  （可选补测，各约半小时脚本）。
- **C 能力保持**：未测——**这是唯一真正挡住"训练完成"结论的层**。
  需要：(1) lm-eval final ckpt vs 基座（tier-1）；(2) E-STD 同预算非循环对照。

**推荐完成定义（写进验收标准）**：
1. 机制完成 = A + B 全项 → 现在已达成（step 1300 可宣告机制收敛）。
2. 预算完成 = 跑满 5M token（为 E-SCALE 预算阶梯保留与论文口径可比的端点）。
3. 能力完成 = C：lm-eval ≥ 基座且深度增益 Pareto 成立 → 训练全流程才算"完成"。

## 5. 引用锚点

- Huginn: Fig 5（坍缩诊断：token 相关系数、循环增益消失）、Fig 6（loss + val ppl@r 族）、
  Fig 7/9（任务-深度饱和）、Fig 8（基准-训练token 曲线，末期仍上涨）、
  §6.1（KL<5e-4 早停退出）、Table 4（非循环孪生对照）。
- Kuo et al. 2026: §3.3（oracle-over-iterations、Front.@90、Std.）、Fig 4（flip rate）、
  §4.3（稳定≠最优，需 acc–stability 双指标）。
- LoopUS: §4.1/附录 A（3B tok 预算制、eval_interval=-1、wikitext 存档点）、
  Fig 5（d=40/80/100 外推、早停 3.39/8）、附录 B（松弛不动点理论）。
