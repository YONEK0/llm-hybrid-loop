# HybridLoop 训练方法（v0.1 完整版）

## 概述

Qwen3.5-4B-Base（冻结 NF4）→ 循环化改造 → 自适应深度退出。
可训练参数 5.28M（LoRA 4.46M + 门 0.82M + 路由器 0.525M），主干全冻结。
训练数据：WikiText-2 70% + GSM8K 训练集 30%（本地混合，FineWeb-Edu 不可达的替代方案）。
硬件：RTX 5060 8GB（NF4 冻结使反向传播可行，峰值 7.8GB）。

## 方法谱系（引用核对后）

### 复用组件（来源论文）

| 组件 | 来源 | 引用 |
|---|---|---|
| SelectiveGate 数学 | LoopUS modeling_lds.py 逐字 | arXiv 2605.11011 |
| B=20/K=5 随机深度监督 | LoopUS 官方 train.sh 配方 | 同上 |
| β·SiLU 单调性损失 | LoopUS 论文 Eq.10 | 同上 |
| E/M/D 三段切分框架 | LoopUS 论文 §3 | 同上 |
| 逐圈独立 MLP 门 | AdaPonderLM 论文 §3 | arXiv 2603.01914 |
| 单调掩码 m←m⊙1(s≥τ) | AdaPonderLM 论文 §3 | 同上 |
| bottom-K ponder 损失 | AdaPonderLM 论文 §4（自监督，无标签） | 同上 |
| 恒等保留首圈 | Retrofitting Recurrent Depth | arXiv 2608.11233 |
| h₀ 每圈再注入 | Fan et al. H^(t)=Block(H^(t-1)+H^(0)) | Kuo et al. 引用 [11] |
| Δloss oracle 标签 | RecurTrace（R1 试验用，后续被 bottom-K 替代） | arXiv 2609.03379 |

### 本项目新增/修改（过程记录）

| 组件 | 描述 | 证据 |
|---|---|---|
| **块定位 M=L8-19** | 表征动力学平台检测（S1a）+ 模块对齐 F→L 接缝规则（S1b） | LoopUS 附录F声明混合架构未覆盖；对比论文用粗索引 |
| **宽开门 A uniform[0.1,1.0]** | LoopUS A_log=log(1..N) 导致 99% 通道死区；改为 uniform 使全部通道有开合范围 | s9_train_l1v3.py reset_A_log() |
| **软性状态衰减** | AdaPonderLM 冻结 KV（省算力）；我们加 (1-keep)·ε·Δh 让停圈 token 保持 5% 精化 | 对照试验：硬冻结 +0.27 CE，软衰减 0 |
| **评测协议 v1.1** | finished = EOS 或 ####数字后自然段界闭合 | 无论文先例（论文用 EOS 或 KL 阈值） |
| **转折点自动存档** | 按质量信号（深度反转/斜率塌缩/分化度）触发永久检查点 | 论文按固定步数存档 |
| **冻结 vs 策略分离试验** | 三项对照 A(全跑)/B(掩码记录)/C(掩码+冻结)隔离损伤来源 | 论文只报结果不做归因 |

### 合规声明

- 底座 Qwen3.5-4B-Base：Apache-2.0（可商用、可修改）
- LoopUS 开源代码：Apache-2.0（可复用、可修改，需注明来源）
- 本项目代码：Apache-2.0
- 论文引用方法（数学公式/架构构造）：学术引用即可，无许可限制
- **无合规风险项**：所有使用的第三方组件均为 Apache-2.0 或学术公开

## 训练流程（四代版本线）

### L0（5.00M token）→ 发现复读退化
- 只训 SelectiveGate + q_head（832k 参数）
- β=0.3，B=20/K=5，ctx=512，纯 WikiText
- 结果：深度反转 0.26M 成立，2.56M 认证，~3M 饱和
- **问题**：自由生成复读病（rep4=6.0 vs base 3.4）；起步税 +0.068 CE

### L1-v2（1.79M）→ 根治复读
- 加恒等首圈 + h₀ 再注入 + LoRA(r=8) + β warmup 0.3→1.0
- 数据换 wiki+gsm 7:3 混合
- T1 反转@0.77M、T2 成熟@1.79M 转折点自动存档
- **问题**：退出深度全部=3（假自适应）；token 级预测不可学（AUC 0.51）

### v3（0.58M）→ 宽开门
- A_log 重置为 uniform[0.1,1.0]，继承 v2 的 LoRA/switch/选择性
- A_log 用 1/10 慢学习率防回缩
- **发现**：退出深度分化到 5 种（KL<0.001），但价值测试揭穿=噪声尾巴

### v4-c（0.46M 路由器 + 继承 v3 权重）→ 自适应退出
- AdaPonderLM 式路由器：逐圈 MLP(hidden 256) + 单调掩码 + bottom-K(λ=0.02, k=0.05)
- **软衰减修正**：SOFT_EPS=0.05（硬冻结代价 +0.27 的对症解）
- 路由器独训 1.5M→总 2.5M：CE 全程 1.92 零退化，8 种深度保持

## 最终配方（v4-c）

```python
# 前向（每次生成一个 token）
h = encode(x)                # E 层（0-7）
h0 = h
mask = ones(...)             # 逐 token 存活标记
for t in 1..B:               # B=8
    inp = h if t==1 else h+h0
    h_prop = block(inp)      # M 层（8-19），LoRA 修改
    h_new = l1gate(h_prop, h, t)   # 恒等首圈 + SelectiveGate
    s = router(h_new, t)     # 逐圈 MLP → sigmoid
    keep = (s >= 1e-4) * mask
    h = h + keep*(h_new-h) + (1-keep)*0.05*(h_new-h)  # 软衰减
    mask = keep
logits = decode(h)           # D 层（20-31）
```

## 实测结果

| 指标 | 数值 | 对照 |
|---|---|---|
| 自适应 CE | 1.920 | v3 窄门基线 2.075 |
| 固定 d8 CE | 1.920 | 自适应=固定（策略零损失） |
| 平均退出深度 | 4.84 / 8 | 全跑 8 圈 → 省 40% |
| 退出深度种类 | 8 | 规则退出=1 种 |
| 冻结损伤 | +0.273 | 软衰减修复后=0 |
| 策略损伤 | 0.000 | 三项对照试验 |
| 自由生成 | 1/6 = base 1/6 | 打平（GSM8K 严格） |
| 深度收益尾巴 | 2/16 块（12%） | gsm8k 逐块测试 |

## 局限

- 无 KV cache（~1-8s/token）
- 任务能力=base 水平（循环不提升能力，只省算力）
- 深度收益 12% 尾巴（需 v5 轮联训或 27B 验证扩大）
- n=6-16 小样本，方向性结论
