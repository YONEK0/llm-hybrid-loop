# HybridLoop — 自适应深度循环混合架构语言模型

**Recurrent-depth (looped transformer) post-training for hybrid-architecture LLMs**
混合架构（线性注意力 + 门控线性注意力 3:1）大模型的深度循环改造 + **逐 token 自适应退出路由**。
基座 Qwen3.5-4B-Base（冻结 NF4），只训练 LoRA + 门 + 路由器（5.28M 参数）。

> **核心成果（v4-c）**：逐 token 自适应深度退出成立——8 种退出深度、平均 4.8/8 圈（省 40% 算力）、
> teacher-forced CE 1.9213 **优于** base 2.0754、自由生成与 base 打平（1/6 vs 1/6）。
> 四轮实验（读出器→宽开门→硬冻结路由→软衰减路由）完整诊断链见结论表。

## 模型

**发行版**：[export/hybridloop-v0.1-hf/](export/hybridloop-v0.1-hf/) — 标准 HF 格式
（`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 直接加载；
权重以 95MB 分卷存储，复原见其中 REASSEMBLE.md）。vLLM 需带 trust_remote_code 插件接入。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(DIR, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(DIR)
out = model.generate(**tok("prompt", return_tensors="pt"), max_new_tokens=128)
model.set_exit_depth(4)   # 可选：固定深度（None = 自适应路由）
```

诚实限制：研究原型。无 KV cache（~1–8s/token）；GSM8K 严格分 ~17%（数量级算错为主）；
base-model 续写风格（非指令模型）；单种子小样本，数字均为方向性。

## 方法谱系

| 组件 | 来源 |
|---|---|
| 循环化框架 / SelectiveGate / B=20·K=5 | [LoopUS](https://arxiv.org/abs/2605.11011)（数学逐字） |
| 恒等首圈 + h₀ 再注入 | [Retrofitting Recurrent Depth](https://arxiv.org/abs/2608.11233)、Fan et al. |
| 逐圈 MLP 路由 + 单调掩码 + bottom-K 自监督 | [AdaPonderLM](https://arxiv.org/abs/2603.01914) |
| Δloss oracle 标签 | [RecurTrace](https://arxiv.org/abs/2609.03379)（R1 试验） |
| 块定位（M=L8-19） | 本项目流程：量化 plateau + 接缝规则 + 候选冻结对比（LoopUS 附录 D 做了定性观察，附录 F 声明混合架构未覆盖） |
| 宽开门初始化（A uniform[0.1,1.0]） | 本项目参数修改：LoopUS 原始 A 分布导致 99% 通道死区 |
| 软性状态衰减（SOFT_EPS=0.05） | 本项目修正：AdaPonderLM 冻结 KV；我们发现冻结状态本身损失 +0.27 CE，改为保留 5% 精化 |

## 关键实验结论（详见 docs/plans/…结论表）

- **深度反转**：0.26M token 出现，2.56M 认证——机制成本比 LoopUS 论文预算小三个数量级
- **token 级关闭门不可学**：Δloss oracle / prediction-space 全特征 AUC 0.51（22 万样本），
  根因=均匀混合无逐 token 分叉（R1/R2 双重验证）
- **自适应在序列级成立**：AdaPonderLM 式单调掩码路由制造逐 token 分叉——v4-c CE 净收益 + 省 40%
- **冻结损伤隔离**：硬冻结状态 +0.273，路由策略本身 0.000（三项对照试验）
- **深度收益是任务域 12% 尾巴**：gsm8k 逐块测试 2/16 块从 d5 获益（单块 −0.07 CE）

## 目录结构

```
s0…s11_*.py         V5 管线（spike→动力学→切分→训练→验收→关闭门战役）
s10_train_v4.py     v4 路由训练（单调掩码+bottom-K，转折点自动存档）
s10_accept.py       v4 验收（8 项：存档/分化/门活跃/活性）
dashboard.py        训练监控（独立进程，0.0.0.0:8741）
scorer_strict.py    评测协议 v1.1 严格评分器（42/42 对抗门）
export/             HF 标准导出（hybridloop-v0.1-hf，分卷权重+复原说明）
docs/               计划、笔记、报告（结论表 append-only）
runs/               训练历史 + 全部版本权重（L0/v2-T1/T2/v3/v4 系列已入 GitHub）
results/            实验数据（B 链、冻结对照、价值测试、读出诊断…）
legacy_v1/          第一代实验线（Ouro/课程学习，仅存档）
logs/               历史训练日志
data/               gsm8k(MIT)/prosqa 本地副本
```

## 环境

`loopus_env`（transformers 5.3.0 — 唯一原生支持 qwen3_5 的版本线），
完整快照见 `requirements-loopus-env.txt`。硬件基线：RTX 5060 8GB（NF4 冻结底座）。

## License

Apache-2.0。底座 Qwen3.5-4B-Base 权重归其原许可方所有。
