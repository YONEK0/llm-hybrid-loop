"""Plot S1a-v2 measurement data (results/s1a_dynamics_v2.json) -> results/figures/.

Fig 1 (hybrid): distance curve w/ layer types & peaks; logit-lens CE + piecewise fit +
plateau; lag-2..8 autocorrelation; P2 per-text transition groups.
Fig 2 (control + comparison): control distance/CE; CE plateau contrast at relative
depth; macro boundaries vs attention transitions (P3 non-coincidence).

Run: loopus_env/Scripts/python.exe plot_s1a.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results/s1a_dynamics_v2.json").read_text(encoding="utf-8"))
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

HY = DATA["models"]["qwen3.5-4b_hybrid"]
CT = DATA["models"]["qwen3-1.7b_std"]

C_LIN, C_ATT, C_MEAN = "tab:blue", "tab:orange", "tab:purple"


def dist_panel(ax, m, title, peaks):
    d = m["mean"]["dist_mean"]
    types = m["layer_types_modules"]
    L = len(d)
    for t in m["texts"]:
        ax.plot(range(L), t["dist_mean"], color="gray", alpha=0.18, lw=0.7, zorder=1)
    ax.plot(range(L), d, color=C_MEAN, lw=2, zorder=3, label="8 文本均值")
    lin_x = [i for i in range(L) if types[i] != "full_attention"]
    att_x = [i for i in range(L) if types[i] == "full_attention"]
    if lin_x:
        ax.scatter(lin_x, [d[i] for i in lin_x], s=18, color=C_LIN, zorder=4,
                   label="DeltaNet 层产生")
    ax.scatter(att_x, [d[i] for i in att_x], s=26, color=C_ATT, zorder=5,
               marker="D", label="全注意力层产生")
    for b in range(4, L, 4):
        ax.axvline(b - 0.5, color="k", alpha=0.10, lw=0.6)
    mu = sum(d) / len(d)
    ax.axhline(mu, color="gray", ls=":", lw=0.8)
    for p in peaks:
        ax.annotate(f"t{p}\n{d[p]:.3f}", (p, d[p]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="darkred")
    ax.set_title(title)
    ax.set_xlabel("transition t_i（由第 i 层产生）")
    ax.set_ylabel("相邻层余弦距离 1−cos")
    ax.legend(fontsize=8, loc="upper right")


def ce_panel(ax, m, title):
    ce = m["mean"]["lens_ce"]
    N = len(ce)
    ax.plot(range(N), ce, color="tab:green", lw=2, label="logit-lens CE")
    fit = m["P3_peaks_vs_macro"]["fits"]["3_seg"]
    breaks = fit["breaks_stage"]
    ys, segs = ce, [0] + breaks + [N]
    for lo, hi in zip(segs[:-1], segs[1:]):
        xs = list(range(lo, hi))
        n = len(xs)
        sx, sy = sum(xs), sum(ys[lo:hi])
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys[lo:hi]))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        inter = (sy - slope * sx) / n
        ax.plot([lo, hi - 1], [inter, inter + slope * (hi - 1)], color="darkred",
                ls="--", lw=1.4, label="三段拟合（BIC 最优）" if lo == 0 else None)
    for b in breaks:
        ax.axvline(b, color="darkred", alpha=0.5, lw=1,
                   label=f"宏观边界 stage {b}（t{b-1}）")
    ax.set_title(title)
    ax.set_xlabel("stage s（0=embedding，s≥1=第 s−1 层输出）")
    ax.set_ylabel("下一 token 交叉熵")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- fig 1: hybrid
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle("S1a-v2 表征动力学 — Qwen3.5-4B-Base（3:1 hybrid）", fontsize=13)

dist_panel(axes[0][0], HY, "相邻层余弦距离（灰=逐文本，紫=均值；峰 t2/t6/t19）",
           HY["P1_periodicity"]["peaks_produced_by_layer"])
ce_panel(axes[0][1], HY, "逐层 logit-lens CE：嵌入悬崖 → 平台 L2–21 → 锐化坡")

ac = HY["P1_periodicity"]["autocorr_lag2_8"]
lags = sorted(int(k) for k in ac)
axes[1][0].bar(lags, [ac[str(l)] for l in lags],
               color=["tab:red" if l == 4 else "tab:gray" for l in lags])
axes[1][0].axhline(0, color="k", lw=0.8)
axes[1][0].set_title("P1：距离曲线自相关（lag-4≈0 → 无周期-4 结构）")
axes[1][0].set_xlabel("lag（层）"); axes[1][0].set_ylabel("自相关")

gpt = HY["P2_transition_groups"]["groups_per_text"]
go = HY["P2_transition_groups"]["groups_overall"]
order = ["linear_attention->linear_attention", "linear_attention->full_attention",
         "full_attention->linear_attention"]
labels = ["L→L\n(DeltaNet 链内)", "L→F\n(注意力输出)", "F→L\n(注意力后一步)"]
for i, key in enumerate(order):
    xs = [i + (j - 3.5) * 0.06 for j in range(len(gpt[key]))]
    axes[1][1].scatter(xs, gpt[key], color=C_LIN if i == 0 else C_ATT,
                       alpha=0.6, s=22, zorder=2)
    m, (lo, hi) = go[key]["mean"], go[key]["ci95"]
    axes[1][1].errorbar([i], [m], yerr=[[m - lo], [hi - m]], color="k",
                        capsize=5, lw=1.4, zorder=3)
    axes[1][1].annotate(f"{m:.4f}", (i, m), textcoords="offset points",
                        xytext=(12, -3), fontsize=9)
ct_mean = CT["P2_transition_groups"]["groups_overall"][
    "full_attention->full_attention"]["mean"]
axes[1][1].axhline(ct_mean, color="tab:green", ls="--", lw=1,
                   label=f"对照 Qwen3-1.7B 全注意力层间 {ct_mean:.4f}")
axes[1][1].set_xticks(range(3), labels)
axes[1][1].set_ylabel("转移距离（8 文本散点 + 均值±95% bootstrap CI）")
axes[1][1].set_title("P2：DeltaNet 链内显著更小（8/8 文本同向，p=0.0078）")
axes[1][1].legend(fontsize=8)

fig.savefig(OUT / "s1a_hybrid.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: control
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle("S1a-v2 — 对照 Qwen3-1.7B（全注意力）与跨模型对比", fontsize=13)

dist_panel(axes[0][0], CT, "对照：距离曲线（无周期结构，仅 t3 一峰 + 出口跳变）",
           CT["P1_periodicity"]["peaks_produced_by_layer"])
ce_panel(axes[0][1], CT, "对照：CE 全程单调下降，无平坦平台")

for m, name, color in ((HY, "hybrid（32 层）", C_MEAN), (CT, "对照（28 层）", "tab:brown")):
    ce = m["mean"]["lens_ce"]
    N = len(ce)
    rel = [s / (N - 1) for s in range(N)]
    axes[1][0].plot(rel, ce, color=color, lw=2, label=name)
axes[1][0].set_ylim(0, 20)
axes[1][0].annotate("平台波动：hybrid 0.65 vs 对照 3.27（5 倍）",
                    (0.35, 17.5), fontsize=10, color="darkred")
axes[1][0].set_title("平台区对比（相对深度坐标，y 截断至 0–20）：hybrid 中段平坦，对照单调下降")
axes[1][0].set_xlabel("相对深度 stage/(L)"); axes[1][0].set_ylabel("logit-lens CE")
axes[1][0].legend(fontsize=9, loc="lower left")

d = HY["mean"]["dist_mean"]
axes[1][1].plot(range(len(d)), d, color=C_MEAN, lw=1.6, label="hybrid 距离曲线")
for t in HY["P3_peaks_vs_macro"]["attn_produced_transitions"]:
    axes[1][1].axvline(t, color=C_ATT, alpha=0.45, ls=":", lw=1.2)
axes[1][1].axvline(0, color="tab:red", lw=1.5, alpha=0.7, label="宏观边界 t1/t24")
for b in HY["P3_peaks_vs_macro"]["best_breaks_transition"]:
    axes[1][1].axvline(b, color="tab:red", lw=1.5, alpha=0.9)
axes[1][1].scatter([3, 7, 19], [d[3], d[7], d[19]], color=C_ATT, marker="D",
                   s=30, zorder=5, label="注意力跳变峰 t3/t7/t19")
axes[1][1].annotate("出口跳变 t31=0.315", (31, d[31]), textcoords="offset points",
                    xytext=(-80, -2), fontsize=9)
axes[1][1].set_title("P3：宏观边界（红）与注意力峰（橙）不重合")
axes[1][1].set_xlabel("transition t_i"); axes[1][1].set_ylabel("1−cos")
axes[1][1].legend(fontsize=8, loc="upper right")

fig.savefig(OUT / "s1a_control_and_comparison.png", dpi=150)
plt.close(fig)
print("written:", OUT / "s1a_hybrid.png")
print("written:", OUT / "s1a_control_and_comparison.png")
