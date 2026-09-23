"""peek_log.py — zero-GPU lightweight behavioral check straight from history.jsonl.

Runs while training continues (no pause, no model load). Reviews everything the
trainer has already logged: eval CE curves, block-hidden-norm trend, A_bar/q_hat
health, and the greedy generation samples — the early-warning check for V1-style
"CE looks fine but output degenerates" failure.

Usage: loopus_env/Scripts/python.exe peek_log.py [--run v5_l0]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5_l0")
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run

    recs = []
    for ln in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    evals = [r for r in recs if r.get("event") == "eval"]
    batches = [r for r in recs if "logs" in r]
    baseline = next((r["dev_ce"] for r in recs if r.get("event") == "dev_baseline"), None)

    print(f"=== 轻量行为快检 {args.run}（零 GPU，训练不中断）===")
    print(f"batches={len(batches)} evals={len(evals)} "
          f"step={batches[-1]['step'] if batches else '-'}")

    print("\n[CE(d) 轨迹]")
    for e in evals:
        c = {k: v for k, v in e["dev_ce"].items() if k.startswith("d")}
        delta = ""
        if baseline:
            delta = "  Δ " + " ".join(
                f"{k}{v - baseline[k]:+.3f}" for k, v in sorted(c.items()) if k in baseline)
        print(f"  step {e['step']:>5}: " + " ".join(f"{k}={v}" for k, v in sorted(c.items()))
              + delta)

    norms = [(e["step"], e["dev_ce"].get("monitor", {}).get("block_hidden_norm"))
             for e in evals]
    norms = [(s, n) for s, n in norms if n]
    if norms:
        print("\n[块后范数（S1b 机制：冻结态 d1→d4 为 10.4→17.7）]")
        for s, n in norms:
            growth = (n.get("d4", 0) / n["d1"] - 1) * 100 if n.get("d1") else None
            print(f"  step {s:>5}: " + " ".join(f"{k}={v}" for k, v in n.items())
                  + (f"   d1→d4 +{growth:.1f}%" if growth is not None else ""))

    print("\n[最近生成样本（每评估点 2 条 × 12 token）]")
    for e in evals[-4:]:
        for s in e["dev_ce"].get("monitor", {}).get("samples", []):
            print(f"  step {e['step']} d={s['depth']}: {s['text'][:110]!r}")

    tail = batches[-150:]
    abar = [l["A_bar_mean"] for r in tail for l in r["logs"] if l.get("A_bar_mean")]
    gn = [l["grad_norm"] for r in tail for l in r["logs"] if l.get("grad_norm")]
    qh = [l.get("q_hat") for r in tail for l in r["logs"] if l.get("q_hat") is not None]
    print("\n[健康度]")
    if abar:
        print(f"  A_bar 近期均值 {sum(abar)/len(abar):.4f} "
              f"范围 [{min(abar):.4f}, {max(abar):.4f}]")
    if qh:
        print(f"  q_hat 近期均值 {sum(qh)/len(qh):.4f} "
              f"范围 [{min(qh):.4f}, {max(qh):.4f}]")
    if gn:
        print(f"  grad_norm 近期最大 {max(gn):.2f}")

    # simple degeneration heuristics on the newest samples
    print("\n[退化启发式检查]")
    flags = []
    newest = evals[-1]["dev_ce"].get("monitor", {}).get("samples", []) if evals else []
    for s in newest:
        t = s["text"].strip()
        if len(t) < 15:
            flags.append(f"样本过短({len(t)}字符)")
        if t and len(set(t.split())) <= 3:
            flags.append("样本词汇量极低(疑似重复退化)")
        if "�" in t:
            flags.append("样本含替换字符(解码异常)")
    print("  " + ("；".join(flags) if flags else "未发现退化信号（长度/词汇/解码均正常）"))
    print(f"\n结论样本总数 {sum(len(e['dev_ce'].get('monitor',{}).get('samples',[])) for e in evals)} 条")


if __name__ == "__main__":
    main()
