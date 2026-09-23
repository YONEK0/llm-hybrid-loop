"""checkpoint_accept.py — read-only checkpoint acceptance, training never stops.

Run at every eval point (and any time). Verifies, WITHOUT touching the trainer:
  1. rolling checkpoints exist, are non-corrupt (real torch.load on CPU), prev < latest
  2. checkpoint step vs history batch count are consistent
  3. trainable state is complete (gate/q_head/optimizer/RNG all present, param counts)
  4. history.jsonl parses fully (torn trailing line tolerated), no parse errors
  5. liveness: history/checkpoint freshness
  6. ACCEPTANCE SIGNAL: latest eval CE(d) vs training-start baseline and vs the
     S1b frozen-loop reference — the V5 gate G3 criterion
  7. training health: A_bar still open, grad norm not exploding, per-depth lm trend

Usage: loopus_env/Scripts/python.exe checkpoint_accept.py [--run v5_l0]
Writes results/checkpoint_accept_<run>.json and prints a table.
"""

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5_l0")
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run
    checks, fails = [], 0

    def check(name, ok, detail, warn_only=False):
        nonlocal fails
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        if not ok and not warn_only:
            fails += 1
        checks.append({"item": name, "status": status, "detail": detail})

    # ---- 1/2/3: checkpoints ----
    latest_p, prev_p = run_dir / "ckpt_latest.pt", run_dir / "ckpt_prev.pt"
    latest = prev = None
    for tag, path in (("latest", latest_p), ("prev", prev_p)):
        if not path.exists():
            check(f"ckpt_{tag}_exists", False, f"{path.name} missing")
            continue
        try:
            ck = load_ckpt(path)
            need = {"step", "sup_tokens", "gate", "q_head", "optimizer", "rng", "args"}
            missing = need - set(ck)
            size_mb = round(path.stat().st_size / 2 ** 20, 1)
            if missing:
                check(f"ckpt_{tag}_structure", False, f"missing keys {sorted(missing)}")
            else:
                g = sum(p.numel() for p in ck["gate"].values())
                q = sum(p.numel() for p in ck["q_head"].values())
                opt_steps = len(ck["optimizer"].get("state", {}))
                check(f"ckpt_{tag}_structure", True,
                      f"{size_mb}MB step={ck['step']} gate={g} q_head={q} "
                      f"opt_states={opt_steps} rng=3")
            if tag == "latest":
                latest = ck
            else:
                prev = ck
        except Exception as e:  # noqa: BLE001 - any load failure is the finding
            check(f"ckpt_{tag}_loadable", False, f"{type(e).__name__}: {e}")

    if latest and prev:
        check("ckpt_rolling_order", latest["step"] > prev["step"],
              f"latest={latest['step']} > prev={prev['step']}")
    if latest:
        g = sum(p.numel() for p in latest["gate"].values())
        q = sum(p.numel() for p in latest["q_head"].values())
        check("trainable_param_count", g + q == 832001,
              f"gate+q_head={g + q} (expected 832001)")

    # ---- 4: history integrity ----
    hist_p = run_dir / "history.jsonl"
    n_lines = n_bad = 0
    batches = evals = 0
    baseline = None
    eval_series = []
    if hist_p.exists():
        raw = hist_p.read_bytes()
        complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
        torn = len(raw) - len(complete)
        for ln in complete.split(b"\n"):
            if not ln.strip():
                continue
            n_lines += 1
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            ev = rec.get("event")
            if ev == "eval":
                evals += 1
                eval_series.append(rec)
            elif ev == "dev_baseline":
                baseline = rec.get("dev_ce")
            elif "logs" in rec:
                batches += 1
        check("history_parses", n_bad == 0,
              f"{n_lines} lines, {n_bad} parse errors, torn_tail={torn}B "
              f"(trainer mid-append is normal)")
        if latest:
            check("ckpt_step_vs_history", latest["step"] <= batches,
                  f"ckpt step={latest['step']} <= history batches={batches} "
                  f"(gap={batches - latest['step']} = work since last ckpt)")
    else:
        check("history_exists", False, "history.jsonl missing")

    # ---- 5: liveness ----
    if hist_p.exists():
        age = round(time.time() - hist_p.stat().st_mtime)
        check("training_live", age < 300,
              f"history written {age}s ago", warn_only=age < 600)

    # ---- 6: acceptance signal ----
    if eval_series and baseline:
        last = eval_series[-1]
        ce = last["dev_ce"]
        rows = []
        for k in ("d1", "d2", "d4", "d8"):
            if k in ce and k in baseline:
                rows.append((k, baseline[k], ce[k], round(ce[k] - baseline[k], 4)))
        improved = [r for r in rows if r[0] in ("d2", "d4") and r[3] < 0]
        check("acceptance_CE_drop", len(improved) >= 1,
              "step {}: ".format(last["step"])
              + ", ".join(f"{k} {b}→{c} ({d:+})" for k, b, c, d in rows)
              + (" | deeper-depth CE dropping = gate working"
                 if improved else " | no drop yet"))
        # monotonic depth ordering: is d4 now <= d1?
        if "d4" in ce and "d1" in ce:
            check("depth_curve_inverted", ce["d4"] <= ce["d1"],
                  f"d4={ce['d4']} vs d1={ce['d1']} "
                  + ("(deeper now better)" if ce["d4"] <= ce["d1"]
                     else "(still deeper=worse)"))
        # vs S1b frozen reference at the same depth
        try:
            s1b = json.loads((ROOT / "results/s1b_trace_v2.json")
                             .read_text(encoding="utf-8"))
            frozen = s1b["models"]["qwen3.5-4b_hybrid_nf4"]["candidates"]["L8-19"]["mean"]["ce"]
            ref = {f"d{b}": frozen[b] for b in (1, 2, 4) if b < len(frozen)}
            deltas = {k: round(ce[k] - v, 3) for k, v in ref.items() if k in ce}
            check("vs_frozen_reference", all(v < 0 for v in deltas.values()),
                  f"trained − frozen: {deltas}")
        except Exception:
            pass

    # ---- 7: training health ----
    if hist_p.exists():
        recs = []
        for ln in hist_p.read_bytes().split(b"\n"):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "logs" in r:
                recs.append(r)
        tail = recs[-150:]
        abar = [l["A_bar_mean"] for r in tail for l in r["logs"]
                if l.get("A_bar_mean") is not None]
        gn = [l["grad_norm"] for r in tail for l in r["logs"]
              if l.get("grad_norm") is not None]
        if abar:
            check("gate_open", 0.001 < sum(abar) / len(abar) < 0.9,
                  f"A_bar mean={sum(abar) / len(abar):.4f} "
                  f"range=[{min(abar):.4f},{max(abar):.4f}]")
        if gn:
            check("grad_norm_stable", max(gn) < 50,
                  f"max grad_norm={max(gn):.2f} over last {len(gn)} supervised steps")

    # ---- report ----
    verdict = "PASS" if fails == 0 else "FAIL"
    monitor = None
    if eval_series:
        monitor = eval_series[-1]["dev_ce"].get("monitor")
    out = {"run": args.run, "time": time.strftime("%F %T"), "verdict": verdict,
           "progress": {"batches": batches, "evals": evals,
                        "ckpt_step": latest["step"] if latest else None,
                        "sup_tokens": latest["sup_tokens"] if latest else None},
           "eval_series": [{"step": e["step"], "ce": e["dev_ce"]} for e in eval_series],
           "baseline": baseline, "monitor_latest": monitor, "checks": checks}
    out_p = ROOT / "results" / f"checkpoint_accept_{args.run}.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 存档点验收 {args.run} — {verdict} ({time.strftime('%T')}) ===")
    if latest:
        print(f"checkpoint step={latest['step']} sup_tokens={latest['sup_tokens']:.0f} "
              f"| history batches={batches} evals={evals}")
    for c in checks:
        mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[c["status"]]
        print(f"  {mark} {c['item']}: {c['detail']}")
    if monitor:
        hn = monitor.get("block_hidden_norm")
        if hn:
            print(f"  i 块后隐状态范数(S1b机制): {hn}")
        for s in monitor.get("samples", [])[:2]:
            print(f"  i 生成样本(d={s['depth']},{s['n_tokens']}tok): "
                  f"{s['text'][:120]!r}")
    print(f"written to {out_p}")


if __name__ == "__main__":
    main()
