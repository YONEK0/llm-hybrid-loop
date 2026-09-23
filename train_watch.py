"""train_watch.py — periodic training check driven by background-task notifications.

NOT a ZCode cron/scheduled task: this is a plain background command. It sleeps
--every minutes, performs a READ-ONLY checkpoint acceptance (never touches the
trainer), prints a compact status, appends the same line to runs/<run>/watch.log,
and exits. The background-task completion notification re-invokes the agent, which
reports and relaunches this — the loop is the "schedule".

Usage:  loopus_env/Scripts/python.exe train_watch.py [--every 30] [--run v5_l0]
Stop:   kill the background task (or tell the agent to stop relaunching).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=30, help="minutes to sleep first")
    ap.add_argument("--run", default="v5_l0")
    args = ap.parse_args()

    if args.every > 0:
        print(f"[watch] sleeping {args.every:g} min before check "
              f"(training untouched)...", flush=True)
        time.sleep(args.every * 60)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checkpoint_accept.py"), "--run", args.run],
        cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr.strip():
        print("[watch] stderr:", proc.stderr.strip()[-500:])

    acc_p = ROOT / "results" / f"checkpoint_accept_{args.run}.json"
    acc = None
    try:
        acc = json.loads(acc_p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[watch] could not read acceptance json: {e}")

    if acc is None:
        line = f"{time.strftime('%F %T')} verdict=? (acceptance json missing)"
        finished = stalled = False
    else:
        p, series = acc["progress"], acc["eval_series"]
        last = series[-1] if series else None
        dvals = {k: v for k, v in sorted(last["ce"].items())
                 if k.startswith("d") and isinstance(v, (int, float))} if last else {}
        ce = ", ".join(f"{k}={v:.3f}" for k, v in dvals.items())
        mon = last["ce"].get("monitor", {}) if last else {}
        if mon.get("block_hidden_norm"):
            hn = mon["block_hidden_norm"]
            ce += " | ‖h‖ " + ",".join(f"{k}={v:.1f}" for k, v in hn.items())
        line = (f"{time.strftime('%F %T')} verdict={acc['verdict']} "
                f"step={p['batches']} evals={p['evals']} "
                f"sup={p['sup_tokens']:.0f} | {ce}")
        alerts = [f"FAIL:{c['item']}" for c in acc["checks"]
                  if c["status"] == "FAIL"]
        if alerts:
            line += " | ALERTS " + ",".join(alerts)
        finished = p["sup_tokens"] >= 5e6
        stalled = any(c["item"] == "training_live" and c["status"] != "PASS"
                      for c in acc["checks"])

    log_p = ROOT / "runs" / args.run / "watch.log"
    with log_p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"[watch] {line}")
    flags = ("TRAINING COMPLETE " if finished else "") + \
            ("STALLED" if stalled else "")
    print(f"[watch] check took {time.perf_counter() - t0:.1f}s {flags}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
