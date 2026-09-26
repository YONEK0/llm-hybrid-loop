"""s8_r1_watch.py — periodic S6/R1 acceptance (plain background command).

Sleeps --every minutes, runs s8_r1_accept.py (read-only: exit-head AUC +
turning-point archive presence), appends a line to runs/<run>/r1_watch.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=float, default=30)
    ap.add_argument('--run', default='v5_l1')
    args = ap.parse_args()
    if args.every > 0:
        print(f'[r1-watch] sleeping {args.every:g} min...', flush=True)
        time.sleep(args.every * 60)
    proc = subprocess.run(
        [sys.executable, str(ROOT / 's8_r1_accept.py'), '--run', args.run],
        cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print('[r1-watch] stderr:', proc.stderr.strip()[-300:], flush=True)


if __name__ == '__main__':
    main()
