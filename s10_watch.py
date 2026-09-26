"""s10_watch.py — periodic v4 acceptance (plain background command)."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=float, default=30)
    ap.add_argument('--run', default='v5_l1v4')
    args = ap.parse_args()
    if args.every > 0:
        print(f'[v4-watch] sleeping {args.every:g} min...', flush=True)
        time.sleep(args.every * 60)
    proc = subprocess.run(
        [sys.executable, str(ROOT / 's10_accept.py'), '--run', args.run],
        cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.strip(), flush=True)
    if proc.stderr.strip():
        print('[v4-watch] stderr:', proc.stderr.strip()[-300:], flush=True)


if __name__ == '__main__':
    main()
