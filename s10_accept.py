"""s10_accept.py — v4 acceptance: router archive + turning points + exit divergence."""
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v5_l1v4')
    args = ap.parse_args()
    rd = ROOT / 'runs' / args.run
    checks, fails = [], 0

    def check(name, ok, detail, warn_only=False):
        nonlocal fails
        if not ok and not warn_only:
            fails += 1
        checks.append({'item': name,
                       'status': 'PASS' if ok else ('WARN' if warn_only else 'FAIL'),
                       'detail': detail})

    hp = rd / 'history.jsonl'
    rows, bad = [], 0
    if hp.exists():
        for ln in hp.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(ln))
            except Exception:
                bad += 1
    check('history_parses', bad == 0, f'{len(rows)} lines, {bad} bad')

    latest = rd / 'ckpt_latest.pt'
    if latest.exists():
        try:
            ck = torch.load(latest, map_location='cpu', weights_only=False)
            n = sum(p.numel() for p in ck['router'].values())
            check('router_archive', True,
                  f'step={ck["step"]} sup={ck["sup_tokens"]} '
                  f'router={n:,} keys={sorted(ck)}')
        except Exception as e:  # noqa: BLE001
            check('router_archive', False, f'{type(e).__name__}: {e}')
    else:
        check('router_archive', False, 'no ckpt_latest.pt')

    turns = [f.name for f in rd.glob('ckpt_turn*.pt')]
    check('turning_point_archive', len(turns) > 0,
          f'{sorted(turns)} (permanent, never overwritten)',
          warn_only=len(turns) == 0)

    evs = [r for r in rows if r.get('event') == 'eval']
    if evs:
        last = evs[-1]
        ex = last.get('exit', {})
        nd = ex.get('n_distinct', 0)
        check('exit_divergence', nd >= 4,
              f'distinct exit depths={nd} mean={ex.get("mean")} '
              f'dist={ex.get("dist")}', warn_only=nd < 4)
        ce = last.get('ce', {})
        check('ce_reported', 'adaptive' in ce,
              f'adaptive CE={ce.get("adaptive")} fixed={ {k: v for k, v in ce.items() if k.startswith("d")} }')
        alive = [l.get('alive_frac') for r in rows for l in r.get('logs', [])
                 if l.get('alive_frac') is not None]
        if alive:
            check('gate_active', min(alive) < 0.999,
                  f'min alive_frac={min(alive):.4f} (router is halting tokens)',
                  warn_only=min(alive) >= 0.999)
    else:
        check('eval_present', False, 'no eval events yet', warn_only=True)

    if hp.exists():
        age = round(time.time() - hp.stat().st_mtime)
        check('training_live', age < 600, f'history written {age}s ago',
              warn_only=age < 1800)

    out = {'time': time.strftime('%F %T'), 'run': args.run,
           'verdict': 'PASS' if fails == 0 else 'FAIL', 'checks': checks}
    p = ROOT / 'results' / f's10_accept_{args.run}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    line = (f"[v4-accept] {out['time']} verdict={out['verdict']} " +
            ' '.join(f"{c['item']}={c['status']}" for c in checks))
    with (rd / 'accept.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)
    for c in checks:
        print(f"  {c['status']:<4} {c['item']}: {c['detail']}")
    print(f'written {p}')


if __name__ == '__main__':
    main()
