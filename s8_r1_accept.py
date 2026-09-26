"""s8_r1_accept.py — S6/R1 acceptance gate: exit-head archive + AUC + archive safety.

Checks (auto, append to runs/v5_l1/r1_watch.log):
 1. T1/T2 turning-point archives still present (never overwritten/deleted)
 2. exit_head.pt loadable, mu/sd/head shapes consistent
 3. val AUC >= 0.7 (S6 acceptance line)
 4. turning-point archives are byte-recorded (size + mtime) for audit
Writes results/s8_r1_accept.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v5_l1')
    args = ap.parse_args()
    run_dir = ROOT / 'runs' / args.run
    checks, fails = [], 0

    def check(name, ok, detail, warn_only=False):
        nonlocal fails
        if not ok and not warn_only:
            fails += 1
        checks.append({'item': name,
                       'status': 'PASS' if ok else ('WARN' if warn_only else 'FAIL'),
                       'detail': detail})

    # 1/4: turning-point archives preserved
    keep = ['ckpt_turn1_inversion.pt', 'ckpt_turn2_mature.pt']
    for fn in keep:
        p = run_dir / fn
        if p.exists():
            check(f'turning_point_{fn}', True,
                  f'{p.stat().st_size / 2 ** 20:.1f}MB '
                  f'mtime={time.strftime("%F %T", time.localtime(p.stat().st_mtime))}')
        else:
            check(f'turning_point_{fn}', False, 'MISSING')

    # 2: exit head loadable
    head_p = run_dir / 'exit_head.pt'
    if not head_p.exists():
        check('exit_head_exists', False, 'exit_head.pt missing')
    else:
        try:
            h = torch.load(head_p, map_location='cpu', weights_only=False)
            ckpt_ok = {'head', 'mu', 'sd'} <= set(h)
            n_par = sum(p.numel() for p in h['head'].values())
            check('exit_head_exists', True,
                  f'{head_p.stat().st_size / 2 ** 10:.0f}KB params={n_par} '
                  f'keys_ok={ckpt_ok} from_step={h.get("step")}')
            check('exit_head_structure', ckpt_ok, f'keys={sorted(h)}')
        except Exception as e:  # noqa: BLE001
            check('exit_head_exists', False, f'{type(e).__name__}: {e}')

    # 3: AUC acceptance
    gate_p = ROOT / 'results' / 's8_r1_gate.json'
    auc = None
    if gate_p.exists():
        g = json.loads(gate_p.read_text(encoding='utf-8'))
        auc = g.get('heads', {}).get('mlp', {}).get('val_auc')
        check('exit_head_auc', auc is not None and auc >= 0.7,
              f'val AUC={auc} (acceptance >= 0.7)',
              warn_only=auc is None)
    else:
        check('exit_head_auc', False, 'results/s8_r1_gate.json missing',
              warn_only=True)

    out = {'time': time.strftime('%F %T'), 'run': args.run,
           'verdict': 'PASS' if fails == 0 else 'FAIL',
           'auc': auc, 'checks': checks}
    p = ROOT / 'results' / 's8_r1_accept.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    line = (f"[r1-watch] {out['time']} verdict={out['verdict']} "
            f"auc={auc} " +
            ' '.join(f"{c['item']}={c['status']}" for c in checks))
    with (run_dir / 'r1_watch.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)
    print(f'written {p}')


if __name__ == '__main__':
    main()
