"""s9_monitor.py — passive step-level monitor for L1-v3 (CPU-only, concurrent with training).

Per check, from runs/v5_l1v3/history.jsonl:
 1. eval series: CE(d) curve, best depth, switch_w, norm trend
 2. marginal depth value from training logs (lm vs supervision depth b)
 3. termination heuristic: best-depth flat AND marginal gain at depth>4 < 0.002
Appends a summary line to runs/v5_l1v3/monitor.log; writes results/s9_monitor.json.
Usage: loopus_env/Scripts/python.exe s9_monitor.py --every 15
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=float, default=15, help='minutes to sleep first')
    ap.add_argument('--run', default='v5_l1v3')
    args = ap.parse_args()
    if args.every > 0:
        print(f'[s9-monitor] sleeping {args.every:g} min...', flush=True)
        time.sleep(args.every * 60)

    run_dir = ROOT / 'runs' / args.run
    hp = run_dir / 'history.jsonl'
    rows = []
    for ln in hp.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    evs = [r for r in rows if r.get('event') in ('dev_baseline', 'eval', 'final')]
    series = []
    for r in evs:
        d = r.get('dev_ce', {})
        ces = {k: v for k, v in d.items()
               if k.startswith('d') and isinstance(v, (int, float))}
        if not ces:
            continue
        best = min(ces.items(), key=lambda x: x[1])
        mon = d.get('monitor', {})
        series.append({'step': r.get('step', 0),
                       'sup': r.get('sup_tokens', 0), 'ce': ces,
                       'best_depth': best[0], 'best_ce': best[1],
                       'switch_w': mon.get('switch_w', {}),
                       'norms': mon.get('block_hidden_norm', {})})

    # marginal depth value from recent training logs (lm at supervision depth b)
    logs = [(r['step'], l['b'], l['lm']) for r in rows for l in r.get('logs', [])]
    recent = logs[-600:]           # last ~120 steps worth
    marg = {}
    if recent:
        for step, b, lm in recent:
            marg.setdefault(b, []).append(lm)
        depth_means = {b: round(sum(v) / len(v), 4)
                       for b, v in sorted(marg.items()) if len(v) >= 8}

    # termination heuristic
    term = {'ready': False, 'reason': 'insufficient evals'}
    if len(series) >= 3:
        last3 = series[-3:]
        bests = [s['best_depth'] for s in last3]
        gaps = []
        for s in last3:
            ce = s['ce']
            if 'd4' in ce and 'd8' in ce:
                gaps.append(round(ce['d8'] - ce['d4'], 4))
        same_best = len(set(bests)) == 1
        flat = all(g <= 0.002 for g in gaps) if gaps else False
        if same_best and flat:
            term = {'ready': True,
                    'reason': f'best_depth stable at {bests[0]}, '
                              f'd8-d4 gap <0.002 for 3 evals {gaps}'}
        else:
            term = {'ready': False,
                    'reason': f'best={bests}, d8-d4 gaps={gaps}'}

    last_step = series[-1]['step'] if series else 0
    sup = series[-1]['sup'] if series else 0
    out = {'time': time.strftime('%F %T'), 'run': args.run,
           'last_step': last_step, 'sup_tokens': sup,
           'n_evals': len(series), 'series': series,
           'depth_marginal_lm': depth_means if recent else {},
           'termination': term}
    p = ROOT / 'results' / 's9_monitor.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    ce_str = ' '.join(f'{k}={v:.3f}' for k, v in
                      (series[-1]['ce'].items() if series else []))
    marg_str = ' '.join(f'b{b}:{m:.3f}' for b, m in list(depth_means.items())[:8]) \
        if recent else '-'
    line = (f"[s9] {out['time']} step={last_step} sup={sup / 1e6:.2f}M "
            f"| {ce_str} | best={series[-1]['best_depth'] if series else '-'} "
            f"| lm(b): {marg_str} | 终止判定: {term['ready']} ({term['reason']})")
    with (run_dir / 'monitor.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)
    print(f'written {p}')


if __name__ == '__main__':
    main()
