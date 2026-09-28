"""s8_r0_value_gsm.py — depth-value per-block test on GSM8K (task domain).

Answers: does any block benefit from loops beyond 2 on task text?
Decisive outputs: per-block argmin depth, fraction with best>2, CE(d) curve,
KL<0.001 exit-vs-best pairing.
"""
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import L1Qwen35, inject_lora, build_gsm_blocks
from s8_r0_l1 import load_ckpt

ROOT = Path(__file__).resolve().parent
MAX_D = 12


def main():
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()
    ck = load_ckpt(model, ROOT / 'runs/v5_l1v4c/ckpt_latest.pt')
    print(f'loaded v3 step={ck["step"]} | gsm8k blocks x d<={MAX_D}', flush=True)

    gsm_all, _n = build_gsm_blocks(tok, 512)
    blocks = [gsm_all[i].unsqueeze(0) for i in range(min(16, _n))]
    records = []
    with torch.no_grad():
        for bi, x in enumerate(blocks):
            h0 = model.encode(x)
            h = h0
            probs_prev = None
            curve = []
            for d in range(1, MAX_D + 1):
                inp = h if d == 1 else h + h0
                h_prop = model.block(inp)
                h = model.l1gate(h_prop, h, d)
                logits = model.decode(h)
                ce = float(F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    x[:, 1:].reshape(-1)))
                probs = logits[:, -50:, :].float().softmax(-1)
                kl = None
                if probs_prev is not None:
                    p0, p1 = probs_prev.clamp_min(1e-12), probs.clamp_min(1e-12)
                    kl = float((p0 * (p0.log() - p1.log())).sum(-1).mean())
                probs_prev = probs
                curve.append({'d': d, 'ce': ce, 'kl': kl})
            records.append(curve)
            print(f'  block {bi}: d1={curve[0]["ce"]:.3f} '
                  f'd{MAX_D}={curve[-1]["ce"]:.3f}', flush=True)

    def exit_kl(cv, thr=0.001):
        for c in cv:
            if c['d'] >= 2 and c['kl'] is not None and c['kl'] < thr:
                return c['d']
        return MAX_D

    print('\n=== GSM8K 逐块深度价值 ===')
    bests, wins = [], 0
    for i, cv in enumerate(records):
        best = min(range(1, MAX_D + 1), key=lambda d: cv[d - 1]['ce'])
        ex = exit_kl(cv)
        bests.append(best)
        win = cv[ex - 1]['ce'] <= cv[1]['ce']
        wins += int(win)
        print(f'  block {i:>2}: best={best:>2} exit_kl={ex:>2} '
              f'ce(best)={cv[best-1]["ce"]:.4f} ce(d2)={cv[1]["ce"]:.4f} '
              f'ce(d1)={cv[0]["ce"]:.4f} {"深度有益" if best > 2 else "浅优"}')
    agg = {d: round(sum(cv[d - 1]['ce'] for cv in records) / len(records), 4)
           for d in range(1, MAX_D + 1)}
    frac = sum(1 for b in bests if b > 2) / len(bests)
    print(f'\n聚合 CE(d): {agg}')
    print(f'最优深度分布: {{d: bests.count(d) for d in set(bests)}} = '
          f'{ {d: bests.count(d) for d in sorted(set(bests))} }')
    print(f'**决定性数字**: 深度>2 有益的块: {int(frac*len(bests))}/{len(bests)} '
          f'({frac*100:.0f}%)')
    print(f'退出 vs d2 配对胜负: {wins}/{len(bests)}')
    out = {'ckpt_step': ck['step'], 'agg_ce': agg,
           'per_block_best': bests,
           'frac_deep_beneficial': frac, 'exit_vs_d2_wins': wins,
           'curves': [[c['ce'] for c in cv] for cv in records]}
    p = ROOT / 'results' / 's8_r0_value_gsm.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'written {p}', flush=True)


if __name__ == '__main__':
    main()
