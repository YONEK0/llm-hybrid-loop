"""s6_r0_baseline.py — S6 R0: zero-training exit-rule Pareto on frozen loop.

Four exit rules simulated on one recorded trajectory pass (16 dev blocks,
d=1..20): velocity threshold, KL threshold (Huginn), patience-N (Zhou 2020),
LoopUS App.E pooled-state convergence rule. Fixed-depth d=4/8/16 as refs.
Writes results/s6_r0_pareto.json
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from s1b_trace_v2 import load_hybrid_nf4
from s2_spike import LoopUSQwen35

MAX_D = 20
N_BLOCKS = 16

ck = torch.load('runs/v5_l0/ckpt_latest.pt', map_location='cuda',
                weights_only=False)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
backbone = load_hybrid_nf4()
for p in backbone.parameters():
    p.requires_grad_(False)
model = LoopUSQwen35(backbone)
model.gate.load_state_dict(ck['gate'])
model.q_head.load_state_dict(ck['q_head'])
model.eval()

ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
text = '\n\n'.join(t for t in ds['text'] if t.strip())
ids = tok.encode(text)
blocks = [torch.tensor([ids[i * 600:(i + 1) * 600][:512]], device='cuda')
          for i in range(N_BLOCKS)]
blocks = [b for b in blocks if b.shape[1] == 512]
print(f'blocks: {len(blocks)}, recording d=1..{MAX_D}', flush=True)

# ---- one trajectory pass: record per-block per-depth quantities ----
records = []
with torch.no_grad():
    for bi, x in enumerate(blocks):
        h = model.encode(x)
        probs_prev = None
        curve = []
        for d in range(1, MAX_D + 1):
            h_new = model.gate(model.block(h), h)
            v = float((h_new - h).norm(dim=-1).mean()
                      / h.norm(dim=-1).mean())
            pooled = float((h_new.mean(1) - h.mean(1)).norm().item())
            h = h_new
            logits = model.decode(h)
            ce = float(F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                x[:, 1:].reshape(-1)))
            pred = logits[:, :-1].argmax(-1)
            acc = float((pred == x[:, 1:]).float().mean())
            last = int(logits[0, -1].argmax(-1))
            probs = logits[:, -50:, :].float().softmax(-1)
            kl = None
            if probs_prev is not None:
                p0 = probs_prev.clamp_min(1e-12)
                p1 = probs.clamp_min(1e-12)
                kl = float((p0 * (p0.log() - p1.log())).sum(-1).mean())
            probs_prev = probs
            curve.append({'d': d, 'v': v, 'pooled': pooled, 'kl': kl,
                          'ce': ce, 'acc': acc, 'last': last})
        records.append(curve)
        print(f'  block {bi}: d1 ce={curve[0]["ce"]:.3f} '
              f'd{MAX_D} ce={curve[-1]["ce"]:.3f}', flush=True)


# ---- simulate exit rules offline on recorded curves ----
def exit_depth(curve, fn, d_min=2):
    for c in curve:
        if c['d'] >= d_min and fn(curve, c):
            return c['d']
    return MAX_D


def pat_ok(curve, c, n):
    i = c['d'] - 1
    if i < n:
        return False
    return all(curve[i - k]['last'] == curve[i - k - 1]['last']
               for k in range(n))


def eval_rule(fn):
    ds_, ces, accs = [], [], []
    for curve in records:
        de = exit_depth(curve, fn)
        c = curve[de - 1]
        ds_.append(de)
        ces.append(c['ce'])
        accs.append(c['acc'])
    n = len(ds_)
    return {'avg_exit_depth': round(sum(ds_) / n, 2),
            'ce_at_exit': round(sum(ces) / n, 4),
            'acc_at_exit': round(sum(accs) / n, 4),
            'per_block_exit': ds_}


grids = {
    'velocity_thr': [0.003, 0.005, 0.01, 0.02, 0.04],
    'kl_thr': [0.0001, 0.0005, 0.001, 0.002],
    'patience': [1, 2, 3],
    'ruleE_eps': [0.02, 0.05, 0.1, 0.2, 0.5],
}
rules = {}
rules['velocity'] = [
    dict(param=t, **eval_rule(lambda cv, c, t=t: c['v'] < t))
    for t in grids['velocity_thr']]
rules['kl'] = [
    dict(param=t, **eval_rule(lambda cv, c, t=t: c['kl'] is not None
                              and c['kl'] < t))
    for t in grids['kl_thr']]
rules['patience'] = [
    dict(param=n, **eval_rule(lambda cv, c, n=n: pat_ok(cv, c, n)))
    for n in grids['patience']]
rules['ruleE'] = [
    dict(param=e, **eval_rule(lambda cv, c, e=e: c['pooled'] < e))
    for e in grids['ruleE_eps']]

fixed = {}
for d in (4, 8, 16, MAX_D):
    ces = [cv[d - 1]['ce'] for cv in records]
    accs = [cv[d - 1]['acc'] for cv in records]
    fixed[f'd{d}'] = {'ce': round(sum(ces) / len(ces), 4),
                      'acc': round(sum(accs) / len(accs), 4)}
ref_ce = fixed['d8']['ce']

out = {'ckpt_step': ck['step'], 'n_blocks': len(records), 'max_d': MAX_D,
       'fixed_depth': fixed, 'rules': rules,
       'note': ('R0 zero-training exit rules; CE gap = ce_at_exit - fixed d8 '
                'ce; Pareto target: low avg_exit_depth with small CE gap. '
                'patience-N: last-token argmax stable N consecutive loops. '
                'ruleE: pooled-state delta < eps (LoopUS App.E).')}
print(json.dumps({'fixed': fixed,
                  'rules': {k: [{'param': r['param'],
                                 'avg_exit': r['avg_exit_depth'],
                                 'ce': r['ce_at_exit'],
                                 'ce_gap_vs_d8': round(
                                     r['ce_at_exit'] - ref_ce, 4)}
                                for r in v] for k, v in rules.items()}},
                 indent=1), flush=True)
open('results/s6_r0_pareto.json', 'w', encoding='utf-8').write(
    json.dumps(out, indent=1))
print('written results/s6_r0_pareto.json', flush=True)
