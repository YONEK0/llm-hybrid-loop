"""s8_r0_l1.py — R0 zero-training exit rules on the L1 T2 checkpoint.

Adapts s6_r0_baseline.py to L1 semantics (identity first loop, h0 reinjection,
depth-indexed switch). Same four exit rules simulated on one trajectory pass.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import L1Qwen35, inject_lora

ROOT = Path(__file__).resolve().parent
MAX_D = 20
N_BLOCKS = 16


def load_ckpt(model, path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    model.l1gate.gate.load_state_dict(ck['gate'])
    with torch.no_grad():
        model.l1gate.s.copy_(ck['switch'].to(model.l1gate.s.device))
        named = dict(model.backbone.named_parameters())
        for name, p in ck['lora'].items():
            named[name].copy_(p.to(named[name].device))
    return ck


@torch.no_grad()
def l1_forward(model, x, depth):
    h = model.encode(x)
    h0 = h
    for t in range(1, depth + 1):
        inp = h if t == 1 else h + h0
        h = model.l1gate(model.block(inp), h, t)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/v5_l1/ckpt_turn2_mature.pt')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()
    ck = load_ckpt(model, args.ckpt)

    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1',
                      split='validation')
    text = '\n\n'.join(t for t in ds['text'] if t.strip())
    ids = tok.encode(text)
    blocks = []
    for i in range(N_BLOCKS):
        seg = ids[i * 600:(i + 1) * 600][:512]
        if len(seg) == 512:
            blocks.append(torch.tensor([seg], device='cuda'))
    print(f'blocks={len(blocks)} d=1..{MAX_D} ckpt_step={ck["step"]}', flush=True)

    records = []
    with torch.no_grad():
        for bi, x in enumerate(blocks):
            h = model.encode(x)
            h0 = h
            probs_prev = None
            curve = []
            for d in range(1, MAX_D + 1):
                inp = h if d == 1 else h + h0
                h_prop = model.block(inp)
                v = float((h_prop - h).norm(dim=-1).mean()
                          / h.norm(dim=-1).mean())
                pooled = float((h_prop.mean(1) - h.mean(1)).norm().item())
                h = model.l1gate(h_prop, h, d)
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

    # simulate exit rules
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

    grids = {'velocity_thr': [0.003, 0.005, 0.01, 0.02, 0.04],
             'kl_thr': [0.0001, 0.0005, 0.001, 0.002],
             'patience': [1, 2, 3],
             'ruleE_eps': [0.02, 0.05, 0.1, 0.2, 0.5]}
    rules = {}
    rules['velocity'] = [dict(param=t, **eval_rule(
        lambda cv, c, t=t: c['v'] < t)) for t in grids['velocity_thr']]
    rules['kl'] = [dict(param=t, **eval_rule(
        lambda cv, c, t=t: c['kl'] is not None and c['kl'] < t))
        for t in grids['kl_thr']]
    rules['patience'] = [dict(param=n_, **eval_rule(
        lambda cv, c, n_=n_: pat_ok(cv, c, n_))) for n_ in grids['patience']]
    rules['ruleE'] = [dict(param=e, **eval_rule(
        lambda cv, c, e=e: c['pooled'] < e)) for e in grids['ruleE_eps']]

    fixed = {}
    for d in (4, 8, 16, MAX_D):
        ces = [cv[d - 1]['ce'] for cv in records]
        accs = [cv[d - 1]['acc'] for cv in records]
        fixed[f'd{d}'] = {'ce': round(sum(ces) / len(ces), 4),
                          'acc': round(sum(accs) / len(accs), 4)}
    ref_ce = fixed['d8']['ce']

    curves = [[c['ce'] for c in cv] for cv in records]
    out = {'ckpt_step': ck['step'], 'turns': ck.get('turns'),
           'per_block_ce_curves': curves,
           'n_blocks': len(records), 'max_d': MAX_D,
           'fixed_depth': fixed, 'rules': rules,
           'note': 'R0 on L1 T2 mature checkpoint'}
    print(json.dumps({'fixed': fixed,
                      'rules': {k: [{'param': r['param'],
                                     'avg_exit': r['avg_exit_depth'],
                                     'ce': r['ce_at_exit'],
                                     'gap_vs_d8': round(
                                         r['ce_at_exit'] - ref_ce, 4)}
                                    for r in v]
                                 for k, v in rules.items()}}, indent=1),
          flush=True)
    p = ROOT / 'results' / 's8_r0_l1.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'written {p}', flush=True)


if __name__ == '__main__':
    main()
