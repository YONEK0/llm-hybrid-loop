"""s7_tier1_a.py — L1 tier-1 A-chain: teacher-forced answer CE, GSM8K 300 items.

Paired with the L0 A-chain (results/s4_tier1_v5_l0.json): same corpus build,
same answer-segment CE, same paired-delta statistics. Configs:
- base   : single pass, gate/reinjection bypassed, adapters fresh (no-op)
- frozen : fresh L1Gate init + identity switch, adapters fresh (no-op)
- trained: full L1 ckpt (gate+switch+lora), L1 loop semantics
Order matters: base/frozen run on the fresh model BEFORE the ckpt is loaded.
Usage: loopus_env/Scripts/python.exe s7_tier1_a.py --ckpt runs/v5_l1/ckpt_turn2_mature.pt
Writes results/s7_tier1_a.json   (~85 min on the 8GB card)
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

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import L1Qwen35, L1Gate, inject_lora, loop_state, build_gsm_probe
from s4_tier1_eval import paired_ci

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def ce_one(model, ids, ms, depth):
    x = torch.tensor([ids], device='cuda')
    if depth is None:
        h = model.encode(x)
        logits = model.decode(model.block(h))
    else:
        logits = model.decode(loop_state(model, x, depth))
    seg = logits[0, ms - 1:-1].float()
    tgt = x[0, ms:]
    return float(F.cross_entropy(seg, tgt.reshape(-1),
                                 reduction="sum").item() / tgt.numel())


def load_l1_ckpt(model, path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    model.l1gate.gate.load_state_dict(ck['gate'])
    with torch.no_grad():
        model.l1gate.s.copy_(ck['switch'].to(model.l1gate.s.device))
        named = dict(model.backbone.named_parameters())
        for name, p in ck['lora'].items():
            named[name].copy_(p.to(named[name].device))
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/v5_l1/ckpt_turn2_mature.pt')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--depths', type=int, nargs='+', default=[1, 2, 4, 8])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    corpus = build_gsm_probe(tok, n_items=args.n)
    print(f'A-chain(L1): n={len(corpus)} depths={args.depths} '
          f'ckpt={args.ckpt}', flush=True)

    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()

    out = {'time': time.strftime('%F %T'), 'n': len(corpus),
           'depths': args.depths, 'ckpt': args.ckpt,
           'teacher_forced_ce': {}, '_per_item': {}}

    def sweep(tag, depth):
        ces = []
        t0 = time.perf_counter()
        for ids, ms in corpus:
            ces.append(ce_one(model, ids, ms, depth))
        key = tag if depth is None else f'{tag}:d{depth}'
        out['teacher_forced_ce'][key] = round(sum(ces) / len(ces), 4)
        out['_per_item'][key] = ces
        print(f'  [A] CE {key}: {out["teacher_forced_ce"][key]} '
              f'(n={len(ces)}, {time.perf_counter()-t0:.0f}s)', flush=True)

    print('== base (single pass, fresh adapters) ==', flush=True)
    sweep('base', None)
    print('== frozen (fresh L1Gate, fresh adapters) ==', flush=True)
    d_hidden = int(model.norm.weight.shape[0])
    model.l1gate = L1Gate(d_hidden, 20).cuda()
    model.eval()
    for dpt in args.depths:
        sweep('frozen', dpt)
    print('== trained (full L1 ckpt) ==', flush=True)
    ck = load_l1_ckpt(model, ROOT / args.ckpt)
    out['ckpt_step'] = ck['step']
    out['sup_tokens'] = ck['sup_tokens']
    out['turns'] = ck.get('turns')
    for dpt in args.depths:
        sweep('trained', dpt)

    deltas = {}
    base = out['_per_item']['base']
    for d in args.depths:
        t = out['_per_item'][f'trained:d{d}']
        fr = out['_per_item'][f'frozen:d{d}']
        deltas[f'trained-base:d{d}'] = paired_ci([a - b for a, b in zip(t, base)])
        deltas[f'trained-frozen:d{d}'] = paired_ci([a - b for a, b in zip(t, fr)])
    t1 = out['_per_item']['trained:d1']
    for d in args.depths:
        td = out['_per_item'][f'trained:d{d}']
        deltas[f'trained:d{d}-d1'] = paired_ci([a - b for a, b in zip(td, t1)])
    out['paired_deltas'] = deltas
    for k, v in deltas.items():
        print(f'  [delta] {k}: {v["mean"]} {v["ci95"]}', flush=True)

    out.pop('_per_item')
    p = ROOT / 'results' / 's7_tier1_a.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'written to {p}', flush=True)


if __name__ == '__main__':
    main()
