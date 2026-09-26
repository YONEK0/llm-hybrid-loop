"""s8_r2_readout.py — S6/R2: prediction-space readout + enriched exit gate.

Fixes R1's two failures:
1. Features: adds prediction-space signals (entropy, max_prob, margin, flip,
   hidden cosine) — literature's strongest readouts (arXiv 2607.20519 App H).
2. Label direction: PROSPECTIVE — y = 1[loss(d+1) < loss(d)] (R1 used
   retrospective "did d improve over d-1", wrong for exit decisions).

Layer 1 (zero-training): univariate AUC per signal + exit Pareto simulation.
Layer 2 (only if needed): trained heads on enriched features.

Usage: loopus_env/Scripts/python.exe s8_r2_readout.py --ckpt runs/v5_l1/ckpt_turn2_mature.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import L1Qwen35, inject_lora

ROOT = Path(__file__).resolve().parent


def build_blocks(tok, split, ctx, n_blocks):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    text = '\n\n'.join(t for t in ds['text'] if t.strip())
    ids = tok.encode(text)
    out = []
    for i in range(n_blocks):
        seg = ids[i * 600:(i + 1) * 600][:ctx]
        if len(seg) == ctx:
            out.append(torch.tensor([seg], device='cuda'))
    return out


@torch.no_grad()
def collect(model, blocks, max_d):
    """Forward-only. Returns per-depth dict of token-aligned tensors."""
    per_depth = {}
    for x in blocks:
        h0 = model.encode(x)
        h = h0
        prev_loss = None
        prev_argmax = None
        tgt = x[:, 1:]
        for d in range(1, max_d + 1):
            inp = h if d == 1 else h + h0
            h_prop = model.block(inp)
            h_old = h
            h_new = model.l1gate(h_prop, h_old, d)
            logits = model.decode(h_new)
            lf = logits[:, :-1].float()
            per_tok = F.cross_entropy(lf.reshape(-1, lf.shape[-1]),
                                      tgt.reshape(-1),
                                      reduction='none').view(tgt.shape)
            probs = F.softmax(lf, dim=-1)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
            mp = probs.max(-1).values
            top2 = lf.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            argmax = lf.argmax(-1)
            flip = torch.zeros_like(ent)
            if prev_argmax is not None:
                flip = (argmax != prev_argmax).float()
            cos = F.cosine_similarity(h_new[:, 1:], h_old[:, 1:], dim=-1)
            dn = (h_prop - h_old)[:, 1:].norm(dim=-1)
            if d in per_depth:
                pd = per_depth[d]
                for k, v in [('loss', per_tok), ('ent', ent), ('mp', mp),
                             ('margin', margin), ('flip', flip), ('cos', cos),
                             ('dn', dn)]:
                    pd[k] = torch.cat([pd[k], v.squeeze(0)])
            else:
                per_depth[d] = {k: v.squeeze(0).clone() for k, v in
                                [('loss', per_tok), ('ent', ent), ('mp', mp),
                                 ('margin', margin), ('flip', flip),
                                 ('cos', cos), ('dn', dn)]}
            prev_loss = per_tok
            prev_argmax = argmax
            h = h_new
    return per_depth


def cpu_auc(scores, labels):
    s_ = scores.detach().float().cpu()
    y_ = labels.detach().float().cpu()
    pos, neg = s_[y_ > 0.5], s_[y_ <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = torch.argsort(s_, stable=True)
    sr, yr = s_[order], y_[order]
    ranks = torch.arange(1, len(sr) + 1, dtype=torch.float32)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and float(sr[j + 1]) == float(sr[i]):
            j += 1
        if j > i:
            ranks[i:j + 1] = float((ranks[i] + ranks[j]) / 2)
        i = j + 1
    np_, nn_ = float(len(pos)), float(len(neg))
    return float((ranks[yr > 0.5].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))


def load_ckpt(model, path):
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
    ap.add_argument('--n-blocks', type=int, default=32)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--max-d', type=int, default=16)
    ap.add_argument('--auc-line', type=float, default=0.7)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()
    ck = load_ckpt(model, args.ckpt)
    print(f'loaded step={ck["step"]} | {args.n_blocks} blocks x d<={args.max_d}',
          flush=True)

    blocks = build_blocks(tok, 'validation', args.ctx, args.n_blocks)
    pd = collect(model, blocks, args.max_d)
    print('collected depths:', sorted(pd.keys()), flush=True)

    out = {'time': time.strftime('%F %T'), 'ckpt': args.ckpt,
           'ckpt_step': ck['step'], 'n_blocks': len(blocks),
           'max_d': args.max_d, 'layer1': {}, 'layer2': {}}

    # ---- Layer 1: prospective labels + univariate AUC per signal ----
    print('\n=== Layer 1: zero-training readout ===')
    for d in range(2, args.max_d):
        if d not in pd or (d + 1) not in pd:
            continue
        cur, nxt = pd[d], pd[d + 1]
        y = (nxt['loss'] < cur['loss']).float()
        feats = {'loss': cur['loss'], 'entropy': cur['ent'],
                 'max_prob': cur['mp'], 'margin': cur['margin'],
                 'flip': cur['flip'], 'h_cosine': cur['cos'],
                 'delta_norm': cur['dn'],
                 'depth': torch.full_like(y, float(d))}
        tag = f'd{d}'
        out['layer1'][tag] = {'pos_rate': round(float(y.mean()), 4), 'auc': {}}
        for fn, fv in feats.items():
            a = cpu_auc(fv, y)
            out['layer1'][tag]['auc'][fn] = round(a, 4) if a else None
        best = max(out['layer1'][tag]['auc'].items(), key=lambda x: x[1] or 0)
        print(f'  {tag}: pos={float(y.mean()):.3f} best={best[0]}'
              f' AUC={best[1]:.4f} | all: '
              + ' '.join(f'{k}={v}' for k, v in out['layer1'][tag]['auc'].items()
                         if v is not None), flush=True)

    p = ROOT / 'results' / 's8_r2_readout.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwritten {p}', flush=True)


if __name__ == '__main__':
    main()
