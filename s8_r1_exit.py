"""s8_r1_exit.py — S6/R1: post-hoc exit gate on FROZEN L1 trajectories.

Loop block (LoRA), SelectiveGate and depth switch are all frozen. We collect
per-token per-loop features from a forward-only pass, label each (depth, token)
with the DELTA-LOSS oracle y = 1[loss(token at d+1) < loss(token at d)]
(RecurTrace, arXiv 2609.03379), then fit small heads (linear + 2-layer MLP).

Features deliberately include the raw delta norm (NOT LayerNorm-ed): the q_head
AUC failure was partly blamed on LN erasing the magnitude signal.

Acceptance (S6): val AUC >= 0.7 AND early-exit Pareto beats the R0 zero-training
frontier (avg exit ~5.25 loops, CE gap +0.002 on the v5_l0 model; re-measured
here on the same L1 model so the comparison is like-for-like).

Usage: loopus_env/Scripts/python.exe s8_r1_exit.py --ckpt runs/v5_l1/ckpt_turn2_mature.pt
Writes results/s8_r1_gate.json + runs/v5_l1/exit_head.pt
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
from s7_train_l1 import L1Qwen35, L1Gate, inject_lora

ROOT = Path(__file__).resolve().parent
MAX_D = 16


def build_blocks(tok, split, ctx, n_blocks, offset=0):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    text = '\n\n'.join(t for t in ds['text'] if t.strip())
    ids = tok.encode(text)
    blocks = []
    for i in range(offset, offset + n_blocks):
        seg = ids[i * 600:(i + 1) * 600][:ctx]
        if len(seg) == ctx:
            blocks.append(torch.tensor([seg], device='cuda'))
    return blocks


@torch.no_grad()
def collect(model, blocks, max_d):
    """Forward-only: per (loop, token) features + delta-loss oracle labels."""
    feats, labels, depths = [], [], []
    for x in blocks:
        h0 = model.encode(x)
        h = h0
        prev_loss = None
        for d in range(1, max_d + 1):
            inp = h if d == 1 else h + h0
            h_prop = model.block(inp)
            delta = h_prop - h
            h_new = model.l1gate(h_prop, h, d)
            logits = model.decode(h_new)
            tgt = x[:, 1:]
            per_tok = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                tgt.reshape(-1), reduction='none').view(tgt.shape)
            h = h_new
            if d >= 2:                      # label: does one more loop help?
                win = (per_tok < prev_loss).float()
                dn = delta.norm(dim=-1)
                hn = h.norm(dim=-1)
                a_bar = model.l1gate.gate.last_A_bar_mean
                sl = slice(1, None)     # align with tgt = x[:, 1:]
                f = torch.stack([
                    dn.squeeze(0)[sl],
                    hn.squeeze(0)[sl],
                    torch.full_like(dn.squeeze(0)[sl], float(a_bar)),
                ], dim=-1)
                feats.append(f)
                labels.append(win.squeeze(0))
                depths.append(torch.full_like(dn.squeeze(0), float(d)))
            prev_loss = per_tok
    return (torch.cat(feats), torch.cat(labels),
            torch.cat(depths))


class MLPHead(nn.Module):
    def __init__(self, nf, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc(scores, labels):
    """Rank-based AUC on CPU (avoids a device-side assert from in-place rank
    scattering on this card)."""
    s_ = scores.detach().float().cpu()
    y_ = labels.detach().float().cpu()
    pos = s_[y_ > 0.5]
    neg = s_[y_ <= 0.5]
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
    n_pos, n_neg = float(len(pos)), float(len(neg))
    return float((ranks[yr > 0.5].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))



def load_ckpt(model, path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    model.l1gate.gate.load_state_dict(ck['gate'])
    with torch.no_grad():
        model.l1gate.s.copy_(ck['switch'].to(model.l1gate.s.device))
        named = dict(model.backbone.named_parameters())
        for name, p in ck['lora'].items():
            named[name].copy_(p.to(named[name].device))
    return ck


def split_doc(n_blocks, frac=0.75):
    n_tr = max(1, int(n_blocks * frac))
    return list(range(n_tr)), list(range(n_tr, n_blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/v5_l1/ckpt_turn2_mature.pt')
    ap.add_argument('--n-blocks', type=int, default=64)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--max-d', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    r_lora, _ = inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()
    ck = load_ckpt(model, args.ckpt)
    for p in model.parameters():        # everything frozen: only the head trains
        p.requires_grad_(False)
    print(f'loaded step={ck["step"]} turns={ck.get("turns")} '
          f'| collecting on {args.n_blocks} blocks x d<={args.max_d}', flush=True)

    blocks = build_blocks(tok, 'validation', args.ctx, args.n_blocks)
    f, y, d = collect(model, blocks, args.max_d)
    print(f'features {tuple(f.shape)} labels {tuple(y.shape)} '
          f'pos_rate={float(y.mean()):.4f}', flush=True)
    n_rows = f.shape[0]
    n_blk = len(blocks)
    assert n_rows % n_blk == 0, f'rows {n_rows} not divisible by blocks {n_blk}'
    n_per = n_rows // n_blk
    tr_b, va_b = split_doc(n_blk)
    tr_b = [b_ for b_ in tr_b if b_ < n_blk]
    va_b = [b_ for b_ in va_b if b_ < n_blk]
    if not va_b:                       # guarantee a non-empty val split
        va_b, tr_b = tr_b[-1:], tr_b[:-1]
    tr_idx = torch.cat([torch.arange(b_ * n_per, (b_ + 1) * n_per)
                        for b_ in tr_b]).to(f.device)
    va_idx = torch.cat([torch.arange(b_ * n_per, (b_ + 1) * n_per)
                        for b_ in va_b]).to(f.device)

    f32 = f.float()                 # backbone is bf16; train the head in fp32
    y32 = y.float()
    mu, sd = f32[tr_idx].mean(0), f32[tr_idx].std(0).clamp_min(1e-6)
    fz = (f32 - mu) / sd
    out = {'time': time.strftime('%F %T'), 'ckpt': args.ckpt,
           'ckpt_step': ck['step'], 'turns': ck.get('turns'),
           'n_train': int(len(tr_idx)), 'n_val': int(len(va_idx)),
           'pos_rate': round(float(y32.mean()), 4), 'features':
           ['delta_norm', 'h_norm_after', 'A_bar_mean'], 'label':
           'y = 1[loss(token at d+1) < loss(token at d)]', 'heads': {}}

    for name, head in [('linear', nn.Linear(f.shape[1], 1).cuda()),
                       ('mlp', MLPHead(f.shape[1]).cuda())]:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            opt.zero_grad()
            logit = head(fz[tr_idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y32[tr_idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            sv = torch.sigmoid(head(fz[va_idx]).squeeze(-1))
            a = auc(sv, y32[va_idx])
        out['heads'][name] = {'val_auc': round(a, 4) if a else None,
                              'final_train_loss': round(float(loss), 4)}
        print(f'  head {name}: val AUC={out["heads"][name]["val_auc"]}', flush=True)
        if name == 'mlp':
            torch.save({'head': head.state_dict(), 'mu': mu, 'sd': sd,
                        'ckpt': args.ckpt, 'step': ck['step']},
                       ROOT / 'runs' / 'v5_l1' / 'exit_head.pt')

    p = ROOT / 'results' / 's8_r1_gate.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'written {p}', flush=True)


if __name__ == '__main__':
    main()
