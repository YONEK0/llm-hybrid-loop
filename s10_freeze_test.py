"""s10_freeze_test.py — Is the CE cost from the halting MASK or the gate MLP?

Three-way isolation on the same v3 base + v4 router:
 A. no mask (run all B loops, gate ignored)         -> baseline
 B. mask but keep state (halted token keeps updating, mask only records)
 C. mask + freeze (v4 as trained)                   -> current behaviour
If A~B << C then the freeze itself is the damage (not the routing policy).
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
from s7_train_l1 import L1Qwen35, inject_lora, build_wiki_blocks
from s8_r0_l1 import load_ckpt as load_v3
from s10_train_v4 import V4Router

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def run(model, router, x, B, mode, tau=1e-4):
    h = model.encode(x)
    h0 = h
    mask = torch.ones(h.shape[:2], device=h.device)
    exits = torch.full(h.shape[:2], float(B), device=h.device)
    for t in range(1, B + 1):
        inp = h if t == 1 else h + h0
        h_prop = model.block(inp)
        h_new = model.l1gate(h_prop, h, t)
        s = router(h_new, t)
        keep = torch.where(s >= tau, torch.ones_like(s),
                           torch.zeros_like(s)) * mask
        first_halt = (mask > 0) & (keep == 0)
        exits[first_halt] = float(t)
        if mode == 'A':
            h = h_new
        elif mode == 'B':
            h = h_new                              # state still updates
            mask = (keep * mask)                   # mask only records
        else:
            h = torch.where(keep.unsqueeze(-1) > 0, h_new, h)
            mask = keep
    logits = model.decode(h)
    ce = float(F.cross_entropy(logits[:, :-1].float().reshape(-1,
               logits.shape[-1]), x[:, 1:].reshape(-1)))
    return ce, exits


def main():
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()
    load_v3(model, 'runs/v5_l1v3/ckpt_v3_step225_diag.pt')
    hidden = int(model.norm.weight.shape[0])
    router = V4Router(hidden, 8, 256).cuda()
    ck = torch.load('runs/v5_l1v4b/ckpt_latest.pt', map_location='cuda',
                    weights_only=False)
    router.load_state_dict(ck['router'])
    router.eval()
    dev_all, n = build_wiki_blocks(tok, 'validation', 512)
    blocks = [dev_all[i].unsqueeze(0) for i in range(4)]
    out = {}
    for mode in ('A', 'B', 'C'):
        ces, exs = [], []
        for x in blocks:
            c, e = run(model, router, x, 8, mode)
            ces.append(c)
            exs.append(e.cpu())
        ex = torch.cat(exs)
        out[mode] = {'ce': round(sum(ces) / len(ces), 4),
                     'mean_exit': round(float(ex.mean()), 2),
                     'n_distinct': int(len(torch.unique(ex)))}
        print(f'  mode {mode}: CE={out[mode]["ce"]} '
              f'mean_exit={out[mode]["mean_exit"]} '
              f'distinct={out[mode]["n_distinct"]}', flush=True)
    print()
    print(f'冻结损伤 = C - B = {out["C"]["ce"] - out["B"]["ce"]:+.4f}')
    print(f'策略损伤 = B - A = {out["B"]["ce"] - out["A"]["ce"]:+.4f}')
    p = ROOT / 'results' / 's10_freeze_test.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'written {p}')


if __name__ == '__main__':
    main()
