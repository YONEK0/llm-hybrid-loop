"""q_head halting-gate AUC diagnostic on held-out WikiText blocks (V5 L0 ckpt)."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import json
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from s1b_trace_v2 import load_hybrid_nf4
from s2_spike import LoopUSQwen35


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    g = n = 0
    for sp in pos:
        for sn in neg:
            g += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
            n += 1
    return round(g / n, 4)


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
          for i in range(8)]
blocks = [b for b in blocks if b.shape[1] == 512]
print(f'blocks: {len(blocks)}, ckpt step {ck["step"]}', flush=True)

recs = []
with torch.no_grad():
    for bi, x in enumerate(blocks):
        h = model.encode(x)
        seq = []
        for d in range(1, 9):
            h_prop = model.block(h)
            delta_pre = h_prop - h
            q = float(model.q_head(delta_pre[:, -1, :].float()).squeeze())
            h = model.gate(h_prop, h)
            logits = model.decode(h)
            pred = logits[:, :-1].argmax(-1)
            tgt = x[:, 1:]
            seq.append({'d': d, 'q': round(q, 4),
                        'acc': float((pred == tgt).float().mean()),
                        'last_ok': bool(pred[0, -1] == tgt[0, -1])})
        recs.append(seq)
        print(f'block {bi}: ' + ' '.join(
            f"d{r['d']}:q={r['q']:.2f},acc={r['acc']:.3f}", flush=True)
            for r in seq) if False else print(
            f"block {bi}: " + ' '.join(
                f"d{r['d']}:q={r['q']:.2f},acc={r['acc']:.3f}"
                for r in seq), flush=True)

qs, oks, fix_pairs = [], [], []
for seq in recs:
    for i, r in enumerate(seq):
        qs.append(r['q'])
        oks.append(r['last_ok'])
        if i + 1 < len(seq) and not r['last_ok']:
            fix_pairs.append((r['q'], seq[i + 1]['last_ok']))

res = {
    'ckpt_step': ck['step'],
    'auc_q_vs_lasttoken_correct': auc(qs, oks),
    'auc_q_vs_fix_next_depth': auc([q for q, _ in fix_pairs],
                                   [f for _, f in fix_pairs]),
    'n_pairs': {'correct': len(qs), 'fix': len(fix_pairs)},
    'mean_q_by_depth': {f'd{d}': round(sum(s[d - 1]['q'] for s in recs) / len(recs), 4)
                        for d in range(1, 9)},
    'mean_acc_by_depth': {f'd{d}': round(sum(s[d - 1]['acc'] for s in recs) / len(recs), 4)
                          for d in range(1, 9)},
    'records': recs,
}
print(json.dumps({k: v for k, v in res.items() if k != 'records'},
                 indent=1), flush=True)
open('results/q_head_auc_v5_l0.json', 'w', encoding='utf-8').write(
    json.dumps(res, ensure_ascii=False, indent=1))
print('written results/q_head_auc_v5_l0.json', flush=True)
