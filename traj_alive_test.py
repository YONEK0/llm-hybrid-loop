"""traj_alive_test.py — dead-trajectory discriminator for V5 L0 ckpt."""
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

vel, ces, kl, flip = {}, {}, {}, {}
with torch.no_grad():
    for bi, x in enumerate(blocks):
        h = model.encode(x)
        probs_prev = None
        for d in range(1, 17):
            h_new = model.gate(model.block(h), h)
            v = float((h_new - h).norm(dim=-1).mean()
                      / h.norm(dim=-1).mean())
            vel.setdefault(d, []).append(v)
            h = h_new
            logits = model.decode(h)
            ce = float(F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                x[:, 1:].reshape(-1)))
            ces.setdefault(d, []).append(ce)
            probs = logits[:, -50:, :].float().softmax(-1)
            if probs_prev is not None:
                p0 = probs_prev.clamp_min(1e-12)
                p1 = probs.clamp_min(1e-12)
                kl.setdefault(d, []).append(
                    float((p0 * (p0.log() - p1.log())).sum(-1).mean()))
                flip.setdefault(d, []).append(
                    float((probs_prev.argmax(-1)
                           != probs.argmax(-1)).float().mean()))
            probs_prev = probs
        print(f'block {bi} done', flush=True)

m = lambda dct, d: round(sum(dct[d]) / len(dct[d]), 5)
out = {
    'ckpt_step': ck['step'],
    'velocity_rel_per_loop': {f'd{d}': m(vel, d) for d in sorted(vel)},
    'ce_by_depth': {f'd{d}': m(ces, d) for d in sorted(ces)},
    'kl_d_minus_1_to_d': {f'd{d}': m(kl, d) for d in sorted(kl)},
    'top1_flip_rate': {f'd{d}': m(flip, d) for d in sorted(flip)},
}
print(json.dumps(out, indent=1), flush=True)
open('results/traj_alive_v5_l0.json', 'w', encoding='utf-8').write(
    json.dumps(out, indent=1))
print('written results/traj_alive_v5_l0.json', flush=True)
