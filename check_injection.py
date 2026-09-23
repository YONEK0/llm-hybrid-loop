"""Check that the loop re-injection projection actually trains now.

Previously trainable_parameters() omitted it, so the injection matrix stayed at its
exact zero initialisation and the re-injection mechanism was inert for a whole run.
"""

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, read_jsonl, setup

setup(0)
model = RecurrentDepthQwen(train_depth=4, block_start=24, block_len=6, rank=8)
rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[:8]

params = model.trainable_parameters()
hidden_size = model.model.config.hidden_size
injection_params = list(model.injection.parameters())
n_injection = sum(p.numel() for p in injection_params)
print(f"trainable tensors={len(params)}  injection params={n_injection} "
      f"({n_injection/1e6:.3f}M)  included={all(any(p is q for q in params) for p in injection_params)}")

before = model.injection.proj.weight[:, hidden_size:].detach().clone()
print(f"injection 'initial-state' half before training: max|w| = {float(before.abs().max()):.6f}")

optimizer = torch.optim.AdamW(params, lr=1e-4)
model.train()
for step, row in enumerate(rows):
    total, report = model.scaffold_loss(row, keep_steps=2, aux_weight=0.2, depth=4)
    total.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if step == len(rows) - 1:
        print(f"  step {step+1}: ce={report['ce']:.3f}")

after = model.injection.proj.weight[:, hidden_size:].detach()
delta = float((after - before).abs().max())
current = float(after.abs().max())
print(f"injection 'initial-state' half after  {len(rows)} steps: max|w| = {current:.6f}  "
      f"max|change| = {delta:.6f}")
print("VERDICT:", "injection is learning" if delta > 1e-6 else "STILL FROZEN - fix ineffective")
