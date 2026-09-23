"""Correctness gate: with zero-initialised deltas, depth=1 must reproduce the base model.

If this fails, the looped forward pass is not a faithful conversion and any training
result would be meaningless.
"""

import json
from pathlib import Path

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, read_jsonl, setup

setup(0)
result = {}
model = RecurrentDepthQwen(train_depth=4, block_start=24, block_len=6, rank=8)
row = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[0]
ids, prefix, target = model.batch(row)
input_ids = torch.tensor([ids], device="cuda")

# looped forward at depth 1 (deltas are zero -> should be identity)
with torch.no_grad():
    loop_logits, _ = model.forward(input_ids, depth=1)
    base_logits = model.model(input_ids=input_ids).logits

diff = (loop_logits.float() - base_logits.float()).abs()
result["max_abs_diff"] = float(diff.max())
result["mean_abs_diff"] = float(diff.mean())
result["equivalent"] = bool(diff.max() < 1e-3)
result["note"] = "depth=1 looped forward vs stock forward; deltas are zero-initialised so they must match"

print(json.dumps(result, indent=2), flush=True)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "equivalence_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
raise SystemExit(0 if result["equivalent"] else 1)
