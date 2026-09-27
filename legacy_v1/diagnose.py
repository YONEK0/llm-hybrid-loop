"""Diagnose the gap between low training loss and zero eval accuracy.

Prints the raw generated text next to the gold answer so we can see whether the
model is wrong, or whether the answer extraction / prompt is wrong.
"""

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, read_jsonl, setup

setup(0)
model = RecurrentDepthQwen(checkpoint=ROOT / "runs/rd-v1/checkpoint")
rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[:5]

print("=== training-target format check ===")
row = rows[0]
ids, prefix, target = model.batch(row)
print("prefix tail   :", repr(model.tokenizer.decode(prefix[-25:])))
print("target decode :", repr(model.tokenizer.decode(target)))
print("teacher-force loss check:")
total, report = model.loss_from_ids(ids, len(prefix), target, aux_weight=0.0)
print("  ce =", report)

print("\n=== greedy generation at several depths ===")
for depth in (1, 4):
    print(f"\n--- depth {depth} ---")
    for row in rows[:3]:
        out = model.generate(row["question"], depth=depth, max_new_tokens=32)
        print(f"  gold={row['answer']!r:>6}  pred={out['answer']!r:>8}  "
              f"tokens={out['generated_tokens']} trunc={out['hit_length_limit']}")
        print(f"    text: {out['text']!r}")
