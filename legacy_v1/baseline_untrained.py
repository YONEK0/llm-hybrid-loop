"""Untrained baseline: what the depth curve looks like before any delta training.

With zero-initialised deltas the looped model at depth K is just the base model's
block applied K times, so this measures the starting point the training must beat.
"""

import json

from recurrent_qwen import RecurrentDepthQwen, ROOT, read_jsonl, setup

setup(0)
model = RecurrentDepthQwen(train_depth=4, block_start=24, block_len=6, rank=8)
rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[:12]
result = {"note": "untrained, zero-initialised deltas", "metric": "greedy answer accuracy",
          "depth_curve": {}}
for depth in (1, 2, 4):
    correct = 0
    tokens = 0
    seconds = 0.0
    for row in rows:
        out = model.generate(row["question"], depth=depth, max_new_tokens=16)
        correct += int(out["answer"] == str(row["answer"]).strip())
        tokens += out["generated_tokens"]
        seconds += out["seconds"]
    result["depth_curve"][depth] = {"accuracy": correct / len(rows), "correct": correct,
                                    "n": len(rows), "mean_tokens": round(tokens / len(rows), 1),
                                    "mean_seconds": round(seconds / len(rows), 2)}
print(json.dumps(result, indent=2), flush=True)
(ROOT / "reports" / "baseline_untrained.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
