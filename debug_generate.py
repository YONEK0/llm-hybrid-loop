import json
from pathlib import Path
from latent_qwen import LatentQwen, ROOT, read_jsonl, setup, answer_from_text

setup(0)
model = LatentQwen(latent_steps=4)
rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[:2]
for row in rows:
    print("=" * 70)
    print("Q:", row["question"][:110])
    print("GOLD:", row["answer"])
    for mode in ("cot", "direct", "latent"):
        out = model.generate(row["question"], mode=mode, max_new_tokens=96)
        print(f"--- {mode}: pred={out['answer']} tokens={out['generated_tokens']} "
              f"limit={out['hit_length_limit']} {out['seconds']:.1f}s")
        print("    text:", repr(out["text"][:260]))
