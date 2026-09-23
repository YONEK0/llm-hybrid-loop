"""Fast ProsQA evaluation: 30 questions, 3 depths, 16 tokens each. ~5 minutes."""

import json
import random
import time

import torch

from prosqa_adapter import ProsQAModel, load_prosqa, prosqa_accuracy, extract_prosqa_answer
from recurrent_qwen import ROOT, setup

setup(11)
model = ProsQAModel(checkpoint=ROOT / "runs/pq-v1/checkpoint")
model.eval()

rows = load_prosqa("test", 30, seed=11)
depths = [1, 4, 8]
MAX_TOKENS = 16
results = {}

# base model reference (frozen, no deltas, no loop)
base = model.model  # Qwen3ForCausalLM
tok = model.tokenizer
system = "Determine the logical relationship by reasoning step by step."
base_correct = 0
base_tokens = 0
print("evaluating base model ...", flush=True)
for row in rows:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if not isinstance(enc, list):
        enc = list(enc["input_ids"])
    ids = [int(t) for t in enc]
    with torch.no_grad():
        for _ in range(48):
            out = base(input_ids=torch.tensor([ids], device="cuda"))
            t = int(out.logits[0, -1].argmax(-1).item())
            ids.append(t)
            if t == tok.eos_token_id:
                break
    text = tok.decode(ids[len(enc):], skip_special_tokens=True)
    base_correct += int(prosqa_accuracy(extract_prosqa_answer(text), row["answer"]))
    base_tokens += len(ids) - len(enc)
results["base"] = {"accuracy": base_correct / len(rows), "correct": base_correct,
                   "mean_tokens": round(base_tokens / len(rows), 1)}
print(f"  base: acc={base_correct/len(rows):.3f} tokens={base_tokens/len(rows):.1f}", flush=True)

# trained model at several depths
for depth in depths:
    correct = 0
    tokens = 0
    samples = []
    print(f"evaluating depth {depth} ...", flush=True)
    for row in rows:
        ids = model.prefix(row["question"])
        with torch.no_grad():
            generated = []
            for _ in range(MAX_TOKENS):
                logits, _ = model.forward(torch.tensor([ids], dtype=torch.long, device="cuda"), depth=depth)
                t = int(logits[0, -1].argmax(-1).item())
                generated.append(t)
                ids.append(t)
                if t == model.tokenizer.eos_token_id:
                    break
        text = model.tokenizer.decode(generated, skip_special_tokens=True)
        pred = extract_prosqa_answer("#### " + text if "####" in text else text)
        ok = prosqa_accuracy(pred, row["answer"])
        correct += int(ok)
        tokens += len(generated)
        if len(samples) < 2:
            samples.append({"gold": row["answer"], "pred": pred, "text": text[:200]})
    results[f"depth_{depth}"] = {"accuracy": correct / len(rows), "correct": correct,
                                 "mean_tokens": round(tokens / len(rows), 1), "samples": samples}
    print(f"  depth {depth}: acc={correct/len(rows):.3f} tokens={tokens/len(rows):.1f}", flush=True)

out = ROOT / "results" / "pq_v1_fast_eval.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n===== FINAL DEPTH CURVE =====")
print(f"  base model (no loop):     {results['base']['accuracy']:.3f}")
for depth in depths:
    print(f"  trained depth {depth:>2}:       {results[f'depth_{depth}']['accuracy']:.3f}")
print(f"\nwritten to {out}")
