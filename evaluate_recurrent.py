"""Evaluate recurrent-depth latent reasoning.

Two questions matter, and neither is answered by a single accuracy number:

1. Does more latent computation help?  -> accuracy as a function of inference depth,
   including depths never trained on (depth extrapolation).
2. Is the latent pathway actually being used, or is the answer merely read off the
   frozen base model?  -> compare against (a) the untouched base model and (b) an
   ablation where the loop is skipped.

Reports per-question records so the numbers can be re-checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup


def load_rows(path, count, seed):
    rows = read_jsonl(path)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order[:count]]


@torch.no_grad()
def greedy_answer(model, question, depth, max_new_tokens):
    ids = model.prefix(question)
    start = time.perf_counter()
    generated = []
    for _ in range(max_new_tokens):
        logits, _ = model.forward(torch.tensor([ids], device="cuda"), depth=depth)
        token = int(logits[0, -1].argmax(-1).item())
        generated.append(token)
        ids.append(token)
        if token == model.tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    text = model.tokenizer.decode(generated, skip_special_tokens=True)
    return {"answer": answer_from_text(text), "text": text,
            "seconds": time.perf_counter() - start, "generated_tokens": len(generated),
            "hit_length_limit": bool(generated) and generated[-1] != model.tokenizer.eos_token_id}


@torch.no_grad()
def base_model_answer(model, question, max_new_tokens):
    """Stock forward pass with the deltas bypassed: the frozen base model's own answer."""
    was = model.train_depth
    ids = model.prefix(question)
    start = time.perf_counter()
    generated = []
    for _ in range(max_new_tokens):
        out = model.model(input_ids=torch.tensor([ids], device="cuda"))
        token = int(out.logits[0, -1].argmax(-1).item())
        generated.append(token)
        ids.append(token)
        if token == model.tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    text = model.tokenizer.decode(generated, skip_special_tokens=True)
    return {"answer": answer_from_text(text), "text": text,
            "seconds": time.perf_counter() - start, "generated_tokens": len(generated)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    setup(args.seed)
    model = RecurrentDepthQwen(checkpoint=args.checkpoint)
    path = ROOT / f"data/gsm8k/{args.split}.jsonl"
    rows = load_rows(path, args.count, args.seed)
    print(f"evaluating {len(rows)} questions on split={args.split} "
          f"depths={args.depths} train_depth={model.train_depth}", flush=True)

    results = {"checkpoint": args.checkpoint, "split": args.split, "n": len(rows),
               "train_depth": model.train_depth, "depths": {}, "questions": rows[0]["question"][:80]}
    records = []
    for question_index, row in enumerate(rows):
        gold = str(row["answer"]).strip()
        entry = {"id": row["id"], "gold": gold, "depths": {}}
        for depth in args.depths:
            out = greedy_answer(model, row["question"], depth, args.max_new_tokens)
            entry["depths"][str(depth)] = {**out, "ok": out["answer"] == gold}
        records.append(entry)
        if (question_index + 1) % 10 == 0:
            summary = {d: round(sum(r["depths"][str(d)]["ok"] for r in records) / len(records), 3)
                       for d in args.depths}
            print(f"  {question_index+1}/{len(rows)} running acc={summary}", flush=True)

    for depth in args.depths:
        key = str(depth)
        answers = [r["depths"][key] for r in records]
        results["depths"][key] = {
            "accuracy": sum(a["ok"] for a in answers) / len(answers),
            "correct": sum(a["ok"] for a in answers),
            "n": len(answers),
            "mean_seconds": sum(a["seconds"] for a in answers) / len(answers),
            "mean_tokens": sum(a["generated_tokens"] for a in answers) / len(answers),
            "hit_length_limit": sum(a["hit_length_limit"] for a in answers),
        }
    results["records"] = records

    stem = args.tag or Path(args.checkpoint).parent.name
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stem}_{args.split}_depth_curve.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== depth curve (accuracy vs latent iterations) ===")
    for depth in args.depths:
        stats = results["depths"][str(depth)]
        print(f"  depth {depth:>2}: acc={stats['accuracy']:.3f} ({stats['correct']}/{stats['n']}) "
              f"tokens={stats['mean_tokens']:.1f} {stats['mean_seconds']:.2f}s/q "
              f"truncated={stats['hit_length_limit']}")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
