"""Compare the trained recurrent-depth model against its own base model and ablations.

Four measurements, all on the held-out test split:

  base          - the untouched 4-bit Qwen3-4B-Thinking (no deltas, no loop)
  trained@K     - the looped model at inference depth K (K = 1,2,4,8,16,32)
  zeroed@K      - deltas zeroed at inference: isolates how much of the answer comes
                  from the base weights versus the trained loop
  repeat@K      - the base model's block re-applied with no trained delta (control for
                  "just looping the same layer helps")

The script prints a table and writes JSON so the claims can be audited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup


def load_rows(count, seed, split="test"):
    rows = read_jsonl(ROOT / f"data/gsm8k/{split}.jsonl")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order[:count]]


@torch.no_grad()
def run_looped(model, question, depth, max_new_tokens):
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
    return {"answer": answer_from_text(text), "text": text, "generated_tokens": len(generated),
            "seconds": time.perf_counter() - start,
            "hit_length_limit": bool(generated) and generated[-1] != model.tokenizer.eos_token_id}


@torch.no_grad()
def run_base(model, question, max_new_tokens):
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
    return {"answer": answer_from_text(text), "text": text, "generated_tokens": len(generated),
            "seconds": time.perf_counter() - start,
            "hit_length_limit": bool(generated) and generated[-1] != model.tokenizer.eos_token_id}


def zero_out_deltas(model):
    saved = {k: v.detach().clone() for k, v in model.deltas.state_dict().items()}
    with torch.no_grad():
        for wrapper in model.deltas.values():
            for parameter in list(wrapper.B):
                parameter.zero_()
    return saved


def restore_deltas(model, saved):
    model.deltas.load_state_dict(saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/rd-v1/checkpoint")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="rd-v1")
    args = parser.parse_args()

    setup(args.seed)
    checkpoint = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    model = RecurrentDepthQwen(checkpoint=checkpoint)
    rows = load_rows(args.count, args.seed)
    print(f"n={len(rows)} train_depth={model.train_depth} depths={args.depths}\n", flush=True)

    results = {"checkpoint": str(checkpoint), "n": len(rows), "split": "test",
               "train_depth": model.train_depth, "measurements": {}}
    per_question = []

    # 1) untouched base model
    base_hits = 0
    base_time = 0.0
    for row in rows:
        out = run_base(model, row["question"], args.max_new_tokens)
        base_hits += int(out["answer"] == str(row["answer"]).strip())
        base_time += out["seconds"]
    results["measurements"]["base"] = {"accuracy": base_hits / len(rows), "correct": base_hits,
                                       "n": len(rows),
                                       "mean_seconds": round(base_time / len(rows), 2)}
    print(f"base model          acc={base_hits/len(rows):.3f} ({base_hits}/{len(rows)})", flush=True)

    # 2) trained loop at several depths
    for depth in args.depths:
        hits, seconds, tokens, truncated = 0, 0.0, 0, 0
        records = []
        for row in rows:
            out = run_looped(model, row["question"], depth, args.max_new_tokens)
            ok = out["answer"] == str(row["answer"]).strip()
            hits += int(ok)
            seconds += out["seconds"]
            tokens += out["generated_tokens"]
            truncated += int(out["hit_length_limit"])
            records.append({"id": row["id"], "gold": str(row["answer"]).strip(),
                            "pred": out["answer"], "ok": ok, "text": out["text"][:200]})
        entry = {"accuracy": hits / len(rows), "correct": hits, "n": len(rows),
                 "mean_seconds": round(seconds / len(rows), 2),
                 "mean_tokens": round(tokens / len(rows), 1), "truncated": truncated}
        results["measurements"][f"trained_d{depth}"] = entry
        per_question.append({"depth": depth, "records": records})
        print(f"trained depth {depth:>2}    acc={hits/len(rows):.3f} ({hits}/{len(rows)}) "
              f"tokens={entry['mean_tokens']:.1f} {entry['mean_seconds']:.2f}s/q", flush=True)

    # 3) zeroed deltas: is the loop carrying the answer, or the base weights?
    saved = zero_out_deltas(model)
    for depth in [d for d in args.depths if d <= 4]:
        hits = 0
        for row in rows[:40]:
            out = run_looped(model, row["question"], depth, args.max_new_tokens)
            hits += int(out["answer"] == str(row["answer"]).strip())
        entry = {"accuracy": hits / 40, "correct": hits, "n": 40}
        results["measurements"][f"zeroed_d{depth}"] = entry
        print(f"deltas=0 depth {depth:>2}   acc={hits/40:.3f} ({hits}/40)", flush=True)
    restore_deltas(model, saved)

    results["per_question"] = per_question
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}_final_test.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
