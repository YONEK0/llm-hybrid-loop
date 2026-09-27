"""Train recurrent-depth latent reasoning on Qwen3-4B-Thinking (single 8GB GPU).

Only the per-iteration low-rank deltas are trained; the 4-bit base model stays frozen.
Supervision is answer-only, so the model must compute the result inside its latent
iterations instead of writing a chain of thought.

    python train_recurrent.py --run-name rd1 --max-steps 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, read_jsonl, setup


def build_samples(model, rows):
    samples = []
    for row in rows:
        ids, prefix, target = model.batch(row)
        samples.append({"ids": ids, "prefix": len(prefix), "target": target,
                        "gold": str(row["answer"]).strip(), "id": row["id"],
                        "question": row["question"]})
    return samples


def quick_accuracy(model, samples, depths=(1, 4), limit=16, max_new_tokens=12):
    out = {}
    subset = samples[:limit]
    for depth in depths:
        correct = 0
        total_time = 0.0
        for sample in subset:
            ids = model.prefix(sample["question"])
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
            total_time += time.perf_counter() - start
            from recurrent_qwen import answer_from_text
            answer = answer_from_text(model.tokenizer.decode(generated, skip_special_tokens=True))
            correct += int(answer == sample["gold"])
        out[depth] = {"accuracy": correct / max(len(subset), 1), "n": len(subset),
                      "seconds_per_question": round(total_time / max(len(subset), 1), 2)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="rd1")
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--dev-count", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-depth", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--block-start", type=int, default=24)
    parser.add_argument("--block-len", type=int, default=6)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    setup(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    dev_rows = [rows[i] for i in order[:args.dev_count]]
    train_rows = [rows[i] for i in order[args.dev_count:args.dev_count + args.train_count]]

    model = RecurrentDepthQwen(checkpoint=args.resume, train_depth=args.train_depth,
                               block_start=args.block_start, block_len=args.block_len,
                               rank=args.rank)
    params = model.trainable_parameters()
    print(f"trainable tensors={len(params)} params={sum(p.numel() for p in params)/1e6:.3f}M "
          f"block={args.block_len} rank={args.rank} depth={args.train_depth}", flush=True)

    print("tokenising ...", flush=True)
    train_samples = build_samples(model, train_rows)
    dev_samples = build_samples(model, dev_rows)
    print(f"train={len(train_samples)} dev={len(dev_samples)}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    history = []
    torch.cuda.reset_peak_memory_stats()
    step = 0
    epoch = 0
    t0 = time.perf_counter()
    model.train()
    while step < args.max_steps:
        epoch += 1
        order = list(range(len(train_samples)))
        random.Random(args.seed * 100 + epoch).shuffle(order)
        for index in order:
            if step >= args.max_steps:
                break
            sample = train_samples[index]
            start = time.perf_counter()
            total, report = model.loss_from_sample(sample, aux_weight=args.aux_weight)
            (total / args.accum).backward()
            if (step + 1) % args.accum == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = None
            torch.cuda.synchronize()
            step += 1
            record = {"step": step, "epoch": epoch, "seconds": round(time.perf_counter() - start, 2),
                      "elapsed": round(time.perf_counter() - t0, 1),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), **report}
            if grad_norm is not None:
                record["grad_norm"] = round(grad_norm, 3)
            history.append(record)
            print(json.dumps(record), flush=True)
            if args.eval_every and step % args.eval_every == 0:
                model.eval()
                acc = quick_accuracy(model, dev_samples, depths=(1, args.train_depth),
                                     limit=12, max_new_tokens=args.max_new_tokens)
                model.train()
                entry = {"step": step, "dev": acc}
                history.append(entry)
                print("EVAL " + json.dumps(entry), flush=True)
                (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.max_steps % args.accum != 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save(run_dir / "checkpoint", extra={"args": vars(args), "steps": step,
                                              "train_seconds": round(time.perf_counter() - t0, 1)})
    history.append({"step": step, "train_seconds": round(time.perf_counter() - t0, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done: {step} steps in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
