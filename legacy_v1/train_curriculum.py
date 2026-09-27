"""Curriculum training: keep a few visible CoT steps, carry the rest in latent iterations.

Motivation: asking a 4B model to answer GSM8K with *no* visible reasoning is the hard
setting that plain answer-only training failed on.  This script instead keeps the first
`keep_steps` rationale steps in the output and lets the recurrent loop account for the
remaining ones, so the difficulty is tunable.

    python train_curriculum.py --run-name cur-k1 --keep-steps 1 --max-steps 300

The keep_steps value can be annealed over the run via --keep-steps-schedule, which is
the actual curriculum: start with more visible reasoning, then withdraw it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup


def parse_schedule(text):
    """'0:2,300:1,600:0' -> [(0,2),(300,1),(600,0)]"""
    if not text:
        return None
    out = []
    for piece in text.split(","):
        step, steps = piece.split(":")
        out.append((int(step), int(steps)))
    return sorted(out)


def keep_steps_at(schedule, step, default):
    if not schedule:
        return default
    value = schedule[0][1]
    for start, count in schedule:
        if step >= start:
            value = count
    return value


@torch.no_grad()
def greedy(model, question, depth, max_new_tokens):
    ids = model.prefix(question)
    generated = []
    for _ in range(max_new_tokens):
        logits, _ = model.forward(torch.tensor([ids], dtype=torch.long, device="cuda"), depth=depth)
        token = int(logits[0, -1].argmax(-1).item())
        generated.append(token)
        ids.append(token)
        if token == model.tokenizer.eos_token_id:
            break
    text = model.tokenizer.decode(generated, skip_special_tokens=True)
    return answer_from_text(text)


def dev_accuracy(model, rows, depths, max_new_tokens):
    out = {}
    for depth in depths:
        correct = sum(int(greedy(model, row["question"], depth, max_new_tokens)
                          == str(row["answer"]).strip()) for row in rows)
        out[depth] = correct / max(len(rows), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="cur-k1")
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--dev-count", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-depth", type=int, default=4)
    parser.add_argument("--keep-steps", type=int, default=1,
                        help="visible CoT steps kept in the target; -1 keeps all")
    parser.add_argument("--keep-steps-schedule", default=None,
                        help="e.g. '0:2,300:1,600:0' to anneal the scaffold away")
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--block-start", type=int, default=24)
    parser.add_argument("--block-len", type=int, default=6)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--depth-sampling", action="store_true",
                        help="draw the loop count from 1..train_depth each step so one "
                             "checkpoint works at any inference depth")
    args = parser.parse_args()

    setup(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    schedule = parse_schedule(args.keep_steps_schedule)

    rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    dev_rows = [rows[i] for i in order[:args.dev_count]]
    train_rows = [rows[i] for i in order[args.dev_count:args.dev_count + args.train_count]]

    model = RecurrentDepthQwen(checkpoint=args.resume, train_depth=args.train_depth,
                               block_start=args.block_start, block_len=args.block_len,
                               rank=args.rank)
    params = model.trainable_parameters()
    print(f"trainable={sum(p.numel() for p in params)/1e6:.3f}M depth={args.train_depth} "
          f"keep_steps={args.keep_steps} schedule={schedule}", flush=True)
    print(f"train={len(train_rows)} dev={len(dev_rows)}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    history = []
    torch.cuda.reset_peak_memory_stats()
    step = 0
    epoch = 0
    t0 = time.perf_counter()
    model.train()
    while step < args.max_steps:
        epoch += 1
        order = list(range(len(train_rows)))
        random.Random(args.seed * 100 + epoch).shuffle(order)
        for index in order:
            if step >= args.max_steps:
                break
            keep = keep_steps_at(schedule, step, args.keep_steps)
            keep = None if keep < 0 else keep
            row = train_rows[index]
            start = time.perf_counter()
            total, report = model.scaffold_loss(row, keep_steps=keep,
                                                aux_weight=args.aux_weight, depth=args.train_depth,
                                                depth_sampling=args.depth_sampling)
            (total / args.accum).backward()
            grad_norm = None
            if (step + 1) % args.accum == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            record = {"step": step, "epoch": epoch, "seconds": round(time.perf_counter() - start, 2),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), **report}
            if grad_norm is not None:
                record["grad_norm"] = round(grad_norm, 3)
            history.append(record)
            print(json.dumps(record), flush=True)
            if args.eval_every and step % args.eval_every == 0:
                model.eval()
                acc = dev_accuracy(model, dev_rows, (args.train_depth,), args.max_new_tokens)
                model.train()
                entry = {"step": step, "keep_steps": report["keep_steps"], "dev": acc}
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
