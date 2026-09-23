"""Train recurrent-depth latent reasoning on ProsQA (synthetic DAG logic).

Why this task: the answer is a specific invented entity ("Tom is a zhorpus."), so a model
cannot score by guessing or by pattern-matching a number.  On GSM8K the base model already
answered many problems once given a couple of gold steps, which made the measurement about
the scaffold rather than the latent loop.  ProsQA is also where Coconut reports its
favourable result, and the reasoning depth (3-6 steps) is known and bounded — which is what
a curriculum needs.

    python train_prosqa.py --run-name pq-v1 --keep-steps 2 --max-steps 600
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from prosqa_adapter import (ProsQAModel, extract_prosqa_answer, load_prosqa,
                            prosqa_accuracy, prosqa_row_to_gsm8k_shape)
from recurrent_qwen import ROOT, setup


def keep_steps_at(schedule, step, default):
    if not schedule:
        return default
    value = schedule[0][1]
    for start, count in schedule:
        if step >= start:
            value = count
    return value


def parse_schedule(text):
    if not text:
        return None
    out = []
    for piece in text.split(","):
        step, steps = piece.split(":")
        out.append((int(step), int(steps)))
    return sorted(out)


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
    return extract_prosqa_answer(text)


@torch.no_grad()
def score_with_gold(model, row, keep_steps, depth):
    """Feed the gold first-k steps, score the answer completion (training-consistent)."""
    import torch.nn.functional as F
    prompt = model.prefix(row["question"])
    scaffold = model.scaffold_tokens(row, keep_steps)
    answer = model.target(row)
    absolute_start = len(prompt) if scaffold else len(prompt) - 1
    span_len = (len(scaffold) - 1 if scaffold else 0) + len(answer)
    labels = torch.tensor([scaffold[1:] + answer], dtype=torch.long, device="cuda").reshape(-1)
    logits, _ = model.forward_split(prompt, scaffold, answer[:-1], depth=depth,
                                    mark_offset=absolute_start, mark_length=span_len)
    lg = logits[:, absolute_start:absolute_start + span_len].reshape(-1, logits.shape[-1])
    ce = float(F.cross_entropy(lg, labels))
    text = model.tokenizer.decode(lg[-len(answer):].argmax(-1).tolist(), skip_special_tokens=True)
    return ce, prosqa_accuracy(extract_prosqa_answer("####" + text), row["answer"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="pq-v1")
    parser.add_argument("--train-count", type=int, default=6000)
    parser.add_argument("--dev-count", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-depth", type=int, default=4)
    parser.add_argument("--keep-steps", type=int, default=2)
    parser.add_argument("--keep-steps-schedule", default=None)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--block-start", type=int, default=24)
    parser.add_argument("--block-len", type=int, default=6)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--depth-sampling", action="store_true")
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    setup(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    schedule = parse_schedule(args.keep_steps_schedule)

    train_rows = [prosqa_row_to_gsm8k_shape(r) for r in load_prosqa("train", args.train_count, args.seed)]
    # keep raw 'steps' so cot_steps can use the list form
    raw = load_prosqa("train", args.train_count, args.seed)
    for adapted, original in zip(train_rows, raw):
        adapted["steps"] = original["steps"]
    dev_rows = [prosqa_row_to_gsm8k_shape(r) for r in load_prosqa("valid", args.dev_count, args.seed)]
    raw_dev = load_prosqa("valid", args.dev_count, args.seed)
    for adapted, original in zip(dev_rows, raw_dev):
        adapted["steps"] = original["steps"]

    model = ProsQAModel(checkpoint=args.resume, train_depth=args.train_depth,
                        block_start=args.block_start, block_len=args.block_len, rank=args.rank)
    params = model.trainable_parameters()
    print(f"trainable={sum(p.numel() for p in params)/1e6:.3f}M depth={args.train_depth} "
          f"train={len(train_rows)} dev={len(dev_rows)} schedule={schedule}", flush=True)

    # baseline sanity: how well does the base model do with gold steps, no training?
    print("pre-training reference (gold steps, depth 4):", flush=True)
    for keep in (len(dev_rows[0].get("steps", [])) or 3, 0):
        ces, hits = [], 0
        for row in dev_rows[:12]:
            ce, ok = score_with_gold(model, row, keep, 4)
            ces.append(ce)
            hits += int(ok)
        print(f"  keep={keep}: CE={sum(ces)/len(ces):.3f} EM={hits/12:.3f}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    history = []
    torch.cuda.reset_peak_memory_stats()
    step, epoch = 0, 0
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
            total, report = model.scaffold_loss(row, keep_steps=keep, aux_weight=args.aux_weight,
                                                depth=args.train_depth,
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
            if step % 50 == 0:
                print(json.dumps(record), flush=True)
            if args.eval_every and step % args.eval_every == 0:
                model.eval()
                entry = {"step": step, "keep_steps": report["keep_steps"], "scored": {}}
                for k in (report["keep_steps"] if report["keep_steps"] is not None else 3, 0):
                    ces, hits = [], 0
                    for row in dev_rows[:20]:
                        ce, ok = score_with_gold(model, row, k, args.train_depth)
                        ces.append(ce)
                        hits += int(ok)
                    entry["scored"][f"k{k}"] = {"ce": round(sum(ces)/len(ces), 3), "em": hits/20}
                free_hits = sum(int(greedy(model, r["question"], args.train_depth, args.max_new_tokens)
                                    == r["answer"]) for r in dev_rows[:10])
                entry["free_acc"] = free_hits / 10
                model.train()
                history.append(entry)
                print("EVAL " + json.dumps(entry), flush=True)
                (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                # periodic checkpoint so the run can be stopped at any eval node
                model.save(run_dir / "checkpoint",
                           extra={"args": vars(args), "steps": step,
                                  "keep_steps_at_save": report["keep_steps"]})
                print(f"CHECKPOINT saved at step {step}", flush=True)

    if args.max_steps % args.accum != 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save(run_dir / "checkpoint", extra={"args": vars(args), "steps": step,
                                              "train_seconds": round(time.perf_counter() - t0, 1)})
    history.append({"step": step, "train_seconds": round(time.perf_counter() - t0, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done: {step} steps in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
