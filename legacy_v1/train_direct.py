"""Scheduled-sampling fine-tune: teach the looped model to EMIT the answer.

Diagnosis this run addresses (2026-09-19): pq-v1 reaches 25% scored EM at keep=0
(teacher-forced), but its free generation only restates the question ("Okay, let's
try to figure out...") — training never saw the model's own prefixes, and pq-v1's
inline free_acc was (mis)measured at depth 4 only.  Classic exposure bias.

Method (scheduled sampling, ProsQA):
  1. greedy rollout from the current model at depth d (no grad), up to L tokens
  2. truncate the rollout at a random position t (sometimes drop it entirely)
  3. one grad step: CE of the answer tokens given prompt + rollout[:t]

so the model learns "after whatever I just said, emit </think>\\n\\n#### <answer>".
Strict free-generation EM (eval_ouro_depth.extract_answer_strict semantics) at
depth 1 is the acceptance metric — the base model scores 0/30 there.

    python train_direct.py --run-name pd-v1 --max-steps 600
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch
import torch.nn.functional as F

from prosqa_adapter import ProsQAModel, load_prosqa, prosqa_accuracy, prosqa_row_to_gsm8k_shape
from recurrent_qwen import ROOT, setup


def extract_strict(text):
    if "####" in text:
        m = re.search(r"([A-Z][A-Za-z]* is a [a-z]+)", text.rsplit("####", 1)[-1])
        return m.group(1) if m else None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        m = re.fullmatch(r"([A-Z][A-Za-z]* is (?:not\s+)?a [a-z]+)\.?", lines[-1])
        if m:
            return m.group(1)
    return None


@torch.no_grad()
def rollout(model, prompt_ids, depth, max_tokens=20):
    cur = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    generated = []
    for _ in range(max_tokens):
        logits, _ = model.forward(cur, depth=depth)
        token = int(logits[0, -1].argmax(-1).item())
        generated.append(token)
        cur = torch.cat([cur, torch.tensor([[token]], device="cuda")], dim=1)
        if token == model.tokenizer.eos_token_id:
            break
    return generated


def answer_loss(model, prompt, prefix, target, depth):
    """CE of `target` tokens given prompt + visible prefix (teacher-forced)."""
    body = list(prompt) + list(prefix)
    logits, _ = model.forward_split(body, [], target[:-1], depth=depth)
    start = len(body) - 1
    lg = logits[:, start:start + len(target)].reshape(-1, logits.shape[-1])
    labels = torch.tensor([target], dtype=torch.long, device="cuda").reshape(-1)
    return F.cross_entropy(lg, labels)


@torch.no_grad()
def strict_free_em(model, rows, depth=1, budget=16):
    hits = 0
    for row in rows:
        ids = model.prefix(row["question"])
        cur = torch.tensor([ids], dtype=torch.long, device="cuda")
        gen = []
        for _ in range(budget):
            logits, _ = model.forward(cur, depth=depth)
            t = int(logits[0, -1].argmax(-1).item())
            gen.append(t)
            cur = torch.cat([cur, torch.tensor([[t]], device="cuda")], dim=1)
            if t == model.tokenizer.eos_token_id:
                break
        text = model.tokenizer.decode(gen, skip_special_tokens=True)
        hits += int(prosqa_accuracy(extract_strict(text), row["answer"]))
    return hits / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="pd-v1")
    ap.add_argument("--resume", default="runs/pq-v1/checkpoint")
    ap.add_argument("--train-count", type=int, default=6000)
    ap.add_argument("--dev-count", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train-depth", type=int, default=4,
                    help="must match the resumed checkpoint's iteration-delta count")
    ap.add_argument("--depth-sampling", action="store_true", default=True)
    ap.add_argument("--rollout-frac", type=float, default=0.5,
                    help="fraction of batches that include a self-generated prefix")
    ap.add_argument("--rollout-len", type=int, default=20)
    ap.add_argument("--block-start", type=int, default=24)
    ap.add_argument("--block-len", type=int, default=6)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval-every", type=int, default=100)
    args = ap.parse_args()

    setup(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    raw_train = load_prosqa("train", args.train_count, args.seed)
    train_rows = []
    for r in raw_train:
        row = prosqa_row_to_gsm8k_shape(r)
        row["steps"] = r["steps"]
        train_rows.append(row)
    raw_dev = load_prosqa("valid", args.dev_count, args.seed)
    dev_rows = []
    for r in raw_dev:
        row = prosqa_row_to_gsm8k_shape(r)
        row["steps"] = r["steps"]
        dev_rows.append(row)

    model = ProsQAModel(checkpoint=ROOT / args.resume, train_depth=args.train_depth,
                        block_start=args.block_start, block_len=args.block_len,
                        rank=args.rank)
    params = model.trainable_parameters()
    print(f"trainable={sum(p.numel() for p in params)/1e6:.3f}M resume={args.resume} "
          f"train={len(train_rows)} dev={len(dev_rows)}", flush=True)

    model.eval()
    em0 = strict_free_em(model, dev_rows[:20])
    print(f"pre-training strict free EM (depth 1, 16 tok): {em0:.3f}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    history = [{"step": 0, "strict_free_em_d1": em0}]
    torch.cuda.reset_peak_memory_stats()
    rng = random.Random(args.seed)
    step = 0
    t0 = time.perf_counter()
    model.train()
    while step < args.max_steps:
        order = list(range(len(train_rows)))
        random.Random(args.seed * 100 + step).shuffle(order)
        for index in order:
            if step >= args.max_steps:
                break
            row = train_rows[index]
            start = time.perf_counter()
            depth = rng.randint(1, args.train_depth) if args.depth_sampling else args.train_depth
            prompt = model.prefix(row["question"])
            target = model.target(row)

            use_rollout = rng.random() < args.rollout_frac
            prefix = []
            if use_rollout:
                model.eval()
                gen = rollout(model, prompt, depth, args.rollout_len)
                model.train()
                if gen:
                    t = rng.randint(1, len(gen))
                    prefix = gen[:t]
            loss = answer_loss(model, prompt, prefix, target, depth)
            (loss / args.accum).backward()
            grad_norm = None
            if (step + 1) % args.accum == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            record = {"step": step, "depth": depth, "rollout": len(prefix),
                      "loss": round(float(loss.detach()), 3),
                      "seconds": round(time.perf_counter() - start, 2),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            if grad_norm is not None:
                record["grad_norm"] = round(grad_norm, 3)
            history.append(record)
            if step % 50 == 0:
                print(json.dumps(record), flush=True)
            if args.eval_every and step % args.eval_every == 0:
                model.eval()
                em1 = strict_free_em(model, dev_rows[:20])
                entry = {"step": step, "strict_free_em_d1": em1}
                history.append(entry)
                print("EVAL " + json.dumps(entry), flush=True)
                (run_dir / "history.json").write_text(json.dumps(history, indent=2),
                                                      encoding="utf-8")
                model.save(run_dir / "checkpoint", extra={"args": vars(args), "steps": step})
                model.train()

    model.save(run_dir / "checkpoint",
               extra={"args": vars(args), "steps": step,
                      "train_seconds": round(time.perf_counter() - t0, 1)})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done: {step} steps in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
