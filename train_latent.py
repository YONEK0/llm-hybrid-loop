"""Train continuous latent reasoning on Qwen3-4B-Thinking (single 8GB GPU).

Method: Coconut-style continuous hidden-state feedback + CODI-style self-distillation
from the same model's explicit-CoT hidden states.  Only LoRA adapters and the latent
bridge are trained; the 4-bit base weights stay frozen.

Example:
    python train_latent.py --max-steps 20 --run-name pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from latent_qwen import LatentQwen, ROOT, read_jsonl, setup


def make_batch(rows, num_latent, device="cuda"):
    return rows


def evaluate(model, rows, mode="latent", max_new_tokens=48, limit=None, steps=None, ablation=None):
    rows = rows[:limit] if limit else rows
    correct = 0
    records = []
    for row in rows:
        out = model.generate(row["question"], mode=mode, max_new_tokens=max_new_tokens,
                             steps=steps, ablation=ablation)
        gold = str(row["answer"]).strip()
        ok = out["answer"] is not None and out["answer"] == gold
        correct += int(ok)
        records.append({"id": row["id"], "gold": gold, "pred": out["answer"], "ok": ok,
                        "seconds": round(out["seconds"], 3), "tokens": out["generated_tokens"],
                        "hit_limit": out["hit_length_limit"]})
    return {"accuracy": correct / max(len(rows), 1), "n": len(rows), "correct": correct,
            "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--valid-count", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--ref-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    setup(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")
    import random
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    valid = [rows[i] for i in order[:args.valid_count]]
    train = [rows[i] for i in order[args.valid_count:args.valid_count + args.train_count]]
    print(f"train={len(train)} valid={len(valid)} latent_steps={args.latent_steps}", flush=True)

    model = LatentQwen(checkpoint=args.resume, lora_rank=args.lora_rank, latent_steps=args.latent_steps)
    params = model.trainable_parameters()
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    history = []
    model.train()
    step = 0
    epoch = 0
    torch.cuda.reset_peak_memory_stats()
    while step < args.max_steps:
        epoch += 1
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            if step >= args.max_steps:
                break
            row = train[index]
            start = time.perf_counter()
            student_ce, kd, report = model.losses(
                row, kd_weight=args.kd_weight, steps=args.latent_steps)
            total = (student_ce + args.kd_weight * kd) / args.accum
            total.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            record = {"step": step, "epoch": epoch, "seconds": round(time.perf_counter() - start, 2),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), **report}
            history.append(record)
            print(json.dumps(record), flush=True)
            if args.eval_every and step % args.eval_every == 0:
                result = evaluate(model, valid, limit=min(8, len(valid)), max_new_tokens=args.max_new_tokens)
                record = {"step": step, "valid_acc": result["accuracy"], "valid_n": result["n"]}
                history.append(record)
                print("EVAL " + json.dumps(record), flush=True)
                model.train()

    # final step of any partial accumulation window
    if args.max_steps % args.accum != 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save(run_dir / "checkpoint", extra={"args": vars(args), "steps": step})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("running final validation ...", flush=True)
    for mode in ("cot", "direct", "latent"):
        result = evaluate(model, valid, mode=mode, max_new_tokens=args.max_new_tokens, limit=8)
        print(f"FINAL {mode}: acc={result['accuracy']:.3f} n={result['n']}", flush=True)
        (run_dir / f"final_{mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
