"""Evaluate in the *training* regime: score the continuation given the gold scaffold.

Why this protocol: the model is trained with the first k reasoning steps provided as
input (teacher-forced scaffold).  Free-running generation therefore measures exposure
bias as much as it measures latent computation, and it cannot answer "did the latent
loop replace the removed steps?".

So this script reports two complementary numbers per configuration:

  scored      - CE and exact-match on the answer tokens when the gold scaffold is fed in
                (matches training; measures whether the hidden computation suffices)
  generated   - free-running greedy accuracy (matches deployment; reported for honesty)

The interesting comparison is across `keep_steps`: if latent looping substitutes for
removed reasoning, accuracy should degrade *more slowly* as steps are removed when more
latent iterations are available.

Controls: depth 1 (deltas active, single block pass = no extra latent compute) and a
zeroed-delta run.
"""

import argparse
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup


@torch.no_grad()
def score_with_scaffold(model, row, keep_steps, depth):
    """Teacher-forced CE over the answer tokens plus exact match of the greedy answer."""
    prompt = model.prefix(row["question"])
    scaffold = model.scaffold_tokens(row, keep_steps)
    answer = [int(t) for t in model.tokenizer.encode(
        "</think>\n\n#### " + row["answer"], add_special_tokens=False)] + [model.tokenizer.eos_token_id]
    logits, _ = model.forward_split(prompt, scaffold, answer[:-1], depth=depth)
    absolute_start = len(prompt) if scaffold else len(prompt) - 1
    span_len = (len(scaffold) - 1 if scaffold else 0) + len(answer)
    labels = torch.tensor([scaffold[1:] + answer], dtype=torch.long, device="cuda").reshape(-1)
    logits = logits[:, absolute_start:absolute_start + span_len].reshape(-1, logits.shape[-1])
    ce = float(F.cross_entropy(logits, labels))
    # greedy answer on the answer-token segment only
    n_answer = len(answer)
    seg = logits[-n_answer:].argmax(-1)
    pred_ids = seg.tolist()
    text = model.tokenizer.decode(pred_ids, skip_special_tokens=True)
    pred = answer_from_text(text)
    return ce, pred == str(row["answer"]).strip(), pred


@torch.no_grad()
def free_generation(model, question, depth, max_new_tokens):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/cur-v2/checkpoint")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--keep-steps", type=int, nargs="+", default=[3, 2, 1, 0])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="cur-v2")
    args = parser.parse_args()

    setup(args.seed)
    model = RecurrentDepthQwen(checkpoint=ROOT / args.checkpoint)
    rows = read_jsonl(ROOT / "data/gsm8k/test.jsonl")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    rows = [rows[i] for i in order[:args.count]]

    max_b = max(float(t.abs().max()) for k, t in model.deltas.state_dict().items() if ".B." in k)
    inj = float(model.injection.proj.weight[:, model.model.config.hidden_size:].abs().max())
    print(f"checkpoint: max|B|={max_b:.4f}  injection_delta={inj:.4f}", flush=True)

    grid = {}
    t0 = time.perf_counter()
    for keep in args.keep_steps:
        for depth in args.depths:
            ces, hits, preds = [], 0, []
            for row in rows:
                ce, ok, pred = score_with_scaffold(model, row, keep, depth)
                ces.append(ce)
                hits += int(ok)
                preds.append(pred)
            entry = {"keep_steps": keep, "depth": depth, "mean_ce": sum(ces) / len(ces),
                     "exact_match": hits / len(rows), "correct": hits, "n": len(rows)}
            grid[f"k{keep}_d{depth}"] = entry
            print(f"  keep={keep} depth={depth}: CE={entry['mean_ce']:.3f} "
                  f"EM={entry['exact_match']:.3f} ({hits}/{len(rows)}) [{time.perf_counter()-t0:.0f}s]",
                  flush=True)

    print("\nfree-running generation (deployment-like):", flush=True)
    free = {}
    for keep in (args.keep_steps[0], 0):
        hits = 0
        for row in rows[:20]:
            pred = free_generation(model, row["question"], 4, args.max_new_tokens)
            hits += int(pred == str(row["answer"]).strip())
        free[f"free_k{keep}_d4"] = hits / 20
        print(f"  free k={keep} d=4: acc={hits/20:.3f} ({hits}/20)", flush=True)

    with torch.no_grad():
        for wrapper in model.deltas.values():
            for parameter in list(wrapper.B):
                parameter.zero_()
    zeroed = {}
    for keep in args.keep_steps:
        ces, hits = [], 0
        for row in rows[:20]:
            ce, ok, _ = score_with_scaffold(model, row, keep, 4)
            ces.append(ce)
            hits += int(ok)
        zeroed[f"zeroed_k{keep}_d4"] = {"mean_ce": sum(ces) / len(ces), "exact_match": hits / 20}
        print(f"  zeroed keep={keep} d=4: CE={sum(ces)/len(ces):.3f} EM={hits/20:.3f}", flush=True)

    result = {"checkpoint": args.checkpoint, "split": "test", "n": len(rows),
              "protocol": "gold scaffold fed as input; CE + exact match on answer tokens",
              "max_delta_B": max_b, "injection_delta": inj,
              "scaffold_grid": grid, "free_running": free, "zeroed_control": zeroed}
    out = ROOT / "results" / f"{args.tag}_scored_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== exact match: rows = visible steps, cols = latent depth ===")
    print("keep\\depth | " + " | ".join(f"d{d}" for d in args.depths))
    for keep in args.keep_steps:
        print(f"{keep:>10} | " + " | ".join(
            f"{grid[f'k{keep}_d{d}']['exact_match']:.3f}" for d in args.depths))
    print("\n=== mean CE (lower is better) ===")
    for keep in args.keep_steps:
        print(f"{keep:>10} | " + " | ".join(
            f"{grid[f'k{keep}_d{d}']['mean_ce']:.3f}" for d in args.depths))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
