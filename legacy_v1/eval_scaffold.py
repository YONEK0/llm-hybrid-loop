"""Evaluate the actual claim: with fewer visible reasoning steps, does latent looping
recover the lost accuracy?

Protocol: give the model the *gold* first k rationale steps as context, then let it
generate the rest.  This matches how the model was trained (scaffold as input) and so
measures the thing the method claims -- that hidden computation can stand in for removed
chain-of-thought steps -- instead of measuring unassisted generation, which conflates it
with exposure bias.

Controls in the same run:
  depth 1        - deltas active but the block runs once (no extra latent computation)
  depth 2/4/8    - progressively more latent iterations
  zeroed deltas  - the loop with the trained deltas disabled

A 2-D grid (keep_steps x depth) is printed so the interaction is visible.
"""

import argparse
import json
from pathlib import Path
import random
import time

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup


@torch.no_grad()
def greedy_from_ids(model, ids, depth, max_new_tokens):
    ids = list(ids)
    generated = []
    for _ in range(max_new_tokens):
        logits, _ = model.forward(torch.tensor([ids], dtype=torch.long, device="cuda"), depth=depth)
        token = int(logits[0, -1].argmax(-1).item())
        generated.append(token)
        ids.append(token)
        if token == model.tokenizer.eos_token_id:
            break
    text = model.tokenizer.decode(generated, skip_special_tokens=True)
    return answer_from_text(text), text


def scaffold_prefix(model, row, keep_steps):
    ids = model.prefix(row["question"])
    steps = model.cot_steps(row)[:keep_steps] if keep_steps else []
    ids += [int(t) for t in model.tokenizer.encode("".join(s + "\n" for s in steps),
                                                   add_special_tokens=False)]
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/cur-v1/checkpoint")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--keep-steps", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="cur-v1")
    args = parser.parse_args()

    setup(args.seed)
    model = RecurrentDepthQwen(checkpoint=ROOT / args.checkpoint)
    rows = read_jsonl(ROOT / "data/gsm8k/test.jsonl")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    rows = [rows[i] for i in order[:args.count]]

    # sanity: are the trained deltas actually non-trivial?
    max_b = max(float(t.abs().max()) for k, t in model.deltas.state_dict().items() if ".B." in k)
    inj = float(model.injection.proj.weight[:, model.model.config.hidden_size:].abs().max())
    print(f"checkpoint sanity: max|B|={max_b:.4f}  injection_delta={inj:.4f}", flush=True)

    grid = {}
    t0 = time.perf_counter()
    for keep in args.keep_steps:
        for depth in args.depths:
            hits, records = 0, []
            for row in rows:
                ids = scaffold_prefix(model, row, keep)
                pred, text = greedy_from_ids(model, ids, depth, args.max_new_tokens)
                ok = pred == str(row["answer"]).strip()
                hits += int(ok)
                records.append({"id": row["id"], "gold": str(row["answer"]).strip(),
                                "pred": pred, "ok": ok, "text": text[:120]})
            acc = hits / len(rows)
            grid[f"k{keep}_d{depth}"] = {"accuracy": acc, "correct": hits, "n": len(rows),
                                         "keep_steps": keep, "depth": depth, "records": records}
            print(f"  keep={keep} depth={depth}: acc={acc:.3f} ({hits}/{len(rows)})  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    # control: trained deltas disabled
    print("\nzeroed-delta control:", flush=True)
    with torch.no_grad():
        for wrapper in model.deltas.values():
            for parameter in list(wrapper.B):
                parameter.zero_()
    for keep in args.keep_steps:
        hits = 0
        for row in rows[:15]:
            ids = scaffold_prefix(model, row, keep)
            pred, _ = greedy_from_ids(model, ids, 4, args.max_new_tokens)
            hits += int(pred == str(row["answer"]).strip())
        grid[f"zeroed_k{keep}_d4"] = {"accuracy": hits / 15, "correct": hits, "n": 15}
        print(f"  zeroed keep={keep} depth=4: acc={hits/15:.3f} ({hits}/15)", flush=True)

    result = {"checkpoint": args.checkpoint, "split": "test", "n": len(rows),
              "protocol": "gold first-k CoT steps given as context, then greedy generation",
              "max_delta_B": max_b, "injection_delta": inj, "grid": grid}
    out = ROOT / "results" / f"{args.tag}_scaffold_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== accuracy grid: rows = visible CoT steps, cols = latent depth ===")
    header = "keep\\depth | " + " | ".join(f"d{d}" for d in args.depths)
    print(header)
    for keep in args.keep_steps:
        cells = [f"{grid[f'k{keep}_d{d}']['accuracy']:.3f}" for d in args.depths]
        print(f"{keep:>10} | " + " | ".join(cells))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
