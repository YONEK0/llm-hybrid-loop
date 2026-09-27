"""Scaled 3:1 token-compression evaluation on ProsQA (n=150, paired).

Upgrades results/pq_v1_fast_eval.json (n=30) with:
  * 3x more questions (150 of 500 test rows, seed 11 -> first 30 match prior runs)
  * the missing fair control: base model at the SAME 16-token budget
  * Wilson 95% CIs, wall-clock per question, and exact McNemar paired tests

Conditions (same 150 questions, greedy):
  base_48    frozen Qwen3-4B-Thinking (4-bit), step-by-step system prompt, 48-token budget
  base_16    identical, 16-token budget
  trained_16 pq-v1 checkpoint, depth=1 free generation, 16-token budget

Run: .venv/Scripts/python.exe eval_compression_scaled.py [--count 150]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from prosqa_adapter import ProsQAModel, load_prosqa, prosqa_accuracy, extract_prosqa_answer
from recurrent_qwen import ROOT, setup


def wilson_ci(correct, n, z=1.96):
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(center - half, 4), round(center + half, 4)


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar on paired correctness vectors. Returns (n01, n10, p)."""
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)  # a wrong, b right
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    k, m = min(n01, n10), n01 + n10
    if m == 0:
        return n01, n10, 1.0
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / 2 ** m
    return n01, n10, min(1.0, 2 * p)


@torch.no_grad()
def generate_base_kv(base, tok, enc_ids, budget):
    """Greedy generation with KV cache (identical logits to full recompute, faster)."""
    past = None
    cur = torch.tensor([enc_ids], dtype=torch.long, device="cuda")
    generated, t0 = [], time.perf_counter()
    for _ in range(budget):
        kwargs = {"use_cache": True}
        if past is not None:
            kwargs["input_ids"] = cur[:, -1:]
            kwargs["past_key_values"] = past
        else:
            kwargs["input_ids"] = cur
        out = base(**kwargs)
        past = out.past_key_values
        t = int(out.logits[0, -1].argmax(-1).item())
        generated.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device="cuda")], dim=1)
        if t == tok.eos_token_id:
            break
    return generated, time.perf_counter() - t0


@torch.no_grad()
def generate_trained(model, enc_ids, budget, depth=1):
    cur = torch.tensor([enc_ids], dtype=torch.long, device="cuda")
    generated, t0 = [], time.perf_counter()
    for _ in range(budget):
        logits, _ = model.forward(cur, depth=depth)
        t = int(logits[0, -1].argmax(-1).item())
        generated.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device="cuda")], dim=1)
        if t == model.tokenizer.eos_token_id:
            break
    return generated, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--checkpoint", default="runs/pq-v1/checkpoint")
    ap.add_argument("--tag", default="compression_scaled_prosqa")
    args = ap.parse_args()

    setup(11)
    model = ProsQAModel(checkpoint=ROOT / args.checkpoint)
    model.eval()
    base, tok = model.model, model.tokenizer
    system = "Determine the logical relationship by reasoning step by step."

    rows = load_prosqa("test", args.count, seed=11)
    n = len(rows)
    print(f"n={n}, checkpoint={args.checkpoint}", flush=True)

    conds = ["base_48", "base_16", "trained_16"]
    res = {c: {"correct_vec": [], "tokens": [], "seconds": [], "texts": []} for c in conds}

    for i, row in enumerate(rows):
        gold = row["answer"]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if not isinstance(enc, list):
            enc = list(enc["input_ids"])
        enc = [int(t) for t in enc]

        for cond, budget in (("base_48", 48), ("base_16", 16)):
            gen, sec = generate_base_kv(base, tok, enc, budget)
            text = tok.decode(gen, skip_special_tokens=True)
            ok = prosqa_accuracy(extract_prosqa_answer(text), gold)
            res[cond]["correct_vec"].append(bool(ok))
            res[cond]["tokens"].append(len(gen))
            res[cond]["seconds"].append(sec)
            res[cond]["texts"].append(text)

        ids = model.prefix(row["question"])
        gen, sec = generate_trained(model, ids, 16)
        text = tok.decode(gen, skip_special_tokens=True)
        ok = prosqa_accuracy(extract_prosqa_answer("#### " + text if "####" in text else text), gold)
        res["trained_16"]["correct_vec"].append(bool(ok))
        res["trained_16"]["tokens"].append(len(gen))
        res["trained_16"]["seconds"].append(sec)
        res["trained_16"]["texts"].append(text)

        if (i + 1) % 25 == 0:
            msg = " ".join(
                f"{c}={sum(res[c]['correct_vec'])}/{len(res[c]['correct_vec'])}" for c in conds)
            print(f"  [{i+1}/{n}] {msg}", flush=True)

    out = {"model": "Qwen3-4B-Thinking-2507-bnb-4bit + pq-v1", "task": "ProsQA test",
           "n": n, "budgets": {"base": 48, "trained": 16}, "conditions": {}}
    for c in conds:
        cv = res[c]["correct_vec"]
        k = sum(cv)
        out["conditions"][c] = {
            "accuracy": round(k / n, 4), "correct": k,
            "wilson95": wilson_ci(k, n),
            "mean_tokens": round(sum(res[c]["tokens"]) / n, 1),
            "mean_seconds": round(sum(res[c]["seconds"]) / n, 3),
            "correct_vec": cv,
        }

    out["paired"] = {}
    for ref in ("base_48", "base_16"):
        n01, n10, p = mcnemar_exact(res["trained_16"]["correct_vec"], res[ref]["correct_vec"])
        out["paired"][f"trained_16_vs_{ref}"] = {
            "trained_right_ref_wrong": n10, "trained_wrong_ref_right": n01,
            "mcnemar_p": round(p, 4)}

    path = ROOT / "results" / f"{args.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== token compression, ProsQA n={n} =====")
    for c in conds:
        e = out["conditions"][c]
        print(f"  {c:11s}: acc={e['accuracy']:.3f} (95% CI {e['wilson95']}) "
              f"tokens={e['mean_tokens']} {e['mean_seconds']}s/q")
    for k, v in out["paired"].items():
        print(f"  {k}: +{v['trained_right_ref_wrong']}/-{v['trained_wrong_ref_right']} p={v['mcnemar_p']}")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
