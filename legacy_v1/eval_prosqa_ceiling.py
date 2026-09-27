"""True ProsQA ceiling of the frozen base model: full thinking budget, strict scoring.

The compression evals capped the base model at 48 tokens, which only ever produces
truncated CoT (strict accuracy 0/30).  This measures what the base model actually
achieves on ProsQA when allowed to finish: 512-token budget, greedy, scored with the
strict extractor (a bare final statement or text after '####').

Run: .venv/Scripts/python.exe eval_prosqa_ceiling.py [--count 30] [--budget 512]
"""

import argparse
import json

import torch

from prosqa_adapter import ProsQAModel, load_prosqa, prosqa_accuracy
from recurrent_qwen import ROOT, setup
from eval_compression_scaled import generate_base_kv
from eval_ouro_depth import extract_answer_strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--budget", type=int, default=512)
    args = ap.parse_args()

    setup(11)
    model = ProsQAModel(checkpoint=ROOT / "runs/pq-v1/checkpoint")
    model.eval()
    base, tok = model.model, model.tokenizer
    system = "Determine the logical relationship by reasoning step by step."

    rows = load_prosqa("test", args.count, seed=11)
    n = len(rows)
    loose = strict = 0
    tokens_total = 0
    records = []
    for i, row in enumerate(rows):
        gold = row["answer"]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": row["question"]}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if not isinstance(enc, list):
            enc = list(enc["input_ids"])
        enc = [int(t) for t in enc]
        with torch.no_grad():
            gen, _ = generate_base_kv(base, tok, enc, args.budget)
        text = tok.decode(gen, skip_special_tokens=True)
        import re
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        p_l = None
        for ln in reversed(lines):
            m = re.search(r"([A-Z][A-Za-z]*\s+is\s+a\s+[a-z]+)", ln)
            if m:
                p_l = m.group(1)
                break
        p_s = extract_answer_strict(text)
        loose += int(prosqa_accuracy(p_l, gold))
        strict += int(prosqa_accuracy(p_s, gold))
        tokens_total += len(gen)
        records.append({"gold": gold, "loose_pred": p_l, "strict_pred": p_s,
                        "tokens": len(gen), "text": text[:400]})
        print(f"[{i+1}/{n}] strict={strict} loose={loose}", flush=True)

    out = {"model": "Qwen3-4B-Thinking-2507 (frozen base)", "task": "ProsQA test",
           "n": n, "budget": args.budget,
           "loose_accuracy": loose / n, "strict_accuracy": strict / n,
           "mean_tokens": round(tokens_total / n, 1), "records": records}
    path = ROOT / "results" / "prosqa_base_ceiling.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nbase ProsQA ceiling @ {args.budget} tokens: strict={strict}/{n} "
          f"loose={loose}/{n} mean_tokens={tokens_total/n:.1f}")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
