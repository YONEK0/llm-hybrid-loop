"""Baseline: how well does the frozen base model score on GSM8K test?

Two references the latent loop should be compared against:

  think  - normal generation with a large token budget (the model reasons in text)
  direct - the <think> block is forced closed, so the model must answer from its
           first forward pass; this matches the information budget our loop gets

Generation uses the KV cache, which the looped forward pass cannot use, so this script
does not reuse recurrent_qwen's forward.
"""

import argparse
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recurrent_qwen import MODEL_DIR, ROOT, answer_from_text, read_jsonl, setup


@torch.no_grad()
def generate(model, tokenizer, ids, max_new_tokens):
    ids = list(ids)
    generated = []
    past = None
    hidden = None
    start = time.perf_counter()
    current = torch.tensor([ids], device="cuda")
    for step in range(max_new_tokens):
        out = model(input_ids=current, past_key_values=past, use_cache=True)
        past = out.past_key_values
        token = int(out.logits[0, -1].argmax(-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        current = torch.tensor([[token]], device="cuda")
    torch.cuda.synchronize()
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated), time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--think-budget", type=int, default=256)
    parser.add_argument("--direct-budget", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    setup(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")
    model.eval()

    rows = read_jsonl(ROOT / "data/gsm8k/test.jsonl")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    rows = [rows[i] for i in order[:args.count]]

    system = "Solve the math problem. Give the final numeric answer in the format #### number."
    result = {"model": "Qwen/Qwen3-4B-Thinking-2507 (4-bit NF4, frozen, KV-cache decoding)",
              "n": len(rows), "modes": {}}

    for mode, budget in (("think", args.think_budget), ("direct", args.direct_budget)):
        correct, tokens, seconds = 0, 0, 0.0
        records = []
        for row in rows:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if not isinstance(encoded, list):
                encoded = list(encoded["input_ids"])
            ids = [int(t) for t in encoded]
            if mode == "direct":
                ids += tokenizer.encode("</think>\n\n####", add_special_tokens=False)
            text, ntok, sec = generate(model, tokenizer, ids, budget)
            if mode == "direct":
                text = "####" + text
            pred = answer_from_text(text)
            ok = pred == str(row["answer"]).strip()
            correct += int(ok)
            tokens += ntok
            seconds += sec
            records.append({"id": row["id"], "gold": str(row["answer"]).strip(), "pred": pred,
                            "ok": ok, "tokens": ntok, "text": text[:300]})
        result["modes"][mode] = {"accuracy": correct / len(rows), "correct": correct, "n": len(rows),
                                 "budget": budget, "mean_tokens": round(tokens / len(rows), 1),
                                 "mean_seconds": round(seconds / len(rows), 2),
                                 "records": records}
        print(f"{mode:>6}: acc={correct/len(rows):.3f} ({correct}/{len(rows)}) "
              f"tokens={tokens/len(rows):.1f} {seconds/len(rows):.2f}s/q", flush=True)

    out = ROOT / "results" / "base_model_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
