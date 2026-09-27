"""Final evaluation: depth curve on ProsQA test set.

The trained model at depth K uses K latent iterations to compute the answer from the
question alone (no visible reasoning steps).  This is the test of the recurrent-depth
hypothesis: does more latent computation improve reasoning accuracy?

Also reports the frozen base model (no deltas, no loop) for reference.
"""

import json
import random
import time

import torch

from prosqa_adapter import ProsQAModel, load_prosqa, prosqa_accuracy, extract_prosqa_answer
from recurrent_qwen import ROOT, setup


@torch.no_grad()
def base_generate(model_core, tokenizer, question, max_new_tokens):
    messages = [
        {"role": "system", "content": "Determine the logical relationship by reasoning step by step."},
        {"role": "user", "content": question},
    ]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(encoded, list):
        encoded = list(encoded["input_ids"])
    ids = [int(t) for t in encoded]
    generated = []
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        out = model_core(input_ids=torch.tensor([ids], device="cuda"))
        token = int(out.logits[0, -1].argmax(-1).item())
        generated.append(token)
        ids.append(token)
        if token == tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_prosqa_answer(text), len(generated), time.perf_counter() - start, text


def main():
    setup(11)
    model = ProsQAModel(checkpoint=ROOT / "runs/pq-v1/checkpoint")
    model.eval()

    test_rows = load_prosqa("test", 100, seed=11)
    print(f"evaluating {len(test_rows)} ProsQA test questions\n")

    depths = [1, 2, 4, 8, 16]
    results = {"depths": {}, "base": {}, "records": []}

    # base model (no deltas, no loop) — use the full CausalLM for lm_head
    base_model = model.model  # Qwen3ForCausalLM (original weights, not PeftModel)
    tok = model.tokenizer
    base_correct = 0
    base_tokens = 0
    base_time = 0.0
    for row in test_rows[:30]:
        pred, ntok, sec, _ = base_generate(base_model, tok, row["question"], 48)
        ok = prosqa_accuracy(pred, row["answer"])
        base_correct += int(ok)
        base_tokens += ntok
        base_time += sec
    results["base"] = {"accuracy": base_correct / 30, "correct": base_correct,
                       "mean_tokens": round(base_tokens / 30, 1),
                       "mean_seconds": round(base_time / 30, 2)}
    print(f"base model (no loop, no deltas): acc={base_correct/30:.3f} tokens={base_tokens/30:.1f}", flush=True)

    # trained model at various depths
    for depth in depths:
        correct = 0
        total_tokens = 0
        total_time = 0.0
        records = []
        for row_idx, row in enumerate(test_rows):
            out = model.generate(row["question"], depth=depth, max_new_tokens=24)
            pred = extract_prosqa_answer("#### " + out["text"]) if "####" in out["text"] else out["text"].strip()
            ok = prosqa_accuracy(pred, row["answer"])
            correct += int(ok)
            total_tokens += out["generated_tokens"]
            total_time += out["seconds"]
            records.append({"idx": row_idx, "gold": row["answer"], "pred": pred, "ok": ok,
                            "depth": depth, "text": out["text"][:200]})
        entry = {"accuracy": correct / len(test_rows), "correct": correct,
                 "n": len(test_rows), "mean_tokens": round(total_tokens / len(test_rows), 1),
                 "mean_seconds": round(total_time / len(test_rows), 2)}
        results["depths"][depth] = entry
        results["records"].extend(records)
        print(f"trained depth {depth:>2}: acc={entry['accuracy']:.3f} ({correct}/{len(test_rows)}) "
              f"tokens={entry['mean_tokens']:.1f} {entry['mean_seconds']:.2f}s/q", flush=True)

    out_path = ROOT / "results" / "pq_v1_depth_curve.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
