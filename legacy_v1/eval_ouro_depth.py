"""True depth-accuracy sweep on Ouro looped LMs (ProsQA test, 30 questions).

modeling_ouro.py always runs all `total_ut_steps` (=4) iterations of the shared
block and appends each iteration's normalized hidden state to hidden_states_list.
The forward kwarg `exit_at_step=k` selects hidden_states_list[k] as the source of
logits, so we can read the model "as if" it had exited after k+1 iterations while
the KV cache stays identical.  Default inference (early_exit_threshold=1.0, never
reached) exits at the last step, i.e. exit_at_step=3.

Run with ouro_env (transformers 4.57.6).

Usage: python eval_ouro_depth.py [model_dir_name]   (default Ouro-1.4B)
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Ouro-1.4B"
MODEL_DIR = str(ROOT / "models" / MODEL_NAME)
SYSTEM = "Solve the logic problem. Give the final answer."
N_QUESTIONS = 30
MAX_NEW_TOKENS = 16


def extract_answer(text):
    """Loose extractor (original protocol): last '<Name> is a <thing>' anywhere."""
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    else:
        text = text.split("</think>")[-1]
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"([A-Z][A-Za-z]*\s+is\s+a\s+[a-z]+)", ln)
        if m:
            return m.group(1)
    return lines[-1].rstrip(".") if lines else None


def extract_answer_strict(text):
    """Strict extractor: only a genuine final answer counts.

    Text after '####', or a last line that IS a bare '<Name> is [not] a <thing>'
    statement.  Restatements like 'Okay, let's try to figure out if Eva is a
    lorpus...' do NOT count (the loose extractor credits those, which inflated
    every pre-2026-09-19 free-generation number; see diagnose_extraction.py).
    """
    if "####" in text:
        m = re.search(r"([A-Z][A-Za-z]*\s+is\s+a\s+[a-z]+)", text.rsplit("####", 1)[-1])
        return m.group(1) if m else None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        m = re.fullmatch(r"([A-Z][A-Za-z]*\s+is\s+(?:not\s+)?a\s+[a-z]+)\.?", lines[-1])
        if m:
            return m.group(1)
    return None


@torch.no_grad()
def generate(model, tokenizer, ids, exit_at_step, max_new_tokens=MAX_NEW_TOKENS):
    generated = []
    current = torch.tensor([ids], dtype=torch.long, device="cuda")
    past = None
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        kwargs = {"input_ids": current, "use_cache": True, "exit_at_step": exit_at_step}
        if past is not None:
            kwargs["past_key_values"] = past
        out = model(**kwargs)
        past = out.past_key_values
        token = int(out.logits[0, -1].argmax(-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        current = torch.tensor([[token]], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, len(generated), time.perf_counter() - start


def main():
    random.seed(11)
    torch.manual_seed(11)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    total_ut = getattr(model.config, "total_ut_steps", 4)
    print(f"{MODEL_NAME}: total_ut_steps={total_ut}", flush=True)

    data_path = ROOT / "data" / "prosqa" / "prosqa_test.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    rows = rows[:N_QUESTIONS]

    results = {"model": MODEL_NAME, "task": "ProsQA test", "n": len(rows), "depths": {}}
    for depth in range(1, total_ut + 1):
        loose, strict = 0, 0
        total_tokens, total_time = 0, 0.0
        samples = []
        for row in rows:
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": row["question"]}]
            enc = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True)
            if not isinstance(enc, list):
                enc = list(enc["input_ids"])
            ids = [int(t) for t in enc]
            text, ntok, sec = generate(model, tokenizer, ids, depth - 1)
            pred_l = extract_answer(text)
            pred_s = extract_answer_strict(text)
            gold = row["answer"].strip().rstrip(".")
            loose += int(pred_l is not None and pred_l.lower() == gold.lower())
            strict += int(pred_s is not None and pred_s.lower() == gold.lower())
            total_tokens += ntok
            total_time += sec
            if len(samples) < 2:
                samples.append({"loose_pred": pred_l, "strict_pred": pred_s,
                                "gold": gold, "text": text[:120]})
        acc_l, acc_s = loose / len(rows), strict / len(rows)
        results["depths"][str(depth)] = {
            "loose_accuracy": acc_l, "strict_accuracy": acc_s,
            "loose_correct": loose, "strict_correct": strict,
            "mean_tokens": round(total_tokens / len(rows), 1),
            "mean_seconds": round(total_time / len(rows), 2),
            "samples": samples,
        }
        print(f"depth={depth}: loose={acc_l:.3f} strict={acc_s:.3f} "
              f"tokens={total_tokens/len(rows):.1f} time={total_time/len(rows):.2f}s",
              flush=True)

    out_path = ROOT / "results" / f"ouro_depth_sweep_{MODEL_NAME}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
