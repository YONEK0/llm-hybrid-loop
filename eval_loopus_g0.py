"""Phase 0-4 / G0: reproduce LoopUS depth gains locally (strict protocol).

Compare official LoopUS Qwen3-1.7B vs original Qwen3-1.7B base:
  A. WikiText-2 test perplexity at fixed depths d in {1,2,4,8}
     (paper: 1.7B ppl 21.1 -> 16.9 with LoopUS; lower with more depth)
  B. ARC-Challenge validation accuracy (multiple-choice log-likelihood
     scoring -> zero extraction-artifact risk, V1 lesson respected)

Depth control: load with n_recursion=d and q_stop_threshold=1.0 (never halts
early -> exactly d reasoning iterations). Base model = standard forward.

VRAM: models are evaluated sequentially (base first, then LoopUS), never both.

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe eval_loopus_g0.py
"""

import json
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH = "Qwen/Qwen3-1.7B"            # HF cache (downloaded via mirror)
LOOPUS_PATH = "models/LoopUS-Qwen3-1.7B"
WIKITEXT_WINDOWS = 60                     # 60 * 1024 tokens of PPL
ARC_MAX = 200                             # ARC-Challenge validation subset
MAX_LENGTH = 1024
DEPTHS = [1, 2, 4, 8]
OUT = "results/loopus_g0_depth_curve.json"


def load_loopus(n_recursion):
    import sys
    sys.path.insert(0, "loopus")
    from utils.inference import load_lds_model
    m = load_lds_model(model_name="Qwen/Qwen3-1.7B", device="cuda",
                       dtype=torch.bfloat16, decomposed_model=LOOPUS_PATH,
                       n_recursion=n_recursion, q_stop_threshold=1.0)
    return m


@torch.no_grad()
def wikitext_ppl(model, tok, is_loopus, depth=None):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text)
    ids = ids[:(WIKITEXT_WINDOWS + 1) * MAX_LENGTH]
    nll, n_tok = 0.0, 0
    for i in range(WIKITEXT_WINDOWS):
        window = torch.tensor([ids[i * MAX_LENGTH:(i + 1) * MAX_LENGTH]],
                              device="cuda")
        out = model(input_ids=window, attention_mask=torch.ones_like(window))
        logits = out.logits[:, :-1].float()
        targets = window[:, 1:]
        nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="sum").item()
        n_tok += targets.numel()
    return math.exp(nll / n_tok)


@torch.no_grad()
def arc_accuracy(model, tok, is_loopus, depth=None):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    rows = list(ds)[:ARC_MAX]
    correct = 0
    for row in rows:
        q = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold = row["answerKey"].strip()
        best, best_ce = None, None
        for ci, ch in enumerate(choices):
            prompt = f"Question: {q}\nAnswer:"
            full = f"Question: {q}\nAnswer: {ch}"
            p_ids = tok.encode(prompt)
            f_ids = tok.encode(full)
            if len(f_ids) > MAX_LENGTH:
                f_ids = f_ids[-MAX_LENGTH:]
                p_ids = p_ids[-MAX_LENGTH:]
            x = torch.tensor([f_ids], device="cuda")
            logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[:, :-1].float()
            # CE over the answer continuation tokens only
            n_cont = len(f_ids) - len(p_ids)
            ce = torch.nn.functional.cross_entropy(
                logits[0, -n_cont - 1:-1], x[0, -n_cont:], reduction="sum")
            if best_ce is None or ce.item() < best_ce:
                best_ce, best = ce.item(), ci
        correct += int(best is not None and labels[best].strip() == gold)
    return correct / len(rows), correct, len(rows)


def eval_model(which):
    results = {"perplexity": {}, "arc_challenge": {}}
    if which == "base":
        tok = AutoTokenizer.from_pretrained(BASE_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_PATH, dtype=torch.bfloat16).cuda().eval()
        t0 = time.perf_counter()
        results["perplexity"]["base"] = round(wikitext_ppl(model, tok, False), 3)
        acc, c, n = arc_accuracy(model, tok, False)
        results["arc_challenge"]["base"] = {
            "acc": round(acc, 4), "correct": c, "n": n}
        print(f"base: ppl={results['perplexity']['base']} "
              f"arc-c={acc:.3f} ({c}/{n}) [{time.perf_counter()-t0:.0f}s]",
              flush=True)
    else:
        tok = None
        for depth in DEPTHS:
            m = load_loopus(depth)
            if tok is None:
                tok = m.tokenizer
            t0 = time.perf_counter()
            ppl = wikitext_ppl(m, tok, True, depth)
            acc, c, n = arc_accuracy(m, tok, True, depth)
            results["perplexity"][f"d{depth}"] = round(ppl, 3)
            results["arc_challenge"][f"d{depth}"] = {
                "acc": round(acc, 4), "correct": c, "n": n}
            print(f"loopus d={depth}: ppl={ppl:.3f} arc-c={acc:.3f} "
                  f"({c}/{n}) [{time.perf_counter()-t0:.0f}s]", flush=True)
            del m
            torch.cuda.empty_cache()
    return results


def main():
    out = {"protocol": {"wikitext_windows": WIKITEXT_WINDOWS,
                        "max_length": MAX_LENGTH, "arc_n": ARC_MAX,
                        "depths": DEPTHS, "scorer": "fixed-depth, q_th=1.0"}}
    out["base_qwen3_1_7b"] = eval_model("base")
    torch.cuda.empty_cache()
    out["loopus_qwen3_1_7b"] = eval_model("loopus")

    path = OUT
    open(path, "w", encoding="utf-8").write(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\n===== G0 SUMMARY =====")
    print("ppl  base vs loopus d1/d2/d4/d8:",
          out["base_qwen3_1_7b"]["perplexity"]["base"],
          [out["loopus_qwen3_1_7b"]["perplexity"][f"d{d}"] for d in DEPTHS])
    print("arc  base vs loopus d1/d2/d4/d8:",
          out["base_qwen3_1_7b"]["arc_challenge"]["base"]["acc"],
          [out["loopus_qwen3_1_7b"]["arc_challenge"][f"d{d}"]["acc"]
           for d in DEPTHS])
    print("written to", path)


if __name__ == "__main__":
    main()
