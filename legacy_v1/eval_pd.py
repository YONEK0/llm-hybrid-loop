"""Final strict evaluation of the pd-v1 scheduled-sampling checkpoint (ProsQA test).

30 test questions (seed 11, same subset as every other eval in this project), strict
extractor, 16-token budget, depths 1 and 4.  Reference points: base model strict 0/30
at 16/48/1024 tokens; pq-v1 (teacher-forced checkpoint) strict 0/30.

Run: .venv/Scripts/python.exe eval_pd.py
"""

import json
import time

import torch

from prosqa_adapter import ProsQAModel, load_prosqa, prosqa_accuracy
from recurrent_qwen import ROOT, setup
from train_direct import extract_strict

MAX_TOKENS = 16


@torch.no_grad()
def free_gen(model, ids, depth):
    cur = torch.tensor([ids], dtype=torch.long, device="cuda")
    gen = []
    t0 = time.perf_counter()
    for _ in range(MAX_TOKENS):
        logits, _ = model.forward(cur, depth=depth)
        t = int(logits[0, -1].argmax(-1).item())
        gen.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device="cuda")], dim=1)
        if t == model.tokenizer.eos_token_id:
            break
    return gen, time.perf_counter() - t0


def main():
    setup(11)
    model = ProsQAModel(checkpoint=ROOT / "runs/pd-v1/checkpoint")
    model.eval()
    rows = load_prosqa("test", 30, seed=11)

    results = {"model": "pd-v1 (scheduled-sampling, step-400)", "task": "ProsQA test",
               "n": 30, "budget": MAX_TOKENS, "scorer": "strict", "depths": {}}
    for depth in (1, 4):
        hits = 0
        samples = []
        for row in rows:
            ids = model.prefix(row["question"])
            gen, sec = free_gen(model, ids, depth)
            text = model.tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_strict(text)
            ok = prosqa_accuracy(pred, row["answer"])
            hits += int(ok)
            if len(samples) < 3:
                samples.append({"gold": row["answer"], "strict_pred": pred, "ok": bool(ok),
                                "text": text[:150]})
        results["depths"][str(depth)] = {"strict_em": hits / 30, "correct": hits,
                                         "samples": samples}
        print(f"depth={depth}: strict EM={hits}/30", flush=True)

    out = ROOT / "results" / "pd_v1_strict_eval.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
