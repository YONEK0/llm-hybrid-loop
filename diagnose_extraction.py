"""Diagnostic: how much of the ProsQA free-generation score is an extraction artifact?

The loose extractor (prosqa_adapter.extract_prosqa_answer) returns the last
"<Name> is a <thing>" pattern anywhere in the visible text.  A model that opens with
"Okay, let's try to figure out if Eva is a lorpus or yumpus..." gets credited whenever
the gold class is the first candidate it merely *mentions*.  This script reruns the
three compression-eval conditions on the canonical 30 questions (seed 11), saves the
full visible texts, and scores each with:

  loose   - the original extractor (should reproduce pq_v1_fast_eval.json)
  strict  - only a genuine final answer counts: text after "####", or a last line that
            IS a complete "<Name> is [not] a <thing>" statement (no reasoning filler)

and reports the artifact rate: loose-correct verdicts whose match sits in the FIRST
line of the visible text (question restatement, not an answer).

Run: .venv/Scripts/python.exe diagnose_extraction.py
"""

import json
import re

import torch

from prosqa_adapter import (ProsQAModel, load_prosqa, prosqa_accuracy,
                            extract_prosqa_answer)
from recurrent_qwen import ROOT, setup

from eval_compression_scaled import generate_base_kv, generate_trained


def extract_strict(text):
    if "####" in text:
        m = re.search(r"([A-Z][A-Za-z]* is a [a-z]+)", text.rsplit("####", 1)[-1])
        return m.group(1) if m else None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        m = re.fullmatch(r"([A-Z][A-Za-z]* is [a-z ]*a [a-z]+)\.?", lines[-1])
        if m:
            return m.group(1)
    return None


def first_line_match(text):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    return bool(re.search(r"([A-Z][A-Za-z]* is a [a-z]+)", lines[0]))


def main():
    setup(11)
    model = ProsQAModel(checkpoint=ROOT / "runs/pq-v1/checkpoint")
    model.eval()
    base, tok = model.model, model.tokenizer
    system = "Determine the logical relationship by reasoning step by step."

    rows = load_prosqa("test", 30, seed=11)
    conds = ("base_48", "base_16", "trained_16")
    rec = {c: [] for c in conds}

    for i, row in enumerate(rows):
        gold = row["answer"]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": row["question"]}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if not isinstance(enc, list):
            enc = list(enc["input_ids"])
        enc = [int(t) for t in enc]

        for cond, budget in (("base_48", 48), ("base_16", 16)):
            gen, _ = generate_base_kv(base, tok, enc, budget)
            rec[cond].append({"gold": gold, "text": tok.decode(gen, skip_special_tokens=True)})
        ids = model.prefix(row["question"])
        gen, _ = generate_trained(model, ids, 16)
        rec["trained_16"].append(
            {"gold": gold, "text": tok.decode(gen, skip_special_tokens=True)})
        print(f"[{i+1}/30]", flush=True)

    out = {"n": 30, "conditions": {}}
    for cond in conds:
        loose = strict = artifact = 0
        rows_out = []
        for r in rec[cond]:
            pl = extract_prosqa_answer(r["text"])
            ps = extract_strict(r["text"])
            ok_l = prosqa_accuracy(pl, r["gold"])
            ok_s = prosqa_accuracy(ps, r["gold"])
            loose += ok_l
            strict += ok_s
            artifact += bool(ok_l and first_line_match(r["text"]))
            rows_out.append({"gold": r["gold"], "loose_pred": pl, "strict_pred": ps,
                             "loose_ok": bool(ok_l), "strict_ok": bool(ok_s),
                             "open_line": (r["text"].split("\n")[0][:90]),
                             "text": r["text"][:300]})
        out["conditions"][cond] = {"loose_acc": loose / 30, "strict_acc": strict / 30,
                                   "artifact_of_loose_correct": artifact, "rows": rows_out}
        print(f"{cond}: loose={loose}/30 strict={strict}/30 "
              f"loose-correct-with-open-line-match={artifact}", flush=True)

    path = ROOT / "results" / "extraction_artifact_diagnostic.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
