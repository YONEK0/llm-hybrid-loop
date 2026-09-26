"""Strict answer extractors + 20-case adversarial validity gate (V5 S2, rule 5).

Rules frozen from the V1 artifact postmortem (docs/reports/PROJECT_REVIEW.md):
  GSM8K: score only if generation FINISHED (EOS within budget) AND text contains '####';
         answer = normalized number after the LAST '####'. No truncated-tail fallback.
  ProsQA: score only '#### <statement>' (anchored) or a final line that is a complete,
          standalone '<Name> is a <thing>.' statement. NEVER scan the whole text for
          'X is a Y' — that is the artifact that fabricated all V1 free-running hits.

Run: loopus_env/Scripts/python.exe scorer_strict.py   (writes results/s2_scorer_gate.json)
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/s2_scorer_gate.json"

STMT = re.compile(r"^([A-Z][A-Za-z]*)\s+is\s+a\s+([a-z]+)\.?$", re.IGNORECASE)
MARKER = re.compile(r"^(?:therefore|thus|so|hence)[,]?\s+(.+)$", re.IGNORECASE)


def _statement_of(sentence):
    """Full statement, optionally led by a closed whitelist of discourse markers."""
    m = STMT.fullmatch(sentence)
    if not m:
        mk = MARKER.match(sentence)
        m = STMT.fullmatch(mk.group(1)) if mk else None
    return m


def _norm_num(text):
    t = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", t):
        return None
    if "." in t:
        t = t.rstrip("0").rstrip(".")
        return t if t not in ("", "-") else "0"
    return str(int(t))


def extract_gsm8k_strict(text, finished):
    """None unless finished AND '#### <number>' present. No fallback of any kind."""
    if not finished or "####" not in text:
        return None
    return _norm_num(text.rsplit("####", 1)[1])


def extract_prosqa_strict(text, finished):
    """'#### <statement>' (anchored at segment start) or complete bare final line."""
    if "####" in text:
        seg = text.rsplit("####", 1)[1].strip()
        m = STMT.match(seg)
        if m:
            return f"{m.group(1)} is a {m.group(2)}".lower()
        seg_head = seg.split("\n")[0].strip()
        m = STMT.match(seg_head)
        if m:
            return f"{m.group(1)} is a {m.group(2)}".lower()
        return None
    if not finished:
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    # last SENTENCE of the last line must be a complete standalone statement
    sentences = re.split(r"(?<=[.!?])\s+", lines[-1])
    m = _statement_of(sentences[-1].strip()) if sentences else None
    if m:
        return f"{m.group(1)} is a {m.group(2)}".lower()
    return None


GOLD_P = "eva is a lorpus"

CASES = [
    # --- GSM8K (gold 13 unless noted) ---
    ("g1 #### hit",              "Let me compute. 6+7=13. #### 13", True, "13"),
    ("g2 last #### wins",        "#### 7 then revise: #### 13", True, "13"),
    ("g3 truncated mid-CoT",     "First 24 plus 8 is 32, and then", False, None),
    ("g4 finished but no ####",  "so the total is 32 dollars", True, None),
    ("g5 thousands separator",   "#### 1,234", True, "1234"),
    ("g6 currency",              "#### $45", True, "45"),
    ("g7 negative decimal",      "#### -3.50", True, "-3.5"),
    ("g8 empty",                 "", True, None),
    ("g9 #### but no number",    "#### done", True, None),
    # --- ProsQA (gold: eva is a lorpus) ---
    ("p1 restatement truncated", "Okay, let's figure out if Eva is a lorpus or a zhorpus. First", False, None),
    ("p2 restatement finished",  "Okay, let's figure out if Eva is a lorpus or a zhorpus.", True, None),
    ("p3 #### statement",        "Checking paths. #### Eva is a lorpus.", True, GOLD_P),
    ("p4 #### with prefix words", "#### So the answer is Eva is a lorpus.", True, None),
    ("p5 bare final statement",  "Rule 1 applies. Therefore, Eva is a lorpus.", True, GOLD_P),
    ("p6 final line continues",  "Therefore, Eva is a lorpus. Next I check zhorpus traits.", True, None),
    ("p7 mid-text only",         "Candidate: Eva is a lorpus.\nBut maybe Eva is a zhorpus.", False, None),
    ("p8 #### wrong entity",     "#### Eva is a zhorpus.", True, "eva is a zhorpus"),
    ("p9 #### lowercase",        "#### eva is a lorpus", True, GOLD_P),
    ("p10 #### then ramble",     "#### Eva is a lorpus.\nWait, let me double check the rule.", True, GOLD_P),
    ("p11 question final line",  "Is Eva a lorpus or a zhorpus?", True, None),
]


def gsm8k_finished(text, eos_hit=False):
    """Protocol v1.1 amended 'finished': EOS, or the answer segment is CLOSED by
    a natural boundary -- a '#### <number>' whose tail is whitespace-only or
    followed by a line-start 'Q:' (new exemplar question). Conservative:
    continued reasoning after the number keeps the item unfinished."""
    if eos_hit:
        return True
    if "####" not in text:
        return False
    seg = text.rsplit("####", 1)[1]
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", seg)
    if m is None:
        return False
    rest = seg[m.end():]
    return rest.strip() == "" or re.search(r"(^|\n)\s*Q:", rest) is not None


def extract_gsm8k_v11(text, eos_hit=False):
    """Protocol v1.1 GSM8K extraction. EOS path keeps the ORIGINAL strict
    semantics (pure number only). Boundary path (no EOS): answer segment must
    be boundary-closed, then its first number is accepted."""
    if eos_hit:
        return extract_gsm8k_strict(text, True)
    if not gsm8k_finished(text, False):
        return None
    seg = text.rsplit("####", 1)[1]
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", seg)
    return _norm_num(m.group(0)) if m else None



BOUNDARY_CASES = [
    ("b1 eos plain",               "so the answer is 72", True, True, None),
    ("b2 eos with hash",           "#### 13", True, True, "13"),
    ("b3 boundary Q after",        "#### 15\n\nQ: A family of 4...", False, True, "15"),
    ("b4 boundary immediate Q",    "#### 15\nQ: next", False, True, "15"),
    ("b5 end-of-budget bare",      "#### 15", False, True, "15"),
    ("b6 trailing spaces",         "#### 15   ", False, True, "15"),
    ("b7 mid-reasoning continues", "#### 60 for the miles. The total cost is", False, False, None),
    ("b8 no hash no eos",          "First 24 plus 8 is 32, and then", False, False, None),
    ("b9 hash no number boundary", "#### done\n\nQ: next", False, False, None),
    ("b10 last hash wins",         "#### 7 then revise: #### 13\n\nQ: x", False, True, "13"),
    ("b11 thousands",              "#### 1,234\n\nQ: x", False, True, "1234"),
    ("b12 currency",               "#### $45\n\nQ:", False, True, "45"),
    ("b13 negative decimal",       "#### -3.50\n\nQ:", False, True, "-3.5"),
    ("b14 units word then Q",      "#### 18 dollars\n\nQ:", False, True, "18"),
    ("b15 calc marker",            "<<2*3=6>> #### 6\n\nQ:", False, True, "6"),
    ("b16 zero half",              "#### 0.50\n\nQ:", False, True, "0.5"),
    ("b17 long ramble no Q",       "#### 15\nWait, let me double check the rule. Actually the answer might be different", False, False, None),
    ("b18 eos keeps pure-only",    "#### 13 blah", True, True, None),
    ("b19 hash no num then Q",     "calc then ####\nQ:", False, False, None),
    ("b20 short tail no Q",        "#### 15 ok", False, False, None),
    ("b21 wrong-vs-gold extract",  "#### 16\n\nQ:", False, True, "16"),
    ("b22 Q inside no hash",       "Q: trick inside\nA: no hash here", True, True, None),
]

def main():
    results, fails = [], 0
    for name, text, finished, expect in CASES:
        got = (extract_gsm8k_strict(text, finished) if name.startswith("g")
               else extract_prosqa_strict(text, finished))
        ok = got == expect
        fails += int(not ok)
        results.append({"case": name, "got": got, "expect": expect, "pass": ok})
    for name, text, eos, exp_fin, exp_pred in BOUNDARY_CASES:
        fin = gsm8k_finished(text, eos)
        pred = extract_gsm8k_v11(text, eos)
        ok = fin == exp_fin and pred == exp_pred
        fails += int(not ok)
        results.append({"case": name, "finished": fin, "got": pred,
                        "exp_fin": exp_fin, "expect": exp_pred, "pass": ok})
    n_total = len(CASES) + len(BOUNDARY_CASES)
    out = {"n_cases": n_total, "n_fail": fails,
           "verdict": "PASS" if fails == 0 else "FAIL", "cases": results}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        if not r["pass"]:
            print("FAIL", r)
    print(f"scorer adversarial gate: {n_total - fails}/{n_total} pass "
          f"-> {out['verdict']} ({OUT})")


if __name__ == "__main__":
    main()
