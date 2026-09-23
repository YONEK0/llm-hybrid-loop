"""s4_tier1_eval.py — tier-1 capability acceptance: final ckpt vs base (V5 L0).

Frozen-protocol compliant (TRAINING_PLAN_V5 "evaluation protocol freeze"):
- teacher-forced scored (answer-segment CE) and free-running strict
  (scorer_strict only, no fallback extraction) kept as separate evidence chains;
- free-running records raw text, pred, gold, has-####, EOS, n_tokens (rule 4);
- same items/order/budget across configs; paired 95% CI; Wilson interval;
- n < 1000 -> results labelled DIRECTIONAL (rule 6).

Configs: trained (ckpt gate+q_head, d in 1/2/4/8), frozen (fresh gate, same
depths), base (original forward, gate bypassed).

Local data note: data/gsm8k/test.jsonl is the reformatted official test split
with fields id/question/rationale/answer (answer = bare number); the standard
'#### N' form is recomposed from rationale + answer.

Usage: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s4_tier1_eval.py --run v5_l0
Writes results/s4_tier1_<run>.json
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s2_spike import LoopUSQwen35
from scorer_strict import extract_gsm8k_strict, _norm_num

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def fwd_logits(model, x, depth):
    """depth=None -> base identity forward (gate bypassed entirely)."""
    h = model.encode(x)
    if depth is None:
        h = model.block(h)
    else:
        for _ in range(depth):
            h = model.gate(model.block(h), h)
    return model.decode(h)


@torch.no_grad()
def answer_ce_d(model, ids, mask_start, depth):
    """Teacher-forced CE over answer tokens [mask_start:] only."""
    x = torch.tensor([ids], device="cuda")
    logits = fwd_logits(model, x, depth=depth)
    seg = logits[0, mask_start - 1:-1].float()
    tgt = x[0, mask_start:]
    return float(F.cross_entropy(seg, tgt.reshape(-1),
                                 reduction="sum").item() / max(1, tgt.numel()))


@torch.no_grad()
def generate(model, tok, prompt_ids, depth, max_new):
    ids = list(prompt_ids)
    eos = tok.eos_token_id
    for _ in range(max_new):
        x = torch.tensor([ids], device="cuda")
        logits = fwd_logits(model, x, depth=depth)
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        if t == eos:
            break
    new = ids[len(prompt_ids):]
    finished = bool(new and new[-1] == eos)
    text = tok.decode(new, skip_special_tokens=True)
    return text, finished, len(new)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - half), 4), round(min(1.0, c + half), 4)]


def paired_ci(deltas):
    n = len(deltas)
    if n < 2:
        return {"mean": None, "ci95": None, "n": n}
    m = sum(deltas) / n
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return {"mean": round(m, 4), "ci95": [round(m - half, 4), round(m + half, 4)],
            "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5_l0")
    ap.add_argument("--n-ce", type=int, default=300)
    ap.add_argument("--n-gen", type=int, default=12)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ctx", type=int, default=768)
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run

    items = []
    with open(ROOT / "data" / "gsm8k" / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = _norm_num(str(r["answer"]))
            if gold is not None:
                a_txt = r["rationale"] + "\n#### " + r["answer"]
                items.append({"q": r["question"], "a": a_txt, "gold": gold})
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    ce_items = [items[i] for i in order[:args.n_ce]]
    gen_items = [items[i] for i in order[args.n_ce:args.n_ce + args.n_gen]]
    n_ce, n_gen = len(ce_items), len(gen_items)

    out = {
        "run": args.run, "time": time.strftime("%F %T"),
        "protocol": ("frozen: teacher-forced=answer-segment CE only; free-running="
                     "scorer_strict only; raw text + pred + has_hash + EOS + n_tokens "
                     "recorded; same items/order/budget; paired 95% CI; Wilson"),
        "verdict_scope": f"DIRECTIONAL (n_ce={n_ce}<1000, n_gen={n_gen}<1000)",
        "seed": args.seed, "n_ce": n_ce, "n_gen": n_gen,
        "items_gsm8k_total": len(items),
        "teacher_forced_ce": {}, "free_running": {},
    }
    print(f"tier-1 @ {out['time']}  n_ce={n_ce} n_gen={n_gen}", flush=True)

    ck = torch.load(run_dir / "ckpt_latest.pt", map_location="cuda",
                    weights_only=False)
    out["ckpt_step"] = ck["step"]
    out["sup_tokens"] = ck["sup_tokens"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    torch.manual_seed(args.seed)
    model = LoopUSQwen35(backbone)
    model.gate.load_state_dict(ck["gate"])
    model.q_head.load_state_dict(ck["q_head"])
    model.eval()

    ce_corpus = []
    for it in ce_items:
        p_ids = tok.encode(it["q"] + "\n\n")
        ids = (p_ids + tok.encode(it["a"]))[:args.ctx]
        if len(ids) > len(p_ids) + 4:
            ce_corpus.append({"ids": ids, "mask_start": len(p_ids), "gold": it["gold"]})

    def sweep_ce(tag, depth):
        ces = []
        t0 = time.perf_counter()
        for rec in ce_corpus:
            ces.append(answer_ce_d(model, rec["ids"], rec["mask_start"], depth))
        key = f"{tag}:d{depth}" if depth is not None else tag
        out["teacher_forced_ce"][key] = {
            "mean": round(sum(ces) / len(ces), 4),
            "n": len(ces), "sec": round(time.perf_counter() - t0, 1),
            "_per_item": ces,
        }
        _p = ROOT / "results" / f"s4_tier1_{args.run}_partial.json"
        _p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                      encoding="utf-8")
        print(f"  [A] CE {key}: {out['teacher_forced_ce'][key]['mean']} "
              f"(n={len(ces)}, {out['teacher_forced_ce'][key]['sec']}s)", flush=True)

    def sweep_gen(tag, depth, model_gen):
        recs = []
        k = 0
        t0 = time.perf_counter()
        cap = args.ctx - args.gen_tokens - 8

        def build(n_shots):
            shots = ""
            for e in ce_items[:n_shots]:
                shots += f"Q: {e['q']}\nA: Let's think step by step. {e['a']}\n\n"
            return shots

        for it in gen_items:
            tail = f"Q: {it['q']}\nA: Let's think step by step."
            n_shots = 2
            prompt = build(n_shots) + tail
            while n_shots > 0 and len(tok.encode(prompt)) > cap:
                n_shots -= 1
                prompt = build(n_shots) + tail
            p_ids = tok.encode(prompt)[:cap]
            text, finished, n_tok = generate(model_gen, tok, p_ids,
                                             depth, args.gen_tokens)
            pred = extract_gsm8k_strict(text, finished)
            hit = (pred is not None and pred == it["gold"])
            k += int(hit)
            recs.append({"q": it["q"], "gold": it["gold"],
                         "raw": text[:600], "pred": pred, "n_shots": n_shots,
                         "has_hash": "####" in text, "finished": finished,
                         "n_tokens": n_tok, "strict_correct": hit})
            print(f"  [B] {tag} item{len(recs)}: pred={pred} gold={it['gold']} "
                  f"{'HIT' if hit else 'miss'} (hash={recs[-1]['has_hash']} "
                  f"eos={finished} {n_tok}tok shots={n_shots})", flush=True)
        out["free_running"][tag] = {
            "depth": depth, "strict_acc": round(k / max(1, len(recs)), 4),
            "k": k, "n": len(recs), "wilson95": wilson(k, len(recs)),
            "sec": round(time.perf_counter() - t0, 1), "items": recs,
        }
        _p = ROOT / "results" / f"s4_tier1_{args.run}_partial.json"
        _p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                      encoding="utf-8")
        print(f"  [B] {tag}: strict {k}/{len(recs)} "
              f"wilson={out['free_running'][tag]['wilson95']}", flush=True)

    print("== trained (ckpt gate) ==", flush=True)
    for d in args.depths:
        sweep_ce("trained", d)
    sweep_gen("trained:d2", 2, model)
    sweep_gen("trained:d4", 4, model)

    print("== frozen (fresh gate) ==", flush=True)
    torch.manual_seed(args.seed + 1)
    model = LoopUSQwen35(backbone)
    model.eval()
    for d in args.depths:
        sweep_ce("frozen", d)
    sweep_gen("frozen:d4", 4, model)

    print("== base (original forward) ==", flush=True)
    sweep_ce("base", None)
    sweep_gen("base:d0", None, model)

    deltas = {}
    for d in args.depths:
        t = out["teacher_forced_ce"][f"trained:d{d}"]["_per_item"]
        fr = out["teacher_forced_ce"][f"frozen:d{d}"]["_per_item"]
        deltas[f"trained-frozen:d{d}"] = paired_ci([a - b for a, b in zip(t, fr)])
    t1 = out["teacher_forced_ce"]["trained:d1"]["_per_item"]
    for d in args.depths:
        td = out["teacher_forced_ce"][f"trained:d{d}"]["_per_item"]
        deltas[f"trained:d{d}-d1"] = paired_ci([a - b for a, b in zip(td, t1)])
    base = out["teacher_forced_ce"]["base"]["_per_item"]
    for d in args.depths:
        td = out["teacher_forced_ce"][f"trained:d{d}"]["_per_item"]
        deltas[f"trained:d{d}-base"] = paired_ci([a - b for a, b in zip(td, base)])
    out["paired_deltas_answer_ce"] = deltas
    for kk, vv in deltas.items():
        print(f"  [delta] {kk}: {vv['mean']} {vv['ci95']}", flush=True)

    p = ROOT / "results" / f"s4_tier1_{args.run}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written to {p}", flush=True)


if __name__ == "__main__":
    main()
