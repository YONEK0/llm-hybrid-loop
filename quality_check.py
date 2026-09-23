"""quality_check.py — post-pause quality inspection for the V5 L0 checkpoint.

GPU-based (must run while training is paused: the 8 GB card cannot host the trainer
and a second model instance). Loads runs/<run>/ckpt_latest.pt (gate + q_head) onto the
frozen NF4 Qwen3.5-4B-Base backbone and measures mechanism quality — NOT task EM:
this is a BASE model trained on WikiText next-token prediction, so the L0 quality
question is "is the loop mechanism sound and stable", answered by:

  1. depth extrapolation: held-out CE at d = 1,2,4,8,16,32 (16 dev blocks)
     — S1b showed frozen loops exploding to 8.3 by b=20; trained must stay flat
  2. long-generation coherence: 64-token greedy generations at d=1,2,4,8 on 3
     prompts, each scored by the model's own CE at the generating depth
     (quantitative coherence, no external judge)
  3. gate diagnostics: A_bar per-channel mean/std, q_hat stats
  4. strict-scorer sanity: generations are scanned with scorer_strict (a base
     model will mostly not emit '####' statements; that is expected and recorded,
     not scored as failure)

Usage: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe quality_check.py --run v5_l0
Writes results/quality_check_<run>_step<N>.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4, text_backbone
from s2_spike import LoopUSQwen35

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def ce_at_depth(model, x, depth):
    h = model.encode(x)
    for _ in range(depth):
        h = model.gate(model.block(h), h)
    logits = model.decode(h)
    return float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                 x[:, 1:].reshape(-1)).item())


@torch.no_grad()
def generate(model, tok, prompt_ids, depth, n_new):
    ids = list(prompt_ids)
    for _ in range(n_new):
        x = torch.tensor([ids], device="cuda")
        h = model.encode(x)
        for _ in range(depth):
            h = model.gate(model.block(h), h)
        logits = model.decode(h)
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        if t == tok.eos_token_id:
            break
    return ids[len(prompt_ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5_l0")
    ap.add_argument("--dev-blocks", type=int, default=16)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=3)
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run

    ck = torch.load(run_dir / "ckpt_latest.pt", map_location="cuda", weights_only=False)
    step = ck["step"]
    out = {"run": args.run, "step": step, "sup_tokens": ck["sup_tokens"],
           "time": time.strftime("%F %T"), "checks": {}}
    print(f"quality check @ step {step} ({ck['sup_tokens']:.0f} sup tokens)", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    model = LoopUSQwen35(backbone)
    model.gate.load_state_dict(ck["gate"])
    model.q_head.load_state_dict(ck["q_head"])
    model.eval()

    # dev blocks: fresh slices of WikiText validation, disjoint from training text
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids_all = tok.encode(text)
    blocks = []
    for i in range(args.dev_blocks):
        seg = ids_all[i * 600:(i + 1) * 600][:512]
        if len(seg) == 512:
            blocks.append(torch.tensor([seg], device="cuda"))
    out["checks"]["n_dev_blocks"] = len(blocks)

    # ---- 1. depth extrapolation ----
    depths = [1, 2, 4, 8, 16, 32]
    ext = {}
    for d in depths:
        t0 = time.perf_counter()
        ces = [ce_at_depth(model, x, d) for x in blocks]
        ext[f"d{d}"] = round(sum(ces) / len(ces), 4)
        print(f"  CE d={d}: {ext[f'd{d}']} [{time.perf_counter()-t0:.0f}s]", flush=True)
    out["checks"]["depth_extrapolation_ce"] = ext
    base = ext["d1"]
    out["checks"]["extrapolation_verdict"] = (
        "FLAT" if all(ext[f"d{d}"] <= base * 1.15 for d in depths)
        else "DEGRADES")

    # ---- 2. long-generation coherence ----
    gens = []
    for i in range(args.prompts):
        x = blocks[i]
        prompt = x[0].tolist()[:400]
        for d in (1, 2, 4, 8):
            t0 = time.perf_counter()
            new_ids = generate(model, tok, prompt, d, args.gen_tokens)
            gen_text = tok.decode(new_ids, skip_special_tokens=True)
            # self-CE of the generated continuation at the generating depth
            full = torch.tensor([prompt + new_ids], device="cuda")
            ce_gen = ce_at_depth(model, full[:, :512], d) if full.shape[1] >= 512 else None
            gens.append({"prompt": i, "depth": d, "n_tokens": len(new_ids),
                         "text": gen_text[:300], "self_ce": ce_gen,
                         "seconds": round(time.perf_counter() - t0, 1)})
            print(f"  gen p{i} d={d}: {len(new_ids)}tok "
                  f"ce={ce_gen if ce_gen is None else round(ce_gen, 3)} "
                  f"[{gens[-1]['seconds']}s] {gen_text[:80]!r}", flush=True)
    out["checks"]["generations"] = gens

    # ---- 3. gate diagnostics ----
    g = ck["gate"]
    a_log = g["A_log"].float()
    a = -torch.exp(a_log)
    with torch.no_grad():
        # A_bar at a representative state: use one dev block
        h = model.encode(blocks[0])
        h_new = model.block(h)
        p = model.gate.dt_input_proj.weight.dtype
        delta = F.softplus(model.gate.delta_proj(
            model.gate.dt_input_proj(h_new.to(p) - h.to(p))))
        a_bar = torch.exp(delta * (-torch.exp(model.gate.A_log)))
    out["checks"]["gate"] = {
        "A_log_min": round(float(a_log.min()), 3), "A_log_max": round(float(a_log.max()), 3),
        "A_bar_mean": round(float(a_bar.mean()), 5),
        "A_bar_std": round(float(a_bar.std()), 5),
        "A_bar_pct_closed_lt0.1": round(float((a_bar < 0.1).float().mean()), 4),
        "A_bar_pct_open_gt0.9": round(float((a_bar > 0.9).float().mean()), 4)}
    print("  gate:", json.dumps(out["checks"]["gate"]), flush=True)

    # ---- 4. strict-scorer sanity (informational) ----
    from scorer_strict import extract_prosqa_strict
    scored = sum(1 for g_ in gens
                 if extract_prosqa_strict(g_["text"], True) is not None)
    out["checks"]["strict_scorer_note"] = (
        f"{scored}/{len(gens)} generations contain a scorable final statement "
        f"(base model on WikiText: low is expected, recorded for the record)")

    out_p = ROOT / "results" / f"quality_check_{args.run}_step{step}.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": out["checks"]["extrapolation_verdict"],
                      "extrapolation": ext, "gate": out["checks"]["gate"]},
                     ensure_ascii=False), flush=True)
    print(f"written to {out_p}", flush=True)


if __name__ == "__main__":
    main()
