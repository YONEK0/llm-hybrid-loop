"""S1a: per-layer representation dynamics of frozen Qwen3.5-4B-Base (hybrid 3:1)
with Qwen3-1.7B (standard attention) as control.  Answers three block-boundary
criteria questions (TRAINING_PLAN_V4_HYBRID.md, decision D6):

  P1  periodicity & phase-lock: is the consecutive-layer cosine-distance curve
      periodic with period ~4, peaks locked to full-attention layers, stable
      across depth?  (decides whether module-aligned cuts are phase-consistent)
  P2  (user Q1) are DeltaNet->DeltaNet transitions really smaller than
      attention-involving transitions?  (mechanism basis for differentiated
      gating)  -- Mann-Whitney U, middle layers only + full-range reference
  P3  (user Q2) how does the attention-peak amplitude envelope relate to the
      macro stage boundaries (logit-lens CE inflection points)?
      (decides which peaks the encoder/reasoning/decoder cuts snap to)

Method: native forward with output_hidden_states=True (model-correct positions
and masks), per-layer cosine distance and logit-lens CE (lm_head(final_norm)).

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s1a_dynamics.py
"""

import json
import math
import time
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX = 512
N_TEXTS = 8
OUT = "results/s1a_dynamics.json"


def text_backbone(model):
    """Return (layers, final_norm, lm_head) for plain or multimodal wrappers."""
    cands = []
    mm = getattr(model, "model", None)
    if mm is not None:
        cands += [getattr(mm, "language_model", None), mm]
    cands += [getattr(model, "language_model", None), model]
    for c in cands:
        if c is not None and hasattr(c, "layers"):
            norm = getattr(c, "norm", None)
            head = model.get_output_embeddings()
            return c.layers, norm, head
    raise RuntimeError("cannot locate text backbone layers")


@torch.no_grad()
def layer_curves(model, tok, texts):
    """Per-layer cosine distance + logit-lens CE, averaged over texts."""
    layers, final_norm, lm_head = text_backbone(model)
    n_layers = len(layers)
    dist_mean = [0.0] * n_layers          # transition i: h_i -> h_{i+1}
    dist_last = [0.0] * n_layers
    lens_ce = [0.0] * (n_layers + 1)      # logit-lens CE from h_i (incl. embeddings)
    n_texts = 0
    for text in texts:
        ids = tok.encode(text)[:CTX]
        if len(ids) < CTX:
            continue
        x = torch.tensor([ids], device="cuda")
        labels = x[:, 1:]
        hs = model(input_ids=x, output_hidden_states=True,
                   use_cache=False).hidden_states
        assert len(hs) == n_layers + 1, (len(hs), n_layers)
        for i in range(n_layers):
            a = hs[i][0].float()
            b = hs[i + 1][0].float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            dist_mean[i] += (1 - cos).mean().item()
            dist_last[i] += (1 - cos[-1]).mean().item()
        for i, h in enumerate(hs):
            lg = lm_head(final_norm(h))[:, :-1].float()
            lens_ce[i] += torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), labels.reshape(-1)).item()
        n_texts += 1
    dist_mean = [round(d / n_texts, 5) for d in dist_mean]
    dist_last = [round(d / n_texts, 5) for d in dist_last]
    lens_ce = [round(c / n_texts, 5) for c in lens_ce]
    return {"dist_mean": dist_mean, "dist_last": dist_last,
            "logitlens_ce": lens_ce, "n_texts": n_texts,
            "layer_types": ["linear" if getattr(l, "layer_type", "") == "linear_attention"
                            else "attn" if i % 4 == 3 else "linear"
                            for i, l in enumerate(layers)]}


def mann_whitney_u(a, b):
    """Mann-Whitney U with normal approximation and tie correction."""
    n1, n2 = len(a), len(b)
    allv = a + b
    order = sorted(range(len(allv)), key=lambda i: allv[i])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    R1 = sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = sum(t ** 3 - t for t in Counter(allv).values())
    sigma = math.sqrt(max(n1 * n2 / 12 * ((len(allv) + 1) - tie /
                   (len(allv) * (len(allv) - 1 + 1e-9))), 1e-9))
    z = (U1 - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"U": round(U1, 1), "z": round(z, 3), "p": round(p, 5),
            "mean_a": round(sum(a) / n1, 5), "mean_b": round(sum(b) / n2, 5)}


def analyze_p1(curve):
    """Peaks, peak spacing, autocorrelation, phase-lock to attention layers."""
    d = curve["dist_mean"]
    peaks = [i for i in range(1, len(d) - 1) if d[i] >= d[i - 1] and d[i] >= d[i + 1]
             and d[i] > sum(d) / len(d)]
    spacings = [b - a for a, b in zip(peaks, peaks[1:])]
    # lag-4 dominance: autocorrelation of (d - mean) at lags 2..8
    mu = sum(d) / len(d)
    var = sum((v - mu) ** 2 for v in d)
    ac = {}
    for lag in range(2, 9):
        num = sum((d[i] - mu) * (d[i + lag] - mu) for i in range(len(d) - lag))
        ac[lag] = round(num / var, 3) if var > 0 else 0.0
    attn_pos = [i for i in range(len(d)) if i % 4 == 3 or i % 4 == 2]
    locked = [p for p in peaks if p % 4 in (2, 3)]
    return {"peaks": peaks, "peak_spacings": spacings,
            "autocorr_lag2_8": ac,
            "lag4_autocorr": ac.get(4),
            "peaks_at_attn_transition": locked,
            "phase_lock_rate": round(len(locked) / max(len(peaks), 1), 3)}


def analyze_p2(curve, lo=4, hi=28):
    """Group transitions by layer-type pair (middle range by default)."""
    d = curve["dist_mean"]
    groups = {"linear->linear": [], "linear->attn": [], "attn->linear": []}
    for i in range(lo, min(hi, len(d))):
        if i % 4 in (0, 1):
            groups["linear->linear"].append(d[i])
        elif i % 4 == 2:
            groups["linear->attn"].append(d[i])
        else:
            groups["attn->linear"].append(d[i])
    ll = groups["linear->linear"]
    att = groups["linear->attn"] + groups["attn->linear"]
    return {"groups": {k: {"n": len(v), "mean": round(sum(v) / max(len(v), 1), 5),
                           "values": v} for k, v in groups.items()},
            "U_test_linear_vs_attn_involving": mann_whitney_u(ll, att)}


def analyze_p3(curve):
    """Attention-peak envelope vs macro boundaries (logit-lens CE inflections)."""
    d = curve["dist_mean"]
    ce = curve["logitlens_ce"]
    attn_peaks = [(i, d[i]) for i in range(2, len(d) - 1)
                  if i % 4 == 2 and d[i] >= d[i - 1] and d[i] > d[i + 1]]
    # macro boundaries: top-2 |second difference| of logit-lens CE
    d2 = [abs(ce[i + 1] - 2 * ce[i] + ce[i - 1]) for i in range(1, len(ce) - 1)]
    ranked = sorted(range(len(d2)), key=lambda i: -d2[i])[:4]
    macro = sorted(i + 1 for i in ranked)          # layer index after transition
    attn_sorted = sorted(attn_peaks, key=lambda t: -t[1])
    return {"attn_peaks_by_layer": attn_peaks,
            "attn_peak_envelope": [round(v, 5) for _, v in attn_peaks],
            "macro_boundary_candidates_top4": macro,
            "largest_attn_peaks": [(i, round(v, 5)) for i, v in attn_sorted[:4]],
            "note": "compare macro candidates with attn peak positions"}


def main():
    t0 = time.perf_counter()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text_all = "\n\n".join(t for t in ds["text"] if t.strip())
    step = max(len(text_all) // N_TEXTS, 1)
    texts = [text_all[i * step:(i + 1) * step] for i in range(N_TEXTS)]

    out = {"ctx": CTX, "n_texts": N_TEXTS, "models": {}}

    print("=== A. Qwen3.5-4B-Base (hybrid 3:1) ===", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B-Base", dtype=torch.bfloat16).cuda().eval()
    curves = layer_curves(model, tok, texts)
    p1 = analyze_p1(curves)
    p2 = analyze_p2(curves)
    p3 = analyze_p3(curves)
    out["models"]["qwen3.5-4b_hybrid"] = {"curves": curves,
                                          "P1_periodicity": p1,
                                          "P2_transition_groups": p2,
                                          "P3_peaks_vs_macro": p3}
    print("P1:", json.dumps(p1), flush=True)
    print("P2:", json.dumps({k: v for k, v in p2.items() if k != "groups"}), flush=True)
    print("P2 groups:", json.dumps({k: v["mean"] for k, v in p2["groups"].items()}), flush=True)
    print("P3:", json.dumps(p3), flush=True)
    del model
    torch.cuda.empty_cache()

    print("=== B. Qwen3-1.7B (standard attention, control) ===", flush=True)
    tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()
    curves2 = layer_curves(model2, tok2, texts)
    out["models"]["qwen3-1.7b_std"] = {"curves": curves2}
    print("logitlens_ce:", curves2["logitlens_ce"], flush=True)
    del model2
    torch.cuda.empty_cache()

    open(OUT, "w", encoding="utf-8").write(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"written to {OUT} [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
