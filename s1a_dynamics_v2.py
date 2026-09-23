"""S1a-v2: corrected per-layer representation dynamics (V5 spec).

Replaces provisional results/s1a_dynamics.json (v1 had: double final-norm on the last
hidden state, layer-type groups assigned by transition index i%4 instead of the real
(type[i-1], type[i]) pair, control model hardcoded to the hybrid's 3:1 pattern, and
significance tests run on text-averaged points).

Conventions (fixed once, stated in the output JSON):
  h_0      = embeddings output (raw)
  h_{i+1}  = raw output of decoder layer i   (captured with forward hooks, model-correct
             masks; output_hidden_states semantics are not relied upon)
  t_i      = transition h_i -> h_{i+1}, i = 0..L-1; produced by layer i (t_0 = embed->L0)
  type(t_i) for i>=1 = (type of layer i-1, type of layer i), read from layer modules
  distance = 1 - cosine(h_i, h_{i+1}); computed on RAW states, final norm never applied
  logit-lens stage s (s=0..L): CE(lm_head(final_norm(h_s))) vs next tokens — final norm
             applied exactly once per stage, uniformly, never on top of model logits
  text     = one WikiText-2 chunk (ctx 512); the bootstrap cluster unit for all statistics

Outputs: results/s1a_dynamics_v2.json (per-text curves + P1/P2/P3 + piecewise fits).
Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s1a_dynamics_v2.py
"""

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX = 512
N_TEXTS = 8
BOOT = 200          # bootstrap resamples for P3 breakpoint distribution
BOOT_CI = 10000     # bootstrap resamples for P2 confidence intervals
SEED = 2026
OUT = Path("results/s1a_dynamics_v2.json")


# ---------------------------------------------------------------- utilities
def text_backbone(model):
    cands = []
    mm = getattr(model, "model", None)
    if mm is not None:
        cands += [getattr(mm, "language_model", None), mm]
    cands += [getattr(model, "language_model", None), model]
    for c in cands:
        if c is not None and hasattr(c, "layers"):
            return c.layers, c.norm, model.get_output_embeddings()
    raise RuntimeError("cannot locate text backbone layers")


def unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def mean_curve(per_text, key):
    n = len(per_text)
    return [round(sum(t[key][i] for t in per_text) / n, 6)
            for i in range(len(per_text[0][key]))]


def bootstrap_ci(values, iters=BOOT_CI, seed=7):
    rnd = random.Random(seed)
    n = len(values)
    stats = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n
                   for _ in range(iters))
    lo, hi = stats[int(0.025 * iters)], stats[int(0.975 * iters)]
    return {"mean": round(sum(values) / n, 6),
            "ci95": [round(lo, 6), round(hi, 6)]}


def sign_test_p(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"pos": pos, "neg": neg, "p_two_sided": round(min(1.0, p), 5)}


def ols_rss(ys, lo, hi):
    n = hi - lo
    if n < 2:
        return None
    xs = range(lo, hi)
    sx, sy = sum(xs), sum(ys[lo:hi])
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys[lo:hi]))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    inter = (sy - slope * sx) / n
    return sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys[lo:hi]))


def piecewise(ys, nseg):
    """Disjoint-segment OLS piecewise fit; breakpoints are stage indices where a new
    segment starts. Returns rss, breakpoints, BIC (k = 2 params per segment)."""
    N = len(ys)
    best = None
    if nseg == 1:
        r = ols_rss(ys, 0, N)
        best = (r, []) if r is not None else None
    elif nseg == 2:
        cands = []
        for b in range(2, N - 1):
            r1, r2 = ols_rss(ys, 0, b), ols_rss(ys, b, N)
            if r1 is not None and r2 is not None:
                cands.append((r1 + r2, [b]))
        best = min(cands) if cands else None
    else:
        cands = []
        for b1 in range(2, N - 3):
            for b2 in range(b1 + 2, N - 1):
                rs = [ols_rss(ys, lo, hi)
                      for lo, hi in ((0, b1), (b1, b2), (b2, N))]
                if all(r is not None for r in rs):
                    cands.append((sum(rs), [b1, b2]))
        best = min(cands) if cands else None
    if best is None or best[0] <= 0:
        return None
    rss, breaks = best
    return {"rss": round(rss, 5), "breaks_stage": breaks,
            "bic": round(N * math.log(rss / N) + 2 * nseg * math.log(N), 3)}


# ---------------------------------------------------------------- measurement
@torch.no_grad()
def curves_for_model(model, tok, texts):
    layers, final_norm, lm_head = text_backbone(model)
    L = len(layers)
    cap = {}
    handles = [l.register_forward_hook(
        (lambda idx: (lambda mod, args, out: cap.__setitem__(idx, unwrap(out).detach())))(i))
        for i, l in enumerate(layers)]

    per_text = []
    for text in texts:
        ids = tok.encode(text)[:CTX]
        if len(ids) < CTX:
            continue
        x = torch.tensor([ids], device="cuda")
        labels = x[:, 1:]
        cap.clear()
        model(input_ids=x, use_cache=False)
        h = [None] * (L + 1)
        h[0] = unwrap(model.get_input_embeddings()(x)).detach()
        for i in range(L):
            h[i + 1] = cap[i]
        dist_mean, dist_last = [0.0] * L, [0.0] * L
        for i in range(L):
            a, b = h[i][0].float(), h[i + 1][0].float()
            cos = F.cosine_similarity(a, b, dim=-1)
            dist_mean[i] = round(float((1 - cos).mean().item()), 6)
            dist_last[i] = round(float((1 - cos[-1]).item()), 6)
        lens_ce = [0.0] * (L + 1)
        for s in range(L + 1):
            lg = lm_head(final_norm(h[s]))[:, :-1].float()
            lens_ce[s] = round(float(F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), labels.reshape(-1)).item()), 5)
        per_text.append({"dist_mean": dist_mean, "dist_last": dist_last,
                         "lens_ce": lens_ce})
        del h
    for hd in handles:
        hd.remove()
    return per_text


# ---------------------------------------------------------------- analyses
def analyze_p1(per_text, layer_types, attn_pos):
    d = mean_curve(per_text, "dist_mean")
    mu = sum(d) / len(d)
    peaks = [i for i in range(1, len(d) - 1)
             if d[i] >= d[i - 1] and d[i] >= d[i + 1] and d[i] > mu]
    var = sum((v - mu) ** 2 for v in d)
    ac = {}
    for lag in range(2, 9):
        num = sum((d[i] - mu) * (d[i + lag] - mu) for i in range(len(d) - lag))
        ac[lag] = round(num / var, 3) if var else 0.0
    per_text_lock = []
    for t in per_text:
        dt = t["dist_mean"]
        m = sum(dt) / len(dt)
        tp = [i for i in range(1, len(dt) - 1)
              if dt[i] >= dt[i - 1] and dt[i] >= dt[i + 1] and dt[i] > m]
        per_text_lock.append(round(
            len([p for p in tp if p in attn_pos]) / max(len(tp), 1), 3))
    return {"peaks_produced_by_layer": peaks,
            "peak_spacings": [b - a for a, b in zip(peaks, peaks[1:])],
            "autocorr_lag2_8": ac,
            "attn_produced_transitions": attn_pos,
            "phase_lock_rate_mean_curve": round(
                len([p for p in peaks if p in attn_pos]) / max(len(peaks), 1), 3),
            "per_text_lock_rates": per_text_lock}


def analyze_p2(per_text, layer_types):
    L = len(layer_types)
    per_group = {}
    for t in per_text:
        buckets = {}
        for i in range(1, L):
            pair = f"{layer_types[i - 1]}->{layer_types[i]}"
            buckets.setdefault(pair, []).append(t["dist_mean"][i])
        for pair, v in buckets.items():
            per_group.setdefault(pair, []).append(round(sum(v) / len(v), 6))
    res = {"groups_per_text": per_group,
           "groups_overall": {k: bootstrap_ci(v) for k, v in per_group.items()}}
    ll = per_group.get("linear_attention->linear_attention")
    lf = per_group.get("linear_attention->full_attention")
    fl = per_group.get("full_attention->linear_attention")
    if ll and lf and fl:
        pooled = [(a + b) / 2 for a, b in zip(lf, fl)]
        eff = [a - p for a, p in zip(ll, pooled)]
        res["linear_vs_attn_involving"] = {
            "linear_to_linear": bootstrap_ci(ll),
            "attn_involving_pooled": bootstrap_ci(pooled),
            "per_text_diff": [round(e, 6) for e in eff],
            "diff_ci95": bootstrap_ci(eff)["ci95"],
            "sign_test": sign_test_p(eff),
            "note": "diff = linear->linear minus pooled attn-involving, "
                    "negative = DeltaNet-chain transitions smaller"}
    return res


def analyze_p3(per_text, layer_types):
    texts_ce = [t["lens_ce"] for t in per_text]
    ce = mean_curve(per_text, "lens_ce")
    fits = {f"{k}_seg": piecewise(ce, k) for k in (1, 2, 3)}
    best_fit_key = min((v["bic"], k) for k, v in fits.items() if v)[1]

    # bootstrap breakpoint distribution over texts
    rnd = random.Random(SEED)
    n = len(texts_ce)
    break_dist, k_dist = Counter(), Counter()
    for _ in range(BOOT):
        sample = [texts_ce[rnd.randrange(n)] for _ in range(n)]
        curve = [round(sum(s[i] for s in sample) / n, 6)
                 for i in range(len(texts_ce[0]))]
        fk = {k: piecewise(curve, k) for k in (1, 2, 3)}
        bk = min((v["bic"], k) for k, v in fk.items() if v)[1]
        k_dist[bk] += 1
        for b in fk[bk]["breaks_stage"]:
            break_dist[b] += 1
    # a breakpoint at stage b sits at transition t_{b-1} (produced by layer b-1)
    d = mean_curve(per_text, "dist_mean")
    attn_trans = sorted(i for i, ty in enumerate(layer_types) if ty == "full_attention")
    return {"mean_lens_ce": ce,
            "fits": fits,
            "best_fit": best_fit_key,
            "best_breaks_stage": fits[best_fit_key]["breaks_stage"],
            "best_breaks_transition": [b - 1 for b in fits[best_fit_key]["breaks_stage"]],
            "bootstrap_best_k": dict(k_dist),
            "bootstrap_breakpoint_top": break_dist.most_common(6),
            "attn_produced_transitions": attn_trans,
            "attn_transition_envelope": [[i, d[i]] for i in attn_trans],
            "note": "break stage b = boundary between stage b-1 and b, i.e. at "
                    "transition t_{b-1} (produced by layer b-1)"}


# ---------------------------------------------------------------- driver
def run_model(tag, model_id):
    print(f"=== {tag} ({model_id}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16).cuda().eval()
    layers, final_norm, lm_head = text_backbone(model)
    layer_types = [getattr(l, "layer_type", "full_attention") for l in layers]
    cfg_types = list(getattr(model.config, "layer_types", []) or [])

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    all_text = "\n\n".join(t for t in ds["text"] if t.strip())
    step = max(len(all_text) // N_TEXTS, 1)
    texts = [all_text[i * step:(i + 1) * step] for i in range(N_TEXTS)]

    t0 = time.perf_counter()
    per_text = curves_for_model(model, tok, texts)
    print(f"curves done in {time.perf_counter()-t0:.0f}s "
          f"(n_texts={len(per_text)})", flush=True)
    attn_pos = [i for i, ty in enumerate(layer_types) if ty == "full_attention"]

    out = {"n_layers": len(layers),
           "layer_types_modules": layer_types,
           "layer_types_config": cfg_types,
           "config_matches_modules": bool(cfg_types) and cfg_types == layer_types,
           "n_texts": len(per_text),
           "mean": {"dist_mean": mean_curve(per_text, "dist_mean"),
                    "dist_last": mean_curve(per_text, "dist_last"),
                    "lens_ce": mean_curve(per_text, "lens_ce")},
           "texts": per_text,
           "P1_periodicity": analyze_p1(per_text, layer_types, attn_pos),
           "P2_transition_groups": analyze_p2(per_text, layer_types),
           "P3_peaks_vs_macro": analyze_p3(per_text, layer_types)}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    out = {"spec": {
        "ctx": CTX, "n_texts": N_TEXTS, "dataset": "wikitext-2-raw-v1/validation",
        "transition": "t_i = h_i -> h_{i+1}, produced by layer i (t_0 = embed->L0)",
        "pair_types": "(type[i-1], type[i]) from layer modules",
        "distance": "1-cosine on raw states; final norm never applied to distances",
        "logitlens": "final_norm applied exactly once per stage, uniformly",
        "stats": "text = bootstrap cluster; no tests on text-averaged points",
        "p3": "1/2/3-segment disjoint OLS fits, BIC selection, bootstrap breakpoints"},
        "models": {}}

    out["models"]["qwen3.5-4b_hybrid"] = run_model("A. hybrid", "Qwen/Qwen3.5-4B-Base")
    out["models"]["qwen3-1.7b_std"] = run_model("B. control", "Qwen/Qwen3-1.7B")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written to {OUT}")

    for tag, m in out["models"].items():
        p1, p2, p3 = m["P1_periodicity"], m["P2_transition_groups"], m["P3_peaks_vs_macro"]
        print(f"\n--- {tag} ---")
        print("P1:", json.dumps({k: p1[k] for k in
              ("peaks_produced_by_layer", "peak_spacings", "autocorr_lag2_8",
               "phase_lock_rate_mean_curve", "per_text_lock_rates")}))
        print("P2 groups:", json.dumps(p2["groups_overall"]))
        if "linear_vs_attn_involving" in p2:
            print("P2 test:", json.dumps(p2["linear_vs_attn_involving"]))
        print("P3:", json.dumps({k: p3[k] for k in
              ("best_fit", "best_breaks_stage", "best_breaks_transition",
               "bootstrap_best_k", "bootstrap_breakpoint_top",
               "attn_transition_envelope")}))


if __name__ == "__main__":
    main()
