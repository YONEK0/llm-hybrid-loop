"""S1b-v2: frozen-weight loop trace for S1a-v2 block candidates (V5 spec).

For each candidate block M = layers[lo:hi]:
  h = E(x);  ce[0] = CE(D(h))            (block skipped)
  for b in 1..N_LOOPS:
      h_new = M(h);  ce[b] = CE(D(h_new))
      drift[b-1] = ||h_new - h|| (L2 per token, mean) and 1-cos
Depth-1 CE (E+M+D once = the untouched original network) doubles as the sanity anchor:
manual depth-1 CE must match a full model(...) forward.

Semantics: attention_mask=None (LoopUS's own convention for unpadded batches — the
_build_attention_mask_mapping helper returns None masks and layers apply causality
internally); use_cache=False everywhere, so full-attention KV and linear conv/recurrent
states are rebuilt from scratch inside every layer call — no cross-iteration state leaks
by construction. Only the designed hidden state passes between loop iterations.

Precision: hybrid runs on the NF4 spike model (fits VRAM; bf16 spills to shared memory
and is ~9x slower — measured in S0.4/S1a). A bf16 spot check (2 texts x 4 loops on the
primary candidate) quantifies the quantization offset on the CE curve.

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s1b_trace_v2.py [hybrid] [control] [spot]
Stages default to all three; results are written incrementally to results/s1b_trace_v2.json.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
CTX, N_TEXTS, N_LOOPS = 512, 8, 20
HY_MODEL, CT_MODEL = "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3-1.7B"
HY_CANDS, CT_CANDS = [(8, 20), (16, 24), (4, 20), (9, 21)], [(10, 18)]
# slice bounds are hi-exclusive: (8,20) = layers 8..19 ("L8-19", 3 modules), etc.
OUT = ROOT / "results/s1b_trace_v2.json"

SIG = {}  # id(layer) -> kwargs profile that worked (avoids repeated TypeErrors)


def text_backbone(model):
    cands = []
    mm = getattr(model, "model", None)
    if mm is not None:
        cands += [getattr(mm, "language_model", None), mm]
    cands += [getattr(model, "language_model", None), model]
    for c in cands:
        if c is not None and hasattr(c, "layers"):
            return (c.layers, getattr(c, "norm", None),
                    model.get_output_embeddings(), getattr(c, "rotary_emb", None))
    raise RuntimeError("cannot locate text backbone layers")


def unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def call_layer(layer, h, pe, pos_ids, cache_pos):
    profile = SIG.get(id(layer))
    attempts = []
    if profile == "full":
        attempts = [dict(attention_mask=None, position_ids=pos_ids,
                         position_embeddings=pe, cache_position=cache_pos)]
    elif profile == "mid":
        attempts = [dict(attention_mask=None, position_ids=pos_ids,
                         cache_position=cache_pos)]
    elif profile == "min":
        attempts = [dict(attention_mask=None)]
    else:
        attempts = [dict(attention_mask=None, position_ids=pos_ids,
                         position_embeddings=pe, cache_position=cache_pos),
                    dict(attention_mask=None, position_ids=pos_ids,
                         cache_position=cache_pos),
                    dict(attention_mask=None)]
    last = None
    for kw in attempts:
        try:
            out = layer(h, use_cache=False, **kw)
            SIG[id(layer)] = ("full" if "position_embeddings" in kw else
                              "mid" if "cache_position" in kw else "min")
            return unwrap(out)
        except TypeError as e:
            last = e
    raise last


@torch.no_grad()
def trace_candidate(model, tok, texts, lo, hi, n_loops=N_LOOPS):
    layers, final_norm, lm_head, rotary = text_backbone(model)
    L = len(layers)
    assert 0 < lo < hi <= L
    E, M, D = layers[:lo], layers[lo:hi], layers[hi:]
    types = [getattr(l, "layer_type", "full_attention") for l in layers]
    meta = {
        "layers": [lo, hi - 1], "n_layers_block": hi - lo,
        "seam_pair": f"{types[hi - 1]}->{types[lo]}",
        "original_input_pair": (f"{types[lo - 1]}->{types[lo]}" if lo > 0 else "embed"),
        "seam_period_complete": (hi - lo) % 4 == 0,
    }
    per_text = []
    for text in texts:
        ids = tok.encode(text)[:CTX]
        if len(ids) < CTX:
            continue
        x = torch.tensor([ids], device="cuda")
        labels = x[:, 1:]
        pos_ids = torch.arange(x.shape[1], device="cuda").unsqueeze(0)
        cache_pos = pos_ids[0]
        h0 = model.get_input_embeddings()(x)
        pe = rotary(h0, pos_ids) if rotary is not None else None

        def ce_of(h):
            lg = lm_head(final_norm(h))[:, :-1].float()
            return round(float(F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), labels.reshape(-1)).item()), 5)

        def full_ce(h):
            # decoder tail MUST run: CE is computed on E -> M(loop) -> D output,
            # never on the raw looped state (that was the v1 s1_trace bug)
            hd = h
            for l in D:
                hd = call_layer(l, hd, pe, pos_ids, cache_pos)
            return ce_of(hd)

        h = h0
        for l in E:
            h = call_layer(l, h, pe, pos_ids, cache_pos)
        ce = [full_ce(h)]
        drift_l2, drift_cos = [], []
        prev = h
        for _ in range(n_loops):
            for l in M:
                h = call_layer(l, h, pe, pos_ids, cache_pos)
            ce.append(full_ce(h))
            a, b = prev[0].float(), h[0].float()
            drift_l2.append(round(float((b - a).norm(dim=-1).mean().item()), 4))
            drift_cos.append(round(float((1 - F.cosine_similarity(a, b, dim=-1)
                                          ).mean().item()), 6))
            prev = h
        per_text.append({"ce": ce, "drift_l2": drift_l2, "drift_cos": drift_cos})
    mean = {"ce": [round(sum(t["ce"][i] for t in per_text) / len(per_text), 5)
                   for i in range(n_loops + 1)],
            "drift_l2": [round(sum(t["drift_l2"][i] for t in per_text) / len(per_text), 4)
                         for i in range(n_loops)],
            "drift_cos": [round(sum(t["drift_cos"][i] for t in per_text) / len(per_text), 6)
                          for i in range(n_loops)]}
    return {"meta": meta, "n_texts": len(per_text), "per_text": per_text, "mean": mean}


def load_hybrid_nf4():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(
        HY_MODEL, dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": 0}).eval()


def load(model_id, **kw):
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, **kw).cuda().eval()


def get_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    all_text = "\n\n".join(t for t in ds["text"] if t.strip())
    step = max(len(all_text) // N_TEXTS, 1)
    return [all_text[i * step:(i + 1) * step] for i in range(N_TEXTS)]


def save(data):
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    stages = [a for a in sys.argv[1:] if not a.startswith("-")] or \
        ["hybrid", "control", "spot"]
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "spec": {"ctx": CTX, "n_texts": N_TEXTS, "n_loops": N_LOOPS,
                 "mask": "attention_mask=None (LoopUS convention, causal internal)",
                 "state": "use_cache=False; no cross-iteration cache by construction",
                 "ce_index": "ce[b] = after b loop iterations; ce[0] = block skipped",
                 "precision": "hybrid on NF4 spike model; bf16 spot check included"}}
    texts = get_texts()

    if "hybrid" in stages:
        print("=== hybrid NF4 ===", flush=True)
        tok = AutoTokenizer.from_pretrained(HY_MODEL)
        model = load_hybrid_nf4()
        layers, final_norm, lm_head, rotary = text_backbone(model)
        entry = {"candidates": {}, "sanity": {}}
        # sanity: manual depth-1 CE == full model forward CE (text 0)
        ids = tok.encode(texts[0])[:CTX]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            full = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        ce_full = float(F.cross_entropy(
            full.reshape(-1, full.shape[-1]), x[:, 1:].reshape(-1)).item())
        SIG.clear()
        t0 = time.perf_counter()
        tr = trace_candidate(model, tok, texts[:1], 8, 19, n_loops=1)
        ce_manual = tr["per_text"][0]["ce"][1]
        entry["sanity"] = {"ce_full_forward": round(ce_full, 5),
                           "ce_manual_depth1": ce_manual,
                           "abs_diff": round(abs(ce_full - ce_manual), 5),
                           "verdict": "PASS" if abs(ce_full - ce_manual) < 0.05 else "FAIL"}
        print("sanity:", json.dumps(entry["sanity"]), flush=True)
        for lo, hi in HY_CANDS:
            t1 = time.perf_counter()
            SIG.clear()
            entry["candidates"][f"L{lo}-{hi - 1}"] = trace_candidate(
                model, tok, texts, lo, hi)
            print(f"L{lo}-{hi - 1}: ce[0/1/5/10/20]="
                  f"{[entry['candidates'][f'L{lo}-{hi - 1}']['mean']['ce'][i] for i in (0, 1, 5, 10, 20)]} "
                  f"[{time.perf_counter() - t1:.0f}s]", flush=True)
            data.setdefault("models", {}).setdefault("qwen3.5-4b_hybrid_nf4", entry)
            save(data)
        del model
        torch.cuda.empty_cache()

    if "control" in stages:
        print("=== control bf16 ===", flush=True)
        tok = AutoTokenizer.from_pretrained(CT_MODEL)
        model = load(CT_MODEL, attn_implementation="sdpa")
        entry = {"candidates": {}}
        for lo, hi in CT_CANDS:
            t1 = time.perf_counter()
            SIG.clear()
            entry["candidates"][f"L{lo}-{hi - 1}"] = trace_candidate(
                model, tok, texts, lo, hi)
            print(f"L{lo}-{hi - 1}: ce[0/1/5/10/20]="
                  f"{[entry['candidates'][f'L{lo}-{hi - 1}']['mean']['ce'][i] for i in (0, 1, 5, 10, 20)]} "
                  f"[{time.perf_counter() - t1:.0f}s]", flush=True)
        data.setdefault("models", {})["qwen3-1.7b_std"] = entry
        save(data)
        del model
        torch.cuda.empty_cache()

    if "spot" in stages:
        print("=== bf16 spot check (hybrid, L8-19, 2 texts, 4 loops) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(HY_MODEL)
        model = load(HY_MODEL, attn_implementation="sdpa")
        SIG.clear()
        t0 = time.perf_counter()
        bf16_tr = trace_candidate(model, tok, texts[:2], 8, 19, n_loops=4)
        nf4_entry = data["models"]["qwen3.5-4b_hybrid_nf4"]["candidates"]["L8-19"]
        nf4_2t_ce = [t["ce"] for t in nf4_entry["per_text"][:2]]
        entry = {"ce_bf16": [t["ce"] for t in bf16_tr["per_text"]],
                 "ce_nf4": nf4_2t_ce,
                 "max_abs_diff": round(max(
                     abs(a - b) for tb, tn in zip(bf16_tr["per_text"], nf4_2t_ce)
                     for a, b in zip(tb["ce"], tn)), 4),
                 "seconds": round(time.perf_counter() - t0, 1)}
        data["models"]["qwen3.5-4b_hybrid_nf4"]["bf16_spot_check"] = entry
        save(data)
        print("spot:", json.dumps({k: entry[k] for k in
                                   ("max_abs_diff", "seconds")}), flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
