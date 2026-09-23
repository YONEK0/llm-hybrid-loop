"""S1 / C1 pre-experiment: frozen-weight depth-recurrence trace measurement.

Measures, WITHOUT any training, whether looping the reasoning block of a frozen
model contracts or diverges:
  - ||h^(b+1) - h^(b)|| per loop iteration (hidden-state drift)
  - decoder CE per loop iteration (does deeper looping hurt next-token loss?)

Models:
  A. Qwen3.5-4B-Base text backbone (hybrid 3:1: 24 linear_attention + 8 full),
     reasoning block = layers 16..23 (modules 4-5, per draft).
  B. Qwen3-1.7B (standard attention, 28 layers), reasoning block = layers
     10..17 (middle 8 layers, size-matched control). Already on disk via HF cache.

Verdict G1: distances contract / CE flat-or-down -> H1 holds, run A4.
Distances diverge / CE explodes -> H1 doubtful, drop A4.

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s1_trace.py
"""

import json
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX = 512
N_TEXTS = 8
N_LOOPS = 20
OUT = "results/s1_trace.json"


def get_layers(model):
    mm = getattr(model, "model", model)
    return mm.layers


def get_embed(model):
    mm = getattr(model, "model", model)
    return mm.get_input_embeddings()


def get_rotary(model):
    mm = getattr(model, "model", model)
    return getattr(mm, "rotary_emb", None)


def get_final_norm(model):
    mm = getattr(model, "model", model)
    return mm.norm


def layer_forward(layer, h, pos_ids, cache_pos, pe):
    """Call a decoder layer across transformers 5.x signature variants."""
    kwargs = dict(position_ids=pos_ids, cache_position=cache_pos)
    if pe is not None:
        kwargs["position_embeddings"] = pe
    try:
        out = layer(h, attention_mask=None, use_cache=False, **kwargs)
    except TypeError:
        out = layer(h, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.no_grad()
def trace_model(model, tok, layer_span, texts):
    layers = get_layers(model)
    embed = get_embed(model)
    rotary = get_rotary(model)
    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()
    lo, hi = layer_span

    dists, ces = [], []
    for text in texts:
        ids = tok.encode(text)[:CTX]
        if len(ids) < CTX // 2:
            continue
        x = torch.tensor([ids], device="cuda")
        labels = x[:, 1:]
        pos_ids = torch.arange(x.shape[1], device="cuda").unsqueeze(0)
        cache_pos = pos_ids[0]
        pe = rotary(x, pos_ids) if rotary is not None else None

        h = embed(x)
        for layer in layers[:lo]:
            h = layer_forward(layer, h, pos_ids, cache_pos, pe)
        h0 = h

        prev = h0
        for b in range(N_LOOPS):
            h = prev
            for layer in layers[lo:hi]:
                h = layer_forward(layer, h, pos_ids, cache_pos, pe)
            logits = lm_head(final_norm(h))[:, :-1].float()
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1)).item()
            d = (h - prev).float().norm(dim=-1).mean().item()
            dists.append(round(d, 4))
            ces.append(round(ce, 4))
            prev = h
    return {"dist_per_iter": dists, "ce_per_iter": ces}


def summarize(trace):
    d = trace["dist_per_iter"]
    c = trace["ce_per_iter"]
    n = len(d) - 1
    ratio_last_first = d[-1] / max(d[0], 1e-9)
    return {
        "dist_first3_mean": round(sum(d[:3]) / 3, 4),
        "dist_last3_mean": round(sum(d[-3:]) / 3, 4),
        "dist_last_over_first": round(ratio_last_first, 4),
        "ce_first3_mean": round(sum(c[:3]) / 3, 4),
        "ce_last3_mean": round(sum(c[-3:]) / 3, 4),
        "ce_min": round(min(c), 4),
        "ce_iter0": round(c[0], 4),
        "n_windows": len(d),
    }


def main():
    t0 = time.perf_counter()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text_all = "\n\n".join(t for t in ds["text"] if t.strip())
    step = max(len(text_all) // N_TEXTS, 1)
    texts = [text_all[i * step:(i + 1) * step] for i in range(N_TEXTS)]

    out = {"ctx": CTX, "n_loops": N_LOOPS, "models": {}}

    print("=== A. Qwen3.5-4B-Base (hybrid 3:1), reasoning=layers16-23 ===", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B-Base", dtype=torch.bfloat16).cuda().eval()
    # AutoModelForCausalLM may return multimodal wrapper; take text backbone
    if not hasattr(model, "get_input_embeddings") or type(model).__name__ != "Qwen3_5TextForCausalLM":
        lm = getattr(model, "language_model", None) or getattr(model, "model", model)
        if hasattr(lm, "lm_head"):
            model = lm
    print("class:", type(model).__name__, flush=True)
    tr = trace_model(model, tok, (16, 24), texts)
    out["models"]["qwen3.5-4b_hybrid_l16_23"] = {"trace_summary": summarize(tr)}
    print(json.dumps(summarize(tr), indent=1), flush=True)
    out["models"]["qwen3.5-4b_hybrid_l16_23"]["raw"] = tr
    del model
    torch.cuda.empty_cache()

    print("=== B. Qwen3-1.7B (standard attention), reasoning=layers10-17 ===", flush=True)
    tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()
    tr2 = trace_model(model2, tok2, (10, 18), texts)
    out["models"]["qwen3-1.7b_std_l10_17"] = {"trace_summary": summarize(tr2)}
    print(json.dumps(summarize(tr2), indent=1), flush=True)
    out["models"]["qwen3-1.7b_std_l10_17"]["raw"] = tr2

    path = OUT
    open(path, "w", encoding="utf-8").write(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"written to {path} [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
