"""Probe which small models can support latent-space reasoning (Coconut/CODI style).

Requirements for the technique:
  1. forward(inputs_embeds=...) accepted
  2. output_hidden_states=True -> last hidden state usable as next input embedding
  3. use_cache=True -> past_key_values usable, and cache can be sliced by position
     (Coconut slices cache to reuse prefix; CODI feeds one latent token at a time)
  4. fits in 8GB VRAM for training (bf16 weights + grads + optimizer + activations)
"""

import os
import sys
import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CANDIDATES = [
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3.5-2B-Base",
]


def probe(model_id):
    out = {"model": model_id}
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        out["arch"] = (cfg.architectures or ["?"])[0]
        out["layers"] = getattr(cfg, "num_hidden_layers", None)
        out["hidden"] = getattr(cfg, "hidden_size", None)
        out["vocab"] = getattr(cfg, "vocab_size", None)
        out["tie"] = getattr(cfg, "tie_word_embeddings", None)
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types:
            from collections import Counter
            out["layer_types"] = dict(Counter(layer_types))
    except Exception as e:
        out["config_error"] = f"{type(e).__name__}: {str(e)[:150]}"
        return out

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        out["tokenizer"] = type(tok).__name__
    except Exception as e:
        out["tokenizer_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out

    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cuda")
        torch.cuda.synchronize()
        out["load_s"] = round(time.time() - t0, 1)
        out["vram_weights_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    except Exception as e:
        out["load_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out

    model.eval()
    ids = tok("Tom is a bompus. Tom is a zhorpus. What is Tom?", return_tensors="pt").input_ids.to("cuda")
    out["seq_len"] = int(ids.shape[1])

    # --- test 1: plain forward with hidden states + cache ---
    try:
        with torch.no_grad():
            o1 = model(input_ids=ids, use_cache=True, output_hidden_states=True)
        out["fwd_ok"] = True
        out["n_hidden"] = len(o1.hidden_states)
        out["hidden_last_shape"] = tuple(o1.hidden_states[-1].shape)
        out["cache_type"] = type(o1.past_key_values).__name__
        try:
            c = o1.past_key_values
            out["cache_len"] = int(c.get_seq_length())
            kl = c.layers[0].keys if hasattr(c, "layers") else None
            out["cache_layer0_shape"] = tuple(kl.shape) if kl is not None else "n/a"
        except Exception as e:
            out["cache_introspect_error"] = str(e)[:100]
    except Exception as e:
        out["fwd_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out

    # --- test 2: latent feedback (feed last hidden state as next input embedding) ---
    try:
        with torch.no_grad():
            h = o1.hidden_states[-1][:, -1, :].unsqueeze(1)
            o2 = model(
                inputs_embeds=h,
                past_key_values=o1.past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        out["latent_feedback_ok"] = True
        out["latent_logits_shape"] = tuple(o2.logits.shape)
        out["cache_len_after"] = int(o2.past_key_values.get_seq_length())
    except Exception as e:
        out["latent_feedback_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out

    # --- test 3: cache prefix slicing (Coconut-style) ---
    try:
        c = o2.past_key_values
        sliced = [
            (k[:, :, :2, :], v[:, :, :2, :]) for k, v in c.to_legacy_cache()
        ] if hasattr(c, "to_legacy_cache") else None
        out["legacy_cache_ok"] = sliced is not None
        if sliced:
            out["legacy_k_shape"] = tuple(sliced[0][0].shape)
    except Exception as e:
        out["legacy_cache_error"] = str(e)[:150]

    # --- test 4: memory for optimizer-step feasibility ---
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        o = model(input_ids=ids, use_cache=False, output_hidden_states=True, labels=ids)
        o.loss.backward()
        torch.cuda.synchronize()
        out["backward_ok"] = True
        out["peak_train_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception as e:
        out["backward_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    del model, o1, o2
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)} | total {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")
    results = []
    for mid in CANDIDATES:
        print(f"--- probing {mid} ---", flush=True)
        r = probe(mid)
        results.append(r)
        for k, v in r.items():
            print(f"   {k}: {v}")
        print(flush=True)

    print("\n\n===== SUMMARY =====")
    for r in results:
        ok = r.get("latent_feedback_ok", False)
        print(f"{r['model']:<28} latent_ok={ok}  peak_train={r.get('peak_train_gb','-')}GB  "
              f"arch={r.get('arch','-')} layers={r.get('layers','-')} hidden={r.get('hidden','-')}")
        if "layer_types" in r:
            print(f"    layer_types: {r['layer_types']}")
