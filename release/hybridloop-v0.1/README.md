# HybridLoop v0.1 — Adaptive-Depth Hybrid-Attention Language Model

Research prototype. A Qwen3.5-4B-Base model retrofitted with a recurrent
middle block (layers 8–19) and a **per-token adaptive exit router**: every
token decides for itself how many loop iterations to run (1–8), via a
monotonic halting mask (AdaPonderLM-style).

## Quick start

```python
from hybridloop_model import load_release, generate
model, router, tok, ck = load_release("hybridloop-v0.1.pt")
print(generate(model, router, tok, "Once upon a time", 128))
```

or `python infer.py "prompt"` / `python infer.py --interactive`.

Requirements: see requirements.txt (8GB-class GPU; the backbone loads in 4-bit).

## How it works

Qwen3.5-4B's 32 layers are split E(0–7) → M(8–19, weight-tied recurrent loop)
→ D(20–31). After each loop iteration an MLP router scores every token;
tokens whose gate falls below τ are frozen (soft decay keeps 5% refinement so
quality is preserved — see the freeze-damage isolation study in the repo).
The LM head reads whatever state each token reached.

Training: 0.58M tokens of adapter-only tuning (LoRA r=8 + gates + router,
backbone frozen) on a wikitext+gsm8k mix, inheriting a 1.79M-token precursor.
Full lineage and the four-round adaptive-exit campaign are documented in the
source repo (`docs/plans/TRAINING_PLAN_V5_HYBRID_LOOPUS.md`).

## Honest limitations

- **Slow**: no KV cache — generation re-runs the loop per token (~1–8 s/token
  on an 8GB laptop GPU). Compute-saving KV reuse is future work.
- **Weak on math**: GSM8K strict accuracy ~17% (6 items, directional); numeric
  magnitude errors are the dominant failure mode.
- **Base-model style**: continuation model, not instruction-tuned.
- **Research grade**: single seed, small eval sets, all numbers directional.
- The backbone is Qwen3.5-4B-Base (Apache-2.0); this package adds the
  loop/routing components trained in this repo (same license).
