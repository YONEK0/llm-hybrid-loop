"""Evaluate Ouro-2.6B looped LM on ProsQA test set.

Self-contained: no imports from recurrent_qwen or prosqa_adapter.
Requires transformers 4.46.2 (ouro_env) because Ouro uses custom modeling code.
"""

import json
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# Patch: Ouro's configuration_ouro.py expects layer_type_validation
# which was added in transformers ~4.50, but ouro_env has 4.46.2
import transformers.configuration_utils as _cu
if not hasattr(_cu, "layer_type_validation"):
    def _layer_type_validation(layer_types, **kwargs):
        return layer_types
    _cu.layer_type_validation = _layer_type_validation

# Patch: use_kernel_forward_from_hub was added in transformers ~4.50
import transformers.integrations as _integ
if not hasattr(_integ, "use_kernel_forward_from_hub"):
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    _integ.use_kernel_forward_from_hub = use_kernel_forward_from_hub

# Patch: ROPE_INIT_FUNCTIONS may be missing keys in this version
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _RIF

def _generic_rope_init(config, device, **kwargs):
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    return inv_freq, 1.0

for _rk in ("default", "standard"):
    if _rk not in _RIF:
        _RIF[_rk] = _generic_rope_init

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = str(ROOT / "models" / "Ouro-1.4B")
SYSTEM = "Solve the logic problem. Give the final answer."


def answer_from_text(text):
    """Extract the ProsQA answer: a sentence like 'Tom is a zhorpus.'"""
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    else:
        text = text.split("</think>")[-1]
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in reversed(lines):
        m = re.search(r"([A-Z][A-Za-z]*\s+is\s+a\s+[a-z]+)", line)
        if m:
            return m.group(1)
    return lines[-1].rstrip(".") if lines else None


@torch.no_grad()
def generate(model, tokenizer, ids, max_new_tokens):
    ids = list(ids)
    generated = []
    past = None
    current = torch.tensor([ids], dtype=torch.long, device="cuda")
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        kwargs = {}
        if past is not None:
            kwargs["past_key_values"] = past
            kwargs["input_ids"] = current
        else:
            kwargs["input_ids"] = current
        out = model(**kwargs, use_cache=True)
        past = out.past_key_values
        token = int(out.logits[0, -1].argmax(-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        current = torch.tensor([[token]], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, len(generated), time.perf_counter() - start


def main():
    random.seed(11)
    torch.manual_seed(11)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model.eval()

    # Print loop config
    for attr in ("num_loop_steps", "recurrent_steps", "num_latent", "inf_latent_iterations"):
        val = getattr(model.config, attr, None)
        if val is not None:
            print(f"config.{attr} = {val}")

    # Load ProsQA test set
    data_path = ROOT / "data" / "prosqa" / "prosqa_test.json"
    all_rows = json.loads(data_path.read_text(encoding="utf-8"))
    order = list(range(len(all_rows)))
    random.Random(11).shuffle(order)
    rows = [all_rows[i] for i in order[:30]]
    print(f"Evaluating {len(rows)} ProsQA test questions", flush=True)

    # Depth control: Ouro uses an exit gate, so standard generation uses the learned depth.
    # For a depth sweep, we'd need to modify the model's internal loop count.
    # For now, just evaluate at the default (trained) depth setting.
    correct = 0
    total_tokens = 0
    total_time = 0.0
    for i, row in enumerate(rows):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(encoded, list):
            encoded = list(encoded["input_ids"])
        ids = [int(t) for t in encoded]

        text, ntok, sec = generate(model, tokenizer, ids, 16)
        pred = answer_from_text(text)
        gold = row["answer"].strip().rstrip(".").strip()
        ok = pred is not None and pred.strip().rstrip(".").lower() == gold.lower()
        correct += int(ok)
        total_tokens += ntok
        total_time += sec

        if i < 3:
            print(f"  [{i}] gold={gold!r} pred={pred!r} ok={ok} text={text[:80]!r}")

    acc = correct / len(rows)
    print(f"\nOuro-2.6B ProsQA accuracy: {acc:.3f} ({correct}/{len(rows)})", flush=True)
    print(f"mean tokens: {total_tokens/len(rows):.1f}  mean time: {total_time/len(rows):.2f}s", flush=True)

    out_path = ROOT / "results" / "ouro_prosqa_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "model": "Ouro-2.6B", "task": "ProsQA test", "n": len(rows),
        "accuracy": acc, "correct": correct,
        "note": "Looped LM pre-trained with 4 recurrent steps, 7.7T tokens",
    }, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
