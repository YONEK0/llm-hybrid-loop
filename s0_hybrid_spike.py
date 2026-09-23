"""S0.4: 4-bit (NF4) loading + trainability spike for Qwen3.5-4B-Base (hybrid 3:1).

Last open S0 item of TRAINING_PLAN_V5_HYBRID_LOOPUS.md:
  - transformers 5.3.0 (loopus_env) + bitsandbytes NF4 on the multimodal hybrid wrapper
  - text-only forward at ctx 512 -> peak VRAM, wall time
  - autograd must flow through the frozen 4-bit backbone into a small fp32 probe module
    (precondition for gate/LoRA training), with zero quantized weights unfrozen

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s0_hybrid_spike.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
CTX = 512
OUT = Path("results/s0_hybrid_spike.json")

result = {"model": MODEL, "ctx": CTX, "env": "loopus_env (transformers 5.3.0, bnb 0.50.2)"}

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, quantization_config=bnb, device_map={"": 0})
model.eval()
result["load_seconds"] = round(time.perf_counter() - t0, 1)
result["model_class"] = type(model).__name__

tm = getattr(getattr(model, "model", None), "language_model", None) \
    or getattr(model, "model", model)
layers = tm.layers
result["n_layers"] = len(layers)
result["config_num_hidden_layers"] = getattr(tm.config, "num_hidden_layers", None)
module_types = [getattr(l, "layer_type", "full_attention") for l in layers]
cfg_types = list(getattr(tm.config, "layer_types", []) or [])
result["layer_type_counts"] = dict(Counter(module_types))
result["config_module_layer_types_match"] = (
    bool(cfg_types) and cfg_types == module_types)
result["gpu"] = torch.cuda.get_device_name(0)

text = "The quick brown fox jumps over the lazy dog. " * 20
ids = tok.encode(text)[:CTX]
x = torch.tensor([ids], device="cuda")
labels = x[:, 1:]

# ---- 1) forward pass, frozen ----
torch.cuda.reset_peak_memory_stats()
t1 = time.perf_counter()
with torch.no_grad():
    out = model(input_ids=x, use_cache=False)
result["forward_seconds"] = round(time.perf_counter() - t1, 2)
result["peak_gib_forward"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
result["forward_loss_free_run"] = round(
    F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]),
                    labels.reshape(-1)).item(), 3)

# ---- 2) trainability probe: fp32 module on top of the last hidden state ----
cap = {}
def last_hook(mod, args, output):
    cap["h"] = output[0] if isinstance(output, tuple) else output
handle = layers[-1].register_forward_hook(last_hook)

dim = int(tm.config.hidden_size)
probe = torch.nn.Linear(dim, dim, dtype=torch.float32, device="cuda")
with torch.no_grad():
    probe.weight.copy_(torch.eye(dim))
    probe.bias.zero_()

model.train()          # probes need grads; quantized weights stay frozen below
for p in model.parameters():
    p.requires_grad_(False)
for p in probe.parameters():
    p.requires_grad_(True)

t2 = time.perf_counter()
cap.clear()
model(input_ids=x, use_cache=False)   # grad-enabled forward, capture via hook
h = cap["h"]
y = probe(h.float()).to(h.dtype)
logits = model.get_output_embeddings()(tm.norm(y)).float()
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels.reshape(-1))
loss.backward()
result["probe_forward_backward_seconds"] = round(time.perf_counter() - t2, 2)

g = probe.weight.grad
result["probe_grad_finite"] = bool(torch.isfinite(g).all().item())
result["probe_grad_absmax"] = float(g.abs().max().item())
result["probe_loss"] = round(float(loss.item()), 3)
result["probe_params"] = sum(p.numel() for p in probe.parameters()
                             if p.requires_grad)
# base-model params requiring grad must be zero: quantized weights stay frozen
result["base_trainable_params"] = sum(p.numel() for p in model.parameters()
                                      if p.requires_grad)
result["quantized_weights_touched"] = result["base_trainable_params"] > 0
result["peak_gib_backward"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
handle.remove()

result["verdict"] = (
    "PASS" if result["probe_grad_finite"] and result["probe_grad_absmax"] > 0
             and result["probe_params"] == dim * dim + dim
             and result["base_trainable_params"] == 0
    else "CHECK")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"written to {OUT}")
