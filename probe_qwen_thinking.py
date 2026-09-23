import json
from pathlib import Path
import time
import traceback

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
result = {"model": "Qwen3-4B-Thinking-2507", "quantization": "NF4 double quantization", "latent_steps": 4}
try:
    torch.set_num_threads(6)
    start = time.perf_counter()
    path = ROOT / "models" / "Qwen3-4B-Thinking-2507-bnb-4bit"
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tokenizer("A box contains 12 red balls and 7 blue balls. How many balls are there?", return_tensors="pt").input_ids.to("cuda")
    backbone = model.get_base_model().model
    model.train()
    torch.cuda.reset_peak_memory_stats()
    out = backbone(input_ids=ids, use_cache=True)
    z = out.last_hidden_state[:, -1:]
    z.retain_grad()
    first_z = z
    cache = out.past_key_values
    for _ in range(4):
        out = backbone(inputs_embeds=z, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        z = out.last_hidden_state[:, -1:]
    logits = model.get_base_model().lm_head(z[:, -1]).float()
    target = tokenizer("19", add_special_tokens=False).input_ids[0]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target], device="cuda"))
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    optimizer.step()
    torch.cuda.synchronize()
    result.update({"latent_forward_backward_step": True, "loss": loss.item(), "first_latent_grad_norm": first_z.grad.float().norm().item() if first_z.grad is not None else None,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "seconds": time.perf_counter() - start,
                   "trainable_parameters": sum(p.numel() for p in params)})
except Exception as e:
    result.update({"latent_forward_backward_step": False, "error": repr(e), "traceback": traceback.format_exc()})
print(json.dumps(result, indent=2), flush=True)
(ROOT / "reports" / "probe_qwen_thinking.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
if not result.get("latent_forward_backward_step"):
    raise SystemExit(1)
