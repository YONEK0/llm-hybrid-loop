import gc
import json
from pathlib import Path
import time
import traceback

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/li287/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
result = {"model": "Qwen3.5-0.8B-Base", "purpose": "architecture compatibility only"}
try:
    torch.set_num_threads(6)
    model = Qwen3_5ForCausalLM.from_pretrained(SOURCE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"], task_type="CAUSAL_LM"))
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    ids = tokenizer("A box contains 12 red balls and 7 blue balls. How many balls are there?", return_tensors="pt").input_ids.to("cuda")
    backbone = model.get_base_model().model
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        full = backbone(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        half = ids.shape[1] // 2
        a = backbone(input_ids=ids[:, :half], use_cache=True)
        b = backbone(input_ids=ids[:, half:], use_cache=True, past_key_values=a.past_key_values)
        result["cache_full_max_diff"] = float((full.float() - b.last_hidden_state[:, -1].float()).abs().max())
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
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "seconds": time.perf_counter() - start})
except Exception as e:
    result.update({"latent_forward_backward_step": False, "error": repr(e), "traceback": traceback.format_exc()})
print(json.dumps(result, indent=2), flush=True)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "probe_hybrid.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
