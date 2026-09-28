"""export_hf.py — export HybridLoop as a standard HF model directory.

Merges the LoRA deltas into the bf16 Qwen3.5-4B text backbone and writes a
directory loadable with:

    AutoModelForCausalLM.from_pretrained(OUT, trust_remote_code=True)

Runs on CPU (31.7GB RAM; the 8GB GPU cannot hold a bf16 4B model).
Usage: loopus_env/Scripts/python.exe export/export_hf.py --out export/hybridloop-v0.1-hf
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from transformers import AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_hybridloop import (HybridLoopConfig, HybridLoopForCausalLM)

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen3.5-4B-Base"
LORA_ALPHA_OVER_R = 2.0          # LoraConfig(lora_alpha=2*r) -> scale = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/v5_l1v4c/ckpt_v4c_final_step245.pt')
    ap.add_argument('--v3', default='runs/v5_l1v3/ckpt_v3_step225_diag.pt')
    ap.add_argument('--out', default='export/hybridloop-v0.1-hf')
    ap.add_argument('--tokenizer-from', default=BASE)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print('[1/5] loading base in bf16 on CPU ...', flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    base.eval()

    print('[2/5] building HybridLoop and copying weights ...', flush=True)
    cfg = HybridLoopConfig(
        text_config=(base.config.get_text_config().to_dict()
                     if hasattr(base.config, 'get_text_config')
                     else base.config.to_dict()),
        loop_lo=8, loop_hi=20, n_switch=20, max_depth=8, gate_hidden=256,
        soft_eps=0.05, tau=1e-4, adaptive_exit=True,
        architectures=["HybridLoopForCausalLM"],
        auto_map={"AutoConfig": "modeling_hybridloop.HybridLoopConfig",
                  "AutoModelForCausalLM":
                      "modeling_hybridloop.HybridLoopForCausalLM"},
        tie_word_embeddings=bool(getattr(base.config, "tie_word_embeddings",
                                         False)))
    model = HybridLoopForCausalLM(cfg)
    sd_base = base.state_dict()
    sd = model.state_dict()
    copied = 0
    for k, v in sd_base.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k].copy_(v)
            copied += 1
    model.load_state_dict(sd, strict=False)
    missing = [k for k in model.state_dict()
               if k not in sd_base or sd_base[k].shape != model.state_dict()[k].shape]
    print(f'      copied {copied} tensors; model-specific (missing from base): '
          f'{len(missing)} (expected: gate/router/switch)', flush=True)
    del base, sd_base, sd
    import gc
    gc.collect()

    print('[3/5] merging LoRA deltas ...', flush=True)
    v3 = torch.load(ROOT / args.v3, map_location='cpu', weights_only=False)
    named = dict(model.named_parameters())
    pairs = {}
    for k, v in v3['lora'].items():
        mod = k.rsplit('.lora_', 1)[0]
        which = 'A' if '.lora_A.' in k else 'B'
        pairs.setdefault(mod, {})[which] = v.float()
    merged = 0
    with torch.no_grad():
        for mod, ab in pairs.items():
            key = mod + '.weight'
            if key not in named:
                print('      WARN: not found', key, flush=True)
                continue
            W = named[key]
            delta = (ab['B'] @ ab['A']) * LORA_ALPHA_OVER_R
            W.copy_((W.float() + delta).to(W.dtype))
            merged += 1
    print(f'      merged {merged}/{len(pairs)} LoRA targets', flush=True)

    print('[4/5] loading gate / switch / router ...', flush=True)
    with torch.no_grad():
        model.l1gate.gate.load_state_dict(v3['gate'])
        model.l1gate.s.copy_(v3['switch'].float())
        v4 = torch.load(ROOT / args.ckpt, map_location='cpu',
                        weights_only=False)
        model.router.load_state_dict(v4['router'])
    lineage = {'l0_tokens': 5002240, 'v2_tokens': 1792000,
               'v3_tokens': 576000, 'v4c_tokens': 460800,
               'archives': ['runs/v5_l0/ckpt_l0_final_step1954.pt',
                            'runs/v5_l1/ckpt_turn1_inversion.pt',
                            'runs/v5_l1/ckpt_turn2_mature.pt',
                            'runs/v5_l1v3/ckpt_v3_step225_diag.pt',
                            'runs/v5_l1v4c/ckpt_v4c_final_step245.pt']}

    print('[5/5] saving HF directory ...', flush=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size='2GB')
    cfg.save_pretrained(out)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_from)
    tok.save_pretrained(out)
    shutil.copy(Path(__file__).resolve().parent / 'modeling_hybridloop.py',
                out / 'modeling_hybridloop.py')
    (out / 'lineage.json').write_text(
        json.dumps(lineage, indent=1), encoding='utf-8')
    print(f'      done in {time.perf_counter() - t0:.0f}s -> {out}',
          flush=True)
    for f in sorted(out.iterdir()):
        print('       ', f.name, f'{f.stat().st_size / 2**20:.1f}MB')


if __name__ == '__main__':
    main()
