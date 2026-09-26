"""s9_train_l1v3.py — L1 v3: wide-open gate (A uniform [0.1,1.0]) for adaptive exit.

Inherits from T2 checkpoint:
  - LoRA (refinement operator)
  - switch (identity-first-loop)
  - gate's dt_input_proj + delta_proj (learned selectivity patterns)
Resets:
  - gate's A_log: from arange[1,2495] (99% channels dead) to uniform[0.1,1.0]
    (all channels have opening range)
Optimizer:
  - A_log at 1/10 learning rate (prevent rapid regression to closed state)

Target: ~300 steps (~0.77M tokens), then R2 diagnostic for exit-depth variance.
Usage: loopus_env/Scripts/python.exe s9_train_l1v3.py --t2 runs/v5_l1/ckpt_turn2_mature.pt
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import (L1Qwen35, L1Gate, inject_lora, build_wiki_blocks,
                         build_gsm_blocks, build_gsm_probe, dev_ce_l1,
                         gen_l1, rep_metrics, src_is_gsm, order_for,
                         train_step_l1, save_ckpt, load_ckpt)

ROOT = Path(__file__).resolve().parent


def load_t2_partial(model, path):
    """Load T2: keep LoRA, switch, dt_input_proj, delta_proj; reset A_log."""
    ck = torch.load(path, map_location='cuda', weights_only=False)
    # LoRA
    with torch.no_grad():
        named = dict(model.backbone.named_parameters())
        for name, p in ck['lora'].items():
            named[name].copy_(p.to(named[name].device))
    # Switch
    with torch.no_grad():
        model.l1gate.s.copy_(ck['switch'].to(model.l1gate.s.device))
    # Gate: keep dt_input_proj + delta_proj, RESET A_log
    gate_sd = ck['gate']
    model.l1gate.gate.dt_input_proj.weight.data.copy_(
        gate_sd['dt_input_proj.weight'])
    model.l1gate.gate.delta_proj.weight.data.copy_(
        gate_sd['delta_proj.weight'])
    model.l1gate.gate.delta_proj.bias.data.copy_(
        gate_sd['delta_proj.bias'])
    print(f'  loaded T2 step={ck["step"]}: LoRA + switch + dt/delta_proj '
          f'(A_log RESET)', flush=True)
    return ck


def reset_A_log(model, lo=0.1, hi=1.0, seed=42):
    """Replace A_log: uniform [lo, hi] instead of log(arange 1..N)."""
    g = torch.Generator().manual_seed(seed)
    hidden = model.l1gate.gate.A_log.shape[0]
    A_new = (torch.rand(hidden, generator=g) * (hi - lo) + lo).to('cuda')
    with torch.no_grad():
        model.l1gate.gate.A_log.copy_(torch.log(A_new))
    # verify opening
    with torch.no_grad():
        probe = torch.tensor([[0.1] * hidden], device='cuda')
        delta_probe = torch.full((1, hidden), 2.5, device='cuda')
        ab = torch.exp(delta_probe * (-torch.exp(model.l1gate.gate.A_log)))
    print(f'  A reset: uniform[{lo},{hi}] -> expected A_bar mean '
          f'~{float(ab.mean()):.3f} (was ~0.083)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='v5_l1v3')
    ap.add_argument('--t2', default='runs/v5_l1/ckpt_turn2_mature.pt')
    ap.add_argument('--target-supervised-tokens', type=float, default=1e6)
    ap.add_argument('--ckpt-every', type=int, default=25)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--B', type=int, default=20)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--gate-lr', type=float, default=1e-4)
    ap.add_argument('--lora-lr', type=float, default=2e-4)
    ap.add_argument('--alog-lr-ratio', type=float, default=0.1)
    ap.add_argument('--a-lo', type=float, default=0.1)
    ap.add_argument('--a-hi', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=2028)
    ap.add_argument('--depths', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--dev-blocks', type=int, default=8)
    ap.add_argument('--resume', default='auto')
    args = ap.parse_args()

    run_dir = ROOT / 'runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'args.json').write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')
    history_path = run_dir / 'history.jsonl'

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    lora_params, _ = inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, args.B)
    load_t2_partial(model, args.t2)
    reset_A_log(model, args.a_lo, args.a_hi, args.seed)

    gate_params = [p for n, p in model.l1gate.gate.named_parameters()
                   if 'A_log' not in n]
    alog_param = model.l1gate.gate.A_log
    switch_params = [model.l1gate.s]
    trainables = (gate_params + switch_params + [alog_param] + lora_params)
    optimizer = torch.optim.AdamW([
        {'params': gate_params + switch_params, 'lr': args.gate_lr},
        {'params': [alog_param], 'lr': args.gate_lr * args.alog_lr_ratio},
        {'params': lora_params, 'lr': args.lora_lr}])
    n_train = sum(p.numel() for p in trainables)
    print(f'trainable: {n_train:,} (A_log at '
          f'{args.gate_lr * args.alog_lr_ratio:.1e} = '
          f'{args.alog_lr_ratio:.0f}x gate lr)', flush=True)

    wiki_blocks, n_wiki = build_wiki_blocks(tok, 'train', args.ctx)
    gsm_blocks, n_gsm = build_gsm_blocks(tok, args.ctx, 'train')
    dev_all, n_dev = build_wiki_blocks(tok, 'validation', args.ctx)
    dev_blocks = [dev_all[i].unsqueeze(0)
                  for i in range(min(args.dev_blocks, n_dev))]
    print(f'pools: wiki {n_wiki} / gsm {n_gsm} | dev {len(dev_blocks)}',
          flush=True)

    step, sup_tokens, turns = 0, 0, []
    ckpt_latest = run_dir / 'ckpt_latest.pt'
    ckpt_prev = run_dir / 'ckpt_prev.pt'
    if args.resume != 'none' and ckpt_latest.exists():
        step, sup_tokens, turns = load_ckpt(ckpt_latest, model, optimizer)
        print(f'resumed from step {step} ({sup_tokens:.0f} sup tokens)', flush=True)

    def append_history(rec):
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    gen_cfg = {'samples': 2, 'tokens': 12, 'depth': 2}
    if step == 0:
        append_history({'event': 'start', 'args': vars(args)})
        append_history({'event': 'dev_baseline',
                        'dev_ce': dev_ce_l1(model, dev_blocks, args.depths,
                                            tok, gen_cfg)})

    t0 = time.perf_counter()
    ep_w, ord_w = -1, None
    ep_g, ord_g = -1, None
    cw = cg = 0
    for _s in range(step):                    # deterministic data replay
        if src_is_gsm(_s):
            cg += 1
        else:
            cw += 1
    while sup_tokens < args.target_supervised_tokens:
        beta = 1.0    # no warmup needed (gate inherits T2's stability)
        if src_is_gsm(step):
            ep = cg // n_gsm
            if ep != ep_g:
                ord_g, ep_g = order_for(ep, n_gsm, args.seed + 1), ep
            x = gsm_blocks[ord_g[cg % n_gsm]].unsqueeze(0)
            cg += 1
        else:
            ep = cw // n_wiki
            if ep != ep_w:
                ord_w, ep_w = order_for(ep, n_wiki, args.seed), ep
            x = wiki_blocks[ord_w[cw % n_wiki]].unsqueeze(0)
            cw += 1
        logs = train_step_l1(model, x, args.B, args.K, beta, optimizer,
                             trainables)
        step += 1
        sup_tokens += args.K * args.ctx
        append_history({'step': step, 'sup_tokens': sup_tokens,
                        'ts': round(time.time(), 2), 'logs': logs})

        if step % args.ckpt_every == 0:
            if ckpt_latest.exists():
                if ckpt_prev.exists():
                    ckpt_prev.unlink()
                os.replace(ckpt_latest, ckpt_prev)
            save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer,
                      args, turns)
        if args.eval_every and step % args.eval_every == 0:
            entry = {'event': 'eval', 'step': step, 'sup_tokens': sup_tokens,
                     'dev_ce': dev_ce_l1(model, dev_blocks, args.depths,
                                         tok, gen_cfg),
                     'sec_per_batch': round(
                         (time.perf_counter() - t0) / step, 2)}
            append_history(entry)
            print('EVAL ' + json.dumps(entry), flush=True)

    save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer, args, turns)
    final = {'event': 'final', 'step': step, 'sup_tokens': sup_tokens,
             'dev_ce': dev_ce_l1(model, dev_blocks, args.depths, tok, gen_cfg)}
    append_history(final)
    print('FINAL ' + json.dumps(final), flush=True)


if __name__ == '__main__':
    main()