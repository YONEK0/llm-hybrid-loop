"""s10_train_v4.py — L1 v4: adaptive exit via monotonic-mask routing (AdaPonderLM).

Why v4: v1-v3 treated the closing gate as a READOUT (predict/observe a fixed
trajectory). All failed (AUC ~0.51) because a uniformly-mixed loop produces no
per-token divergence to read. AdaPonderLM (arXiv 2603.01914) shows the gate must
be a ROUTER that CHANGES the trajectory: per-token monotonic halting mask.

Three components (all from AdaPonderLM):
 1. per-iteration MLP gates: s_t = sigmoid(MLP_t(h_t)), one MLP per iteration
    (a shared MLP is "highly unstable" per the paper)
 2. monotonic mask: m <- m * 1(s_t >= tau); once a token halts it stays halted,
    its state frozen for all later iterations (KV reuse in their impl)
 3. self-supervised training: LM loss + lambda * bottom-K ponder loss (mean of
    the smallest k fraction of gate values) — NO oracle labels needed.

Inherits from v3 ckpt (runs/v5_l1v3/ckpt_v3_step225_diag.pt): LoRA refinement
operator, depth switch (identity first loop), and the wide-open SelectiveGate.

Key metric: exit-depth distribution per token (uniform => no adaptivity;
spread => real adaptive computation).
Usage: loopus_env/Scripts/python.exe s10_train_v4.py --run-name v5_l1v4
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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import (L1Qwen35, inject_lora, build_wiki_blocks,
                         build_gsm_blocks, order_for, src_is_gsm, save_ckpt,
                         load_ckpt, rep_metrics)
from s8_r0_l1 import load_ckpt as load_v3

ROOT = Path(__file__).resolve().parent
SOFT_EPS = 0.05       # halted tokens keep refining at 5% (no hard freeze)
_SOFT_EPS_T = torch.tensor(0.05, dtype=torch.bfloat16)  # bf16-safe scalar


class IterGate(nn.Module):
    """Per-iteration two-layer MLP gate over the FULL hidden state."""

    def __init__(self, hidden, h_mult=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, h_mult), nn.ReLU(), nn.Linear(h_mult, 1))
        nn.init.constant_(self.net[-1].bias, 2.0)   # start: keep going

    def forward(self, h):
        return self.net(h.float()).squeeze(-1)


class V4Router(nn.Module):
    def __init__(self, hidden, B, h_mult=256):
        super().__init__()
        self.gates = nn.ModuleList([IterGate(hidden, h_mult)
                                    for _ in range(B)])

    def forward(self, h, t):
        return torch.sigmoid(self.gates[t - 1](h))


@torch.no_grad()
def router_eval(model, router, x, depth, tau=1e-4):
    """Inference-time adaptive routing: mask+state forward, returns exit depths."""
    h = model.encode(x)
    h0 = h
    mask = torch.ones(h.shape[:2], device=h.device)
    exits = torch.full(h.shape[:2], float(depth), device=h.device)
    for t in range(1, depth + 1):
        inp = h if t == 1 else h + h0
        h_prop = model.block(inp)
        h_new = model.l1gate(h_prop, h, t)
        s = router(h_new, t)
        keep = ((s >= tau).float() * mask).to(h.dtype)
        first_halt = (mask > 0) & (keep == 0)
        exits[first_halt] = float(t)
        h = h + keep.unsqueeze(-1) * (h_new - h) + (
            1 - keep.unsqueeze(-1)) * _SOFT_EPS_T * (h_new - h)
        mask = keep
    return exits.squeeze(0)


def train_step_v4(model, router, x, B, K, beta, lam, k_frac, optimizer,
                  trainables, clip=1.0, tau=1e-4):
    """One step: LM loss + beta*monotonicity + lambda*bottom-K ponder."""
    with torch.no_grad():
        h0 = model.encode(x)
    h = h0
    mask = torch.ones(h.shape[:2], device=h.device)
    supervised = sorted(random.sample(range(B), K))
    labels = x
    logs = []
    gate_vals = []
    for b in range(B):
        t = b + 1
        h_old = h.detach()
        inp = h_old if t == 1 else h_old + h0
        h_prop = model.block(inp)
        h_new = model.l1gate(h_prop, h_old, t)
        s = router(h_new, t)                      # differentiable gate value
        gate_vals.append(s)
        keep = torch.where(s >= tau, torch.ones_like(s), torch.zeros_like(s))
        keep = (keep * mask).to(h.dtype)
        h_alive = h_old + keep.unsqueeze(-1) * (h_new - h_old) + (
            1 - keep.unsqueeze(-1)) * _SOFT_EPS_T * (h_new - h_old)
        h = h_alive.detach()
        mask = (keep * mask).detach()
        if b in supervised:
            with torch.enable_grad():
                h_g = h_old.detach().requires_grad_(False)
                h_alive_g = h_g + keep.unsqueeze(-1) * (h_new - h_g) + (
                    1 - keep.unsqueeze(-1)) * _SOFT_EPS_T * (h_new - h_g)
                logits = model.decode(h_alive_g)
                sl, st_ = logits[:, :-1], labels[:, 1:]
                lm = F.cross_entropy(sl.float().reshape(-1, sl.shape[-1]),
                                     st_.reshape(-1))
                with torch.no_grad():
                    lo_ = model.decode(h_old)[:, :-1]
                    lm_prev = F.cross_entropy(
                        lo_.float().reshape(-1, sl.shape[-1]), st_.reshape(-1))
            alive_f = float(mask.mean())
            logs.append({'b': b, 'lm': round(float(lm), 4),
                         'lm_prev': round(float(lm_prev), 4),
                         'alive_frac': round(alive_f, 4)})
            h = h_alive_g.detach()
        else:
            logs[-1] if False else None
    gate_cat = torch.cat([g.view(-1) for g in gate_vals])
    n_bottom = max(1, int(k_frac * gate_cat.numel()))
    ponder = gate_cat.topk(n_bottom, largest=False).values.mean()
    with torch.enable_grad():
        lm_final = lm + beta * F.silu(lm - lm_prev)
    loss = lm_final + lam * ponder
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(trainables, clip))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for lg in logs:
        lg.update({'grad_norm': round(gn, 4),
                   'ponder': round(float(ponder), 5),
                   'loss': round(float(loss), 4)})
    return logs


def eval_v4(model, router, blocks, B, tau=1e-4):
    """CE at fixed depths + adaptive-routing CE + exit-depth distribution."""
    out = {'ce': {}, 'exit': {}}
    all_exits, ce_adapt = [], []
    for x in blocks:
        h = model.encode(x)
        h0 = h
        mask = torch.ones(h.shape[:2], device=h.device)
        exits = torch.full(h.shape[:2], float(B), device=h.device)
        for t in range(1, B + 1):
            inp = h if t == 1 else h + h0
            h_prop = model.block(inp)
            h_new = model.l1gate(h_prop, h, t)
            s = router(h_new, t)
            keep = (torch.where(s >= tau, torch.ones_like(s),
                               torch.zeros_like(s)) * mask).to(h.dtype)
            first_halt = (mask > 0) & (keep == 0)
            exits[first_halt] = float(t)
            h = h + keep.unsqueeze(-1) * (h_new - h) + (
                1 - keep.unsqueeze(-1)) * _SOFT_EPS_T * (h_new - h)
            mask = keep
            if t in (1, 2, 4, 8):
                logits = model.decode(h)
                out['ce'].setdefault(f'd{t}', []).append(float(F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    x[:, 1:].reshape(-1))))
        logits = model.decode(h)
        ce_adapt.append(float(F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1))))
        all_exits.append(exits.squeeze(0).detach())
    out['ce'] = {k: round(sum(v) / len(v), 4) for k, v in out['ce'].items()}
    out['ce']['adaptive'] = round(sum(ce_adapt) / len(ce_adapt), 4)
    ex = torch.cat(all_exits).cpu()
    dist = {int(d): int((ex == d).sum()) for d in torch.unique(ex)}
    out['exit'] = {'dist': dist, 'mean': round(float(ex.mean()), 2),
                   'n_distinct': len(dist)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='v5_l1v4')
    ap.add_argument('--v3', default='runs/v5_l1v3/ckpt_v3_step225_diag.pt')
    ap.add_argument('--target-supervised-tokens', type=float, default=5e5)
    ap.add_argument('--ckpt-every', type=int, default=25)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--B', type=int, default=8)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--k-frac', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--gate-h', type=int, default=256)
    ap.add_argument('--seed', type=int, default=2029)
    ap.add_argument('--dev-blocks', type=int, default=4)
    ap.add_argument('--resume', default='auto')
    args = ap.parse_args()

    run_dir = ROOT / 'runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'args.json').write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')
    hp = run_dir / 'history.jsonl'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)   # switch must match the v3 archive
    v3 = load_v3(model, args.v3)          # LoRA + switch + wide SelectiveGate
    for p in model.parameters():
        p.requires_grad_(False)           # frozen backbone: train router only

    hidden = int(model.norm.weight.shape[0])
    router = V4Router(hidden, args.B, args.gate_h).cuda()
    trainables = list(router.parameters())
    optimizer = torch.optim.AdamW(trainables, lr=args.lr)
    n_par = sum(p.numel() for p in trainables)
    print(f'v4 router: {n_par:,} params | {args.B} per-iteration MLPs '
          f'hidden={args.gate_h} | inherited v3 step={v3["step"]}', flush=True)

    wiki_blocks, n_wiki = build_wiki_blocks(tok, 'train', args.ctx)
    gsm_blocks, n_gsm = build_gsm_blocks(tok, args.ctx)
    dev_all, n_dev = build_wiki_blocks(tok, 'validation', args.ctx)
    dev_blocks = [dev_all[i].unsqueeze(0)
                  for i in range(min(args.dev_blocks, n_dev))]

    step, sup_tokens = 0, 0
    ckpt_latest = run_dir / 'ckpt_latest.pt'
    ckpt_prev = run_dir / 'ckpt_prev.pt'
    if args.resume != 'none' and ckpt_latest.exists():
        ck = torch.load(ckpt_latest, map_location='cuda', weights_only=False)
        router.load_state_dict(ck['router'])
        optimizer.load_state_dict(ck['optimizer'])
        step, sup_tokens = ck['step'], ck['sup_tokens']
        torch.set_rng_state(ck['rng']['torch'].cpu())
        torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng']['cuda']])
        random.setstate(ck['rng']['python'])
        print(f'resumed from step {step} ({sup_tokens:.0f} sup tokens)',
              flush=True)

    def append_history(rec):
        with hp.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def _save(src, dst):
        if src is None or not Path(src).exists():
            ck = {'step': step, 'sup_tokens': sup_tokens,
                  'router': router.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'rng': {'torch': torch.get_rng_state(),
                          'cuda': torch.cuda.get_rng_state_all(),
                          'python': random.getstate()},
                  'args': vars(args)}
        else:
            ck = torch.load(src, map_location='cuda', weights_only=False)
        ck['turns'] = turns
        tmp = Path(str(dst) + '.tmp')
        torch.save(ck, tmp)
        os.replace(tmp, dst)

    _hist = []
    turns = []

    cw = cg = 0
    for _s in range(step):
        if src_is_gsm(_s):
            cg += 1
        else:
            cw += 1
    ep_w, ord_w = -1, None
    ep_g, ord_g = -1, None

    if step == 0:
        append_history({'event': 'start', 'args': vars(args)})
        append_history({'event': 'eval', 'step': 0, 'sup_tokens': 0,
                        **eval_v4(model, router, dev_blocks, args.B)})

    while sup_tokens < args.target_supervised_tokens:
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
        logs = train_step_v4(model, router, x, args.B, args.K, args.beta,
                             args.lam, args.k_frac, optimizer, trainables)
        step += 1
        sup_tokens += args.K * args.ctx
        append_history({'step': step, 'sup_tokens': sup_tokens,
                        'ts': round(time.time(), 2), 'logs': logs})
        if step % args.ckpt_every == 0:
            if ckpt_latest.exists():
                if ckpt_prev.exists():
                    ckpt_prev.unlink()
                os.replace(ckpt_latest, ckpt_prev)
            tmp = ckpt_latest.with_suffix('.tmp')
            torch.save({'step': step, 'sup_tokens': sup_tokens,
                        'router': router.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rng': {'torch': torch.get_rng_state(),
                                'cuda': torch.cuda.get_rng_state_all(),
                                'python': random.getstate()},
                        'args': vars(args)}, tmp)
            os.replace(tmp, ckpt_latest)
        if args.eval_every and step % args.eval_every == 0:
            e = eval_v4(model, router, dev_blocks, args.B)
            rec = {'event': 'eval', 'step': step, 'sup_tokens': sup_tokens,
                   **e}
            append_history(rec)
            print('EVAL ' + json.dumps(rec), flush=True)
            _hist.append(e)
            _nd = e['exit']['n_distinct']
            _mean = e['exit']['mean']
            _ce = e['ce'].get('adaptive')
            _fire = []
            if 't1_split' not in turns and _nd >= 4 and _mean < args.B - 0.5:
                _fire.append(('t1_split', 'ckpt_turn1_split.pt'))
            if ('t1_split' in turns and 't2_mature' not in turns
                    and len(_hist) >= 4
                    and abs(_hist[-4]['ce'].get('adaptive', 9)
                            - (_ce or 9)) < 0.01 and _nd >= 4):
                _fire.append(('t2_mature', 'ckpt_turn2_mature.pt'))
            for _name, _fn in _fire:
                turns = turns + [_name]
                _save(ckpt_latest if ckpt_latest.exists() else None,
                      run_dir / _fn)
                append_history({'event': 'turn', 'name': _name,
                                'step': step, 'sup_tokens': sup_tokens,
                                'exit': e['exit'], 'ce': e['ce']})
                print(f'TURNING-POINT {_name} saved @ step {step} '
                      f'(distinct={_nd} mean_exit={_mean})', flush=True)
    fin = {'event': 'final', 'step': step, 'sup_tokens': sup_tokens,
           **eval_v4(model, router, dev_blocks, args.B)}
    append_history(fin)
    print('FINAL ' + json.dumps(fin), flush=True)


if __name__ == '__main__':
    main()
