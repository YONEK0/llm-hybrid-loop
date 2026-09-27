"""s11_b_v4.py — B-chain free-running test for v4-c adaptive routing.

Compares three decoding modes on the SAME GSM8K items (paired, same seed/order,
1-shot exemplar, 256-token budget, strict v1.1 scorer + loose diagnostic):
  - base      : single block pass, no loop (gate bypassed)
  - fixed_d4  : fixed 4 loops, no routing
  - adaptive  : v4-c router decides per-token depth (adaptive exit)

Asks: does adaptive routing help or hurt REAL free generation (not CE)?
Writes results/s11_b_v4.json
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import L1Qwen35, inject_lora, loop_state
from s8_r0_l1 import load_ckpt as load_v3
from s10_train_v4 import V4Router

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def decode_routed(model, router, x, B, tau=1e-4, eps=0.05):
    """v4-c semantics: per-token monotonic mask, soft decay for halted tokens."""
    h = model.encode(x)
    h0 = h
    mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
    for t in range(1, B + 1):
        inp = h if t == 1 else h + h0
        h_prop = model.block(inp)
        h_new = model.l1gate(h_prop, h, t)
        s = router(h_new, t)
        keep = ((s >= tau).float() * mask).to(h.dtype)
        h = h + keep.unsqueeze(-1) * (h_new - h) + (
            1 - keep.unsqueeze(-1)) * eps * (h_new - h)
        mask = keep
    return model.decode(h)


@torch.no_grad()
def generate(model, router, tok, prompt_ids, mode, B, max_new):
    ids = list(prompt_ids)
    eos = tok.eos_token_id
    for _ in range(max_new):
        x = torch.tensor([ids], device='cuda')
        if mode == 'base':
            h = model.encode(x)
            logits = model.decode(model.block(h))
        elif mode == 'adaptive':
            logits = decode_routed(model, router, x, B)
        else:
            logits = model.decode(loop_state(model, x, int(mode[1:])))
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        if t == eos:
            break
    new = ids[len(prompt_ids):]
    finished = bool(new and new[-1] == eos)
    return tok.decode(new, skip_special_tokens=True), finished, len(new)


def loose_pred(text):
    if '####' in text:
        seg = text.rsplit('####', 1)[1]
        m = re.search(r'-?\d[\d,]*(?:\.\d+)?', seg)
        if m:
            return m.group(0).replace(',', '').rstrip('.').rstrip('0').rstrip('.') or '0'
    nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--router', default='runs/v5_l1v4c/ckpt_v4c_final_step245.pt')
    ap.add_argument('--v3', default='runs/v5_l1v3/ckpt_v3_step225_diag.pt')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--gen-tokens', type=int, default=256)
    ap.add_argument('--B', type=int, default=8)
    ap.add_argument('--modes', nargs='+',
                    default=['base', 'd4', 'adaptive'])
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    from scorer_strict import extract_gsm8k_v11, gsm8k_finished, _norm_num
    items = []
    for line in open(ROOT / 'data' / 'gsm8k' / 'test.jsonl', encoding='utf-8'):
        r = json.loads(line)
        g = _norm_num(str(r['answer']))
        if g is not None:
            items.append({'q': r['question'],
                          'a': r['rationale'] + '\n#### ' + r['answer'],
                          'gold': g})
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    ex = items[order[0]]
    gen_items = [items[i] for i in order[300:300 + args.n]]
    n = len(gen_items)
    print(f's11_b_v4: n={n} budget={args.gen_tokens} modes={args.modes}',
          flush=True)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, 8, 19, 8)
    model = L1Qwen35(backbone, 20)
    model.eval()
    load_v3(model, args.v3)
    hidden = int(model.norm.weight.shape[0])
    router = V4Router(hidden, args.B, 256).cuda()
    ck = torch.load(args.router, map_location='cuda', weights_only=False)
    router.load_state_dict(ck['router'])
    router.eval()
    print(f'loaded v3 base + v4-c router (step={ck["step"]})', flush=True)

    shots = f"Q: {ex['q']}\nA: Let's think step by step. {ex['a']}\n\n"
    out = {'time': time.strftime('%F %T'), 'router': args.router,
           'router_step': ck['step'], 'n': n, 'gen_tokens': args.gen_tokens,
           'protocol': 'strict v1.1 (boundary finish) + loose diagnostic',
           'verdict_scope': f'DIRECTIONAL (n={n}<1000)', 'runs': {}}
    prev = {}
    if Path(ROOT / 'results' / 's11_b_v4.json').exists():
        try:
            prev = json.loads((ROOT / 'results' / 's11_b_v4.json')
                              .read_text(encoding='utf-8'))['runs']
        except Exception:
            prev = {}
    for mode in args.modes:
        if mode in prev and len(prev[mode].get('items', [])) >= n:
            out['runs'][mode] = prev[mode]      # mode already complete
            print(f'== {mode}: complete in prior run, skipped', flush=True)
            continue
        recs, ks, kl, nf, ne = [], 0, 0, 0, 0
        t0 = time.perf_counter()
        for i, it in enumerate(gen_items):
            prompt = shots + f"Q: {it['q']}\nA: Let's think step by step."
            p_ids = tok.encode(prompt)[:768 - args.gen_tokens - 8]
            text, eos_hit, n_tok = generate(model, router, tok, p_ids, mode,
                                            args.B, args.gen_tokens)
            fin = gsm8k_finished(text, eos_hit)
            pred = extract_gsm8k_v11(text, eos_hit)
            lp = loose_pred(text)
            hit = pred is not None and pred == it['gold']
            lhit = lp is not None and lp == it['gold']
            ks += int(hit)
            kl += int(lhit)
            nf += int(fin)
            ne += int(eos_hit)
            recs.append({'q': it['q'], 'gold': it['gold'], 'raw': text[:900],
                         'pred': pred, 'loose': lp, 'finished': fin,
                         'eos': eos_hit, 'n_tokens': n_tok,
                         'strict_correct': hit, 'loose_correct': lhit})
            print(f'  [{mode}] {i+1}/{n}: strict={hit} loose={lhit} '
                  f'gold={it["gold"]} fin={fin} [{time.perf_counter()-t0:.0f}s]',
                  flush=True)
        out['runs'][mode] = {'strict': ks, 'loose': kl, 'n': n,
                             'finished_rate': round(nf / n, 4),
                             'eos_rate': round(ne / n, 4),
                             'sec': round(time.perf_counter() - t0, 1),
                             'items': recs}
        print(f'== {mode}: strict {ks}/{n} loose {kl}/{n} fin {nf}/{n}',
              flush=True)
        p = ROOT / 'results' / 's11_b_v4.json'
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding='utf-8')
    print('written results/s11_b_v4.json', flush=True)


if __name__ == '__main__':
    main()
