"""s7_b_recheck.py — B-chain free-running recheck on L1 turning-point ckpts.

Paired-comparable with the L0 B-chain (results/s4_tier1_b_freegen.json):
same seed/order/items/exemplar (1-shot, order[0]), same 256-token budget,
same strict scorer + same loose diagnostic. Adds repetition metrics to
directly measure the L0 repetition disease (invests-1000 loops).
Configs: trained:<d> (L1 loop_state semantics) + base:d0 (gate bypassed).
Usage: loopus_env/Scripts/python.exe s7_b_recheck.py --ckpt runs/v5_l1/ckpt_turn2_mature.pt
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
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s7_train_l1 import L1Qwen35, inject_lora, loop_state, rep_metrics
from s4_tier1_eval import wilson
from scorer_strict import (extract_gsm8k_strict, extract_gsm8k_v11,
                            gsm8k_finished, _norm_num)

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def gen_one(model, tok, prompt_ids, depth, max_new):
    """depth=None -> base single pass (gate/reinjection bypassed)."""
    ids = list(prompt_ids)
    eos = tok.eos_token_id
    for _ in range(max_new):
        x = torch.tensor([ids], device='cuda')
        if depth is None:
            h = model.encode(x)
            h = model.block(h)
            logits = model.decode(h)
        else:
            logits = model.decode(loop_state(model, x, depth))
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        if t == eos:
            break
    new = ids[len(prompt_ids):]
    finished = bool(new and new[-1] == eos)
    return tok.decode(new, skip_special_tokens=True), finished, len(new)


def loose_pred(text):
    """Diagnostic-only (protocol rule 4): number after last ####, else last number."""
    if '####' in text:
        seg = text.rsplit('####', 1)[1]
        m = re.search(r'-?\d[\d,]*(?:\.\d+)?', seg)
        if m:
            return m.group(0).replace(',', '').rstrip('.').rstrip('0').rstrip('.') or '0'
    nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/v5_l1/ckpt_turn2_mature.pt')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--gen-tokens', type=int, default=256)
    ap.add_argument('--depths', type=int, nargs='+', default=[4])
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    items = []
    for line in open(ROOT / 'data' / 'gsm8k' / 'test.jsonl', encoding='utf-8'):
        r = json.loads(line)
        g = _norm_num(str(r['answer']))
        if g is not None:
            items.append({'q': r['question'], 'a': r['rationale'] + chr(10) + '#### ' + r['answer'], 'gold': g})
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    ex = items[order[0]]
    gen_items = [items[i] for i in order[300:300 + args.n]]
    n = len(gen_items)
    print(f's7_b_recheck: n={n} budget={args.gen_tokens} ckpt={args.ckpt}', flush=True)

    ck = torch.load(ROOT / args.ckpt, map_location='cuda', weights_only=False)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    r_lora = int(ck['args'].get('lora_r', 8))
    inject_lora(backbone, 8, 19, r_lora)
    model = L1Qwen35(backbone, 20)
    model.l1gate.gate.load_state_dict(ck['gate'])
    with torch.no_grad():
        model.l1gate.s.copy_(ck['switch'].to(model.l1gate.s.device))
        named = dict(model.backbone.named_parameters())
        for name, p in ck['lora'].items():
            named[name].copy_(p.to(named[name].device))
    model.eval()
    print(f'loaded L1 ckpt step={ck["step"]} sup={ck["sup_tokens"]:.0f} '
          f'turns={ck.get("turns")}', flush=True)

    shots = f"Q: {ex['q']}\nA: Let's think step by step. {ex['a']}\n\n"
    out = {'time': time.strftime('%F %T'), 'ckpt': args.ckpt,
           'ckpt_step': ck['step'], 'sup_tokens': ck['sup_tokens'],
           'turns': ck.get('turns'), 'n': n, 'gen_tokens': args.gen_tokens,
           'shots': 1,
           'protocol': ('free-running strict (rule 2) + loose diagnostic '
                        '(rule 4) + repetition metrics; paired items/order/'
                        'budget with L0 run results/s4_tier1_b_freegen.json'),
           'verdict_scope': f'DIRECTIONAL (n={n}<1000)', 'runs': {}}

    configs = [(f'trained:d{d}', d) for d in args.depths] + [('base:d0', None)]
    for tag, depth in configs:
        recs, k, kl, nf, nh, ne = [], 0, 0, 0, 0, 0
        t0 = time.perf_counter()
        for i, it in enumerate(gen_items):
            prompt = shots + f"Q: {it['q']}\nA: Let's think step by step."
            p_ids = tok.encode(prompt)[:768 - args.gen_tokens - 8]
            text, finished, n_tok = gen_one(model, tok, p_ids,
                                            depth, args.gen_tokens)
            fin = gsm8k_finished(text, finished)
            pred = extract_gsm8k_v11(text, finished)
            lp = loose_pred(text)
            hit = (pred is not None and pred == it['gold'])
            lhit = (lp is not None and lp == it['gold'])
            k += int(hit)
            kl += int(lhit)
            nf += int(fin)
            ne += int(finished)
            nh += int('####' in text)
            rp = rep_metrics(text)
            recs.append({'q': it['q'], 'gold': it['gold'], 'raw': text[:900],
                         'pred': pred, 'loose': lp, 'has_hash': '####' in text,
                         'finished': fin, 'eos': finished, 'n_tokens': n_tok,
                         'strict_correct': hit, 'loose_correct': lhit,
                         'rep': rp})
            print(f"  [{tag}] item{i+1}/{n}: strict={hit} loose={lhit} "
                  f"gold={it['gold']} loose_pred={lp} (hash={recs[-1]['has_hash']} "
                  f"eos={finished} {n_tok}tok rep4={rp['rep4']} "
                  f"d2={rp['distinct2']}) [{time.perf_counter()-t0:.0f}s]",
                  flush=True)
        out['runs'][tag] = {
            'depth': depth, 'strict_acc': round(k / n, 4),
            'loose_acc': round(kl / n, 4), 'k': k, 'k_loose': kl, 'n': n,
            'wilson95_loose': wilson(kl, n),
            'finished_rate': round(nf / n, 4),
            'eos_rate': round(ne / n, 4), 'hash_rate': round(nh / n, 4),
            'sec': round(time.perf_counter() - t0, 1), 'items': recs}
        print(f"== {tag}: strict {k}/{n} loose {kl}/{n} fin {nf}/{n} eos {ne}/{n} "
              f"hash {nh}/{n}", flush=True)
        p = ROOT / 'results' / 's7_b_recheck.json'
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding='utf-8')

    print('written results/s7_b_recheck.json', flush=True)


if __name__ == '__main__':
    main()
