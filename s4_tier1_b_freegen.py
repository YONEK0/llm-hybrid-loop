"""s4_tier1_b_freegen.py — tier-1 B chain: free-running strict GSM8K probe.

Redesigned after the 96-token budget failure (all items hit budget without EOS
=> protocol rule 2 scores them None). Now: 256-token budget so the model can
finish, 1-shot CoT exemplar (format insurance), n=4 paired items x 3 configs
(trained:d1 / trained:d4 / base). Same items/order/budget across configs.
Writes results/s4_tier1_b_freegen.json
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
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s2_spike import LoopUSQwen35
from scorer_strict import extract_gsm8k_strict, _norm_num
from s4_tier1_eval import generate, wilson

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--gen-tokens', type=int, default=256)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

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
    ex = items[order[0]]                    # 1-shot exemplar
    gen_items = [items[i] for i in order[300:300 + args.n]]
    n = len(gen_items)

    ck = torch.load(ROOT / 'runs' / 'v5_l0' / 'ckpt_latest.pt',
                    map_location='cuda', weights_only=False)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    torch.manual_seed(args.seed)
    model = LoopUSQwen35(backbone)
    model.gate.load_state_dict(ck['gate'])
    model.q_head.load_state_dict(ck['q_head'])
    model.eval()

    shots = f"Q: {ex['q']}\nA: Let's think step by step. {ex['a']}\n\n"
    out = {'time': time.strftime('%F %T'), 'n': n,
           'gen_tokens': args.gen_tokens, 'shots': 1,
           'protocol': ('free-running strict (rule 2: budget-exhausted without '
                        'EOS counts as unfinished, no fallback); raw+pred+has_'
                        'hash+EOS+n_tokens recorded; same items/order/budget'),
           'verdict_scope': f'DIRECTIONAL (n={n}<1000)', 'runs': {}}
    print(f'B-chain freegen: n={n} budget={args.gen_tokens} 1-shot', flush=True)

    configs = [('trained:d1', 1), ('trained:d4', 4), ('base:d0', None)]
    for tag, depth in configs:
        recs, k, nf, nh = [], 0, 0, 0
        t0 = time.perf_counter()
        for i, it in enumerate(gen_items):
            prompt = shots + f"Q: {it['q']}\nA: Let's think step by step."
            p_ids = tok.encode(prompt)[:768 - args.gen_tokens - 8]
            text, finished, n_tok = generate(model, tok, p_ids,
                                             depth, args.gen_tokens)
            pred = extract_gsm8k_strict(text, finished)
            hit = (pred is not None and pred == it['gold'])
            k += int(hit)
            nf += int(finished)
            nh += int('####' in text)
            recs.append({'q': it['q'], 'gold': it['gold'], 'raw': text[:900],
                         'pred': pred, 'has_hash': '####' in text,
                         'finished': finished, 'n_tokens': n_tok,
                         'strict_correct': hit})
            print(f"  [{tag}] item{i+1}/{n}: pred={pred} gold={it['gold']} "
                  f"{'HIT' if hit else 'miss'} (hash={recs[-1]['has_hash']} "
                  f"eos={finished} {n_tok}tok) [{time.perf_counter()-t0:.0f}s]",
                  flush=True)
        out['runs'][tag] = {'depth': depth, 'strict_acc': round(k / n, 4),
                            'k': k, 'n': n, 'wilson95': wilson(k, n),
                            'finished_rate': round(nf / n, 4),
                            'hash_rate': round(nh / n, 4),
                            'sec': round(time.perf_counter() - t0, 1),
                            'items': recs}
        print(f"== {tag}: strict {k}/{n} finished {nf}/{n} hash {nh}/{n} "
              f"wilson={out['runs'][tag]['wilson95']}", flush=True)
        p = ROOT / 'results' / 's4_tier1_b_freegen.json'
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding='utf-8')

    for tag in out['runs']:
        r = out['runs'][tag]
        print(f"{tag}: strict={r['strict_acc']} finished={r['finished_rate']} "
              f"hash={r['hash_rate']}", flush=True)
    print('written results/s4_tier1_b_freegen.json', flush=True)


if __name__ == '__main__':
    main()
