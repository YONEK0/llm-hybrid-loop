"""S3: L0 training run — gate + q_head only, frozen NF4 Qwen3.5-4B hybrid, M=L8-19.

Durability contract (user requirement: power cuts are expected; slow is acceptable):
  - checkpoint every --ckpt-every batches: model + AdamW + RNG + step, atomic write
    (tmp then os.replace), rolling ckpt_latest.pt / ckpt_prev.pt, <15 MB each
  - data order is deterministic from the seed (order_for(epoch)), so resume only
    needs the step counter — no data cursor in the checkpoint
  - every --eval-every batches: dev CE at d in --depths on held-out validation blocks,
    appended to history.jsonl (the depth curve survives any interruption)
  - --resume auto picks up ckpt_latest.pt and continues exactly

Run (real launch):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    loopus_env/Scripts/python.exe s3_train_l0.py
Smoke test (checkpoint/resume rehearsal):
  ... s3_train_l0.py --run-name v5_l0_smoke --max-batches 2 --ckpt-every 1 \
      --eval-every 0 --B 4 --K 1
  ... s3_train_l0.py --run-name v5_l0_smoke --max-batches 4 --ckpt-every 1 \
      --eval-every 0 --B 4 --K 1 --resume auto
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s2_spike import BETA, LR, LoopUSQwen35, train_step

ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v5_l0")
    ap.add_argument("--target-supervised-tokens", type=float, default=5e6)
    ap.add_argument("--max-batches", type=int, default=10 ** 9)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--final-depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--dev-blocks", type=int, default=8)
    ap.add_argument("--gen-samples", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=12)
    ap.add_argument("--gen-depth", type=int, default=2)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--resume", default="auto")
    return ap.parse_args()


def build_blocks(tok, split, ctx):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text)
    n = len(ids) // ctx
    flat = torch.tensor(ids[:n * ctx], dtype=torch.long, device="cuda")
    return flat.view(n, ctx), n


def order_for(epoch, n, seed):
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


@torch.no_grad()
def dev_ce(model, dev_blocks, depths, tok=None, gen_cfg=None):
    """Encode each dev block ONCE and reuse for all depths (re-encoding per depth was
    the dominant eval cost: 88 full forwards at NF4 speeds).

    Also records the S1b mechanism monitor: mean L2 norm of the looped hidden state
    after the block at each depth (frozen loops showed direction converging but norm
    re-inflating), plus optional greedy generation samples."""
    encoded = [model.encode(x) for x in dev_blocks]
    out, norms = {}, {}
    for d in depths:
        ces, hn = [], []
        for x, h0 in zip(dev_blocks, encoded):
            h = h0
            for _ in range(d):
                h = model.gate(model.block(h), h)
            hn.append(float(h[0].float().norm(dim=-1).mean().item()))
            logits = model.decode(h)
            ces.append(float(F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                x[:, 1:].reshape(-1)).item()))
        out[f"d{d}"] = round(sum(ces) / len(ces), 4)
        norms[f"d{d}"] = round(sum(hn) / len(hn), 3)
    monitor = {"block_hidden_norm": norms}
    if tok is not None and gen_cfg and gen_cfg.get("samples", 0) > 0:
        samples = []
        for i in range(gen_cfg["samples"]):
            x = dev_blocks[i % len(dev_blocks)]
            ids, text = gen_sample(model, tok, x[0].tolist(),
                                   gen_cfg.get("depth", 2),
                                   gen_cfg.get("tokens", 12))
            samples.append({"depth": gen_cfg.get("depth", 2),
                            "n_tokens": len(ids), "text": text})
        monitor["samples"] = samples
    out["monitor"] = monitor
    return out


@torch.no_grad()
def gen_sample(model, tok, prompt_ids, depth, max_new):
    """Greedy generation from the looped state (full re-forward per token: this
    pipeline has no KV cache). Returns (ids, decoded_text)."""
    ids = list(prompt_ids)
    eos = tok.eos_token_id
    for _ in range(max_new):
        x = torch.tensor([ids], device="cuda")
        h = model.encode(x)
        for _ in range(depth):
            h = model.gate(model.block(h), h)
        logits = model.decode(h)
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        if t == eos:
            break
    return ids[len(prompt_ids):], tok.decode(ids[len(prompt_ids):],
                                             skip_special_tokens=True)


def save_ckpt(path, step, sup_tokens, model, optimizer, args):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step, "sup_tokens": sup_tokens,
        "gate": model.gate.state_dict(),
        "q_head": model.q_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate()},
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)          # atomic: a power cut cannot corrupt the file


def load_ckpt(path, model, optimizer):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    model.gate.load_state_dict(ck["gate"])
    model.q_head.load_state_dict(ck["q_head"])
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["rng"]["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng"]["cuda"]])
    random.setstate(ck["rng"]["python"])
    return ck["step"], ck["sup_tokens"]


def main():
    args = parse_args()
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = run_dir / "history.jsonl"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    backbone = load_hybrid_nf4()
    for p in backbone.parameters():
        p.requires_grad_(False)
    model = LoopUSQwen35(backbone)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)

    train_blocks, n_train = build_blocks(tok, "train", args.ctx)
    dev_all, n_dev = build_blocks(tok, "validation", args.ctx)
    dev_blocks = [dev_all[i].unsqueeze(0)
                  for i in range(min(args.dev_blocks, n_dev))]
    print(f"train blocks={n_train} dev blocks={len(dev_blocks)} "
          f"ctx={args.ctx} B={args.B} K={args.K}", flush=True)

    step, sup_tokens = 0, 0
    ckpt_latest, ckpt_prev = run_dir / "ckpt_latest.pt", run_dir / "ckpt_prev.pt"
    if args.resume and ckpt_latest.exists():
        step, sup_tokens = load_ckpt(ckpt_latest, model, optimizer)
        print(f"resumed from step {step} ({sup_tokens:.0f} sup tokens)", flush=True)
    elif args.resume and ckpt_prev.exists():
        step, sup_tokens = load_ckpt(ckpt_prev, model, optimizer)
        print(f"resumed from PREV step {step}", flush=True)

    def append_history(record):
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    gen_cfg = {"samples": args.gen_samples, "tokens": args.gen_tokens,
               "depth": args.gen_depth}
    if step == 0:
        append_history({"event": "start", "args": vars(args)})
        if not args.no_baseline:
            append_history({"event": "dev_baseline",
                            "dev_ce": dev_ce(model, dev_blocks, args.final_depths,
                                             tok, gen_cfg)})

    t0, batches_done = time.perf_counter(), 0
    cur_epoch, order = -1, None
    while sup_tokens < args.target_supervised_tokens and step < args.max_batches:
        epoch = step // n_train
        if epoch != cur_epoch:
            order, cur_epoch = order_for(epoch, n_train, args.seed), epoch
        x = train_blocks[order[step % n_train]].unsqueeze(0)
        logs = train_step(model, x, x, args.B, args.K, args.beta, optimizer)
        step += 1
        batches_done += 1
        sup_tokens += args.K * args.ctx
        append_history({"step": step, "sup_tokens": sup_tokens,
                        "ts": round(time.time(), 2), "logs": logs})

        if step % args.ckpt_every == 0:
            if ckpt_latest.exists():
                if ckpt_prev.exists():
                    ckpt_prev.unlink()
                os.replace(ckpt_latest, ckpt_prev)
            save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer, args)
        if args.eval_every and step % args.eval_every == 0:
            entry = {"event": "eval", "step": step, "sup_tokens": sup_tokens,
                     "dev_ce": dev_ce(model, dev_blocks, args.depths, tok, gen_cfg),
                     "sec_per_batch": round((time.perf_counter() - t0)
                                            / max(batches_done, 1), 2)}
            append_history(entry)
            save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer, args)
            print("EVAL " + json.dumps(entry), flush=True)
            t0, batches_done = time.perf_counter(), 0

    save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer, args)
    final = {"event": "final", "step": step, "sup_tokens": sup_tokens,
             "dev_ce": dev_ce(model, dev_blocks, args.final_depths, tok, gen_cfg)}
    append_history(final)
    print("FINAL " + json.dumps(final), flush=True)
    print(f"done: {step} batches, {sup_tokens:.0f} supervised tokens", flush=True)


if __name__ == "__main__":
    main()
