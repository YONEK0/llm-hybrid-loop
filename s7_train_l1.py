"""s7_train_l1.py — L1 v2 training: hybrid-loop post-training, checkpoint-resilient.

Upgrades over L0 (every item traces to a measured tier-1/S6 lesson):
1. h0 re-injection from loop 2 on  (Block(H^{t-1}+H^0), Fan-style)  <- B-chain repetition
2. identity-preserving first loop via per-depth switch s_d          <- +0.068 startup tax
   out = sig(s_d)*h_prop + (1-sig(s_d))*SelectiveGate(h_prop, h);
   s_1 init +4 (loop1 ~ full block pass ~ base), s_{d>1} init -4 (gated refinement)
3. loop-block LoRA (peft inject, r=8, layers 8..19 projections)     <- no depth returns
4. beta warmup 0.3 -> 1.0 over first 300 supervised steps           <- weak monotonicity
5. domain-mixed data: WikiText 70% + GSM8K-train CoT 30% (FineWeb-Edu unreachable)
6. milestone checkpoints 0.5/1/2/3/5M (never overwritten)           <- 3M ckpt loss
7. pathology probe (rep-rate) + GSM8K task CE probe in evals       <- late discovery

Checkpoint style identical to s3_train_l0: atomic write, rolling pair, seeded
deterministic data order, RNG-state restore. run dir: runs/v5_l1/
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoTokenizer

from s1b_trace_v2 import load_hybrid_nf4
from s2_spike import SelectiveGate, LoopUSQwen35

ROOT = Path(__file__).resolve().parent
# permanent archives: evaluation turning points only (T1/T2/...); no fixed-step archives


class L1Gate(nn.Module):
    """SelectiveGate + per-depth identity switch (see docstring items 1-2)."""

    def __init__(self, hidden, B=20):
        super().__init__()
        self.gate = SelectiveGate(hidden)
        init = [-4.0] * B
        init[0] = 4.0
        self.s = nn.Parameter(torch.tensor(init))
        self.last_w = None

    def forward(self, h_prop, h, d):
        w = torch.sigmoid(self.s[d - 1])
        self.last_w = float(w.item())
        g = self.gate(h_prop, h)
        return w * h_prop + (1 - w) * g


def loop_state(model, x, depth):
    """L1 forward semantics: loop1 plain block pass, loops>=2 reinject h0."""
    h = model.encode(x)
    h0 = h
    for t in range(1, depth + 1):
        inp = h if t == 1 else h + h0
        h = model.l1gate(model.block(inp), h, t)
    return h
def build_wiki_blocks(tok, split, ctx):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text)
    n = len(ids) // ctx
    flat = torch.tensor(ids[:n * ctx], dtype=torch.long, device="cuda")
    return flat.view(n, ctx), n


def build_gsm_blocks(tok, ctx, split_file="train"):
    items = []
    with open(ROOT / "data" / "gsm8k" / f"{split_file}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append(f"Q: {r['question']}\nA: Let's think step by step. "
                         f"{r['rationale']}\n#### {r['answer']}\n\n")
    ids = tok.encode("".join(items))
    n = len(ids) // ctx
    flat = torch.tensor(ids[:n * ctx], dtype=torch.long, device="cuda")
    return flat.view(n, ctx), n


def build_gsm_probe(tok, n_items=32, ctx=512):
    """Held-out teacher-forced task probe: test-split items, answer-segment CE."""
    corpus = []
    with open(ROOT / "data" / "gsm8k" / "test.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_items:
                break
            r = json.loads(line)
            p_ids = tok.encode(r["question"] + "\n\n")
            ids = (p_ids + tok.encode(r["rationale"]
                     + "\n#### " + r["answer"]))[:ctx]
            if len(ids) > len(p_ids) + 4:
                corpus.append((ids, len(p_ids)))
    return corpus


@torch.no_grad()
def task_ce(model, corpus, depth):
    ces = []
    for ids, ms in corpus:
        x = torch.tensor([ids], device="cuda")
        logits = model.decode(loop_state(model, x, depth))
        seg = logits[0, ms - 1:-1].float()
        tgt = x[0, ms:]
        ces.append(float(F.cross_entropy(seg, tgt.reshape(-1),
                                         reduction="sum").item() / tgt.numel()))
    return round(sum(ces) / len(ces), 4)


def rep_metrics(text):
    """Pathology probe: distinct-2 ratio + max repeated 4-gram count."""
    ws = text.split()
    if len(ws) < 5:
        return {"distinct2": 1.0, "rep4": 0}
    big = list(zip(ws, ws[1:]))
    d2 = len(set(big)) / max(1, len(big))
    quad = list(zip(ws, ws[1:], ws[2:], ws[3:]))
    cnt = {}
    for q in quad:
        cnt[q] = cnt.get(q, 0) + 1
    return {"distinct2": round(d2, 4), "rep4": max(cnt.values())}


@torch.no_grad()
def gen_l1(model, tok, prompt_ids, depth, max_new):
    ids = list(prompt_ids)
    eos = tok.eos_token_id
    for _ in range(max_new):
        x = torch.tensor([ids], device="cuda")
        logits = model.decode(loop_state(model, x, depth))
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        if t == eos:
            break
    new = ids[len(prompt_ids):]
    return new, tok.decode(new, skip_special_tokens=True)


@torch.no_grad()
def dev_ce_l1(model, dev_blocks, depths, tok=None, gen_cfg=None):
    out, norms = {}, {}
    for d in depths:
        ces, hn = [], []
        for x in dev_blocks:
            h = loop_state(model, x, d)
            hn.append(float(h[0].float().norm(dim=-1).mean().item()))
            logits = model.decode(h)
            ces.append(float(F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                x[:, 1:].reshape(-1)).item()))
        out[f"d{d}"] = round(sum(ces) / len(ces), 4)
        norms[f"d{d}"] = round(sum(hn) / len(hn), 3)
    monitor = {"block_hidden_norm": norms,
               "switch_w": {f"d{d}": round(float(torch.sigmoid(model.l1gate.s[d-1]).item()), 3)
                            for d in depths}}
    if tok is not None and gen_cfg and gen_cfg.get("samples", 0) > 0:
        samples = []
        for i in range(gen_cfg["samples"]):
            x = dev_blocks[i % len(dev_blocks)]
            _, text = gen_l1(model, tok, x[0].tolist(),
                             gen_cfg.get("depth", 2), gen_cfg.get("tokens", 12))
            samples.append({"depth": gen_cfg.get("depth", 2),
                            "n_tokens": gen_cfg.get("tokens", 12), "text": text})
        monitor["samples"] = samples
    out["monitor"] = monitor
    return out


def inject_lora(backbone, lo=8, hi=19, r=8):
    """peft in-place LoRA on projection linears of loop-block layers [lo, hi]."""
    targets = []
    for name, mod in backbone.named_modules():
        parts = name.split(".")
        if (len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers"
                and parts[2].isdigit() and lo <= int(parts[2]) <= hi
                and name.endswith(("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj",
                                   "in_proj", "out_proj"))):
            targets.append(name)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                     target_modules=targets)
    inject_adapter_in_model(cfg, backbone)
    params = [p for n, p in backbone.named_parameters()
              if "lora_" in n and p.requires_grad]
    return params, len(targets)


def train_step_l1(model, x, B, K, beta, optimizer, trainables, clip=1.0):
    """Deep supervision as L0, minus q_loss; L1 loop semantics; per-depth beta."""
    with torch.no_grad():
        h0 = model.encode(x)
    h = h0
    supervised = sorted(random.sample(range(B), K))
    labels = x
    logs = []
    for b in range(B):
        t = b + 1
        h_old = h.detach()
        inp = h_old if t == 1 else h_old + h0
        h_prop = model.block(inp)
        if b in supervised:
            h_new = model.l1gate(h_prop, h_old, t)
            logits = model.decode(h_new)
            sl, st_ = logits[:, :-1], labels[:, 1:]
            lm = F.cross_entropy(sl.float().reshape(-1, sl.shape[-1]),
                                 st_.reshape(-1))
            with torch.no_grad():
                lo_ = model.decode(h_old)[:, :-1]
                lm_prev = F.cross_entropy(
                    lo_.float().reshape(-1, sl.shape[-1]), st_.reshape(-1))
            loss = lm + beta * F.silu(lm - lm_prev)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(trainables, clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            logs.append(dict(b=b, lm=round(float(lm), 4),
                             lm_prev=round(float(lm_prev), 4),
                             grad_norm=round(gn, 4),
                             A_bar_mean=round(model.l1gate.gate.last_A_bar_mean, 5),
                             switch_w=round(model.l1gate.last_w, 4),
                             beta=round(beta, 3)))
            h = h_new.detach()
        else:
            with torch.no_grad():
                h = model.l1gate(h_prop, h_old, t)
    return logs


def lora_state(backbone):
    return {n: p.detach().cpu() for n, p in backbone.named_parameters()
            if "lora_" in n}


def save_ckpt(path, step, sup_tokens, model, optimizer, args, turns):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step, "sup_tokens": sup_tokens,
        "gate": model.l1gate.gate.state_dict(),
        "switch": model.l1gate.s.detach().cpu(),
        "lora": lora_state(model.backbone),
        "optimizer": optimizer.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate()},
        "turns": turns, "args": vars(args),
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, optimizer):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    model.l1gate.gate.load_state_dict(ck["gate"])
    with torch.no_grad():
        model.l1gate.s.copy_(ck["switch"].to(model.l1gate.s.device))
        cur = lora_state(model.backbone)
        for n, p in ck["lora"].items():
            cur[n].copy_(p.to(cur[n].device))
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["rng"]["torch"].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng"]["cuda"]])
    random.setstate(ck["rng"]["python"])
    turns = [m for m in ck.get("turns", [])
             if isinstance(m, str)]
    return ck["step"], ck["sup_tokens"], turns


class L1Qwen35(LoopUSQwen35):
    """LoopUSQwen35 with L1Gate replacing the plain gate (E/M/D machinery reused)."""

    def __init__(self, backbone, B=20):
        super().__init__(backbone)
        self.backbone = backbone
        d = int(self.norm.weight.shape[0])
        self.l1gate = L1Gate(d, B).cuda()


def src_is_gsm(step):
    """Deterministic 30% GSM8K interleave pattern (resume-safe: pure fn of step)."""
    return (step * 7 + 3) % 10 < 3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v5_l1")
    ap.add_argument("--target-supervised-tokens", type=float, default=5e6)
    ap.add_argument("--max-batches", type=int, default=10 ** 9)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=300)
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--gate-lr", type=float, default=1e-4)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--beta-warmup-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--final-depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--dev-blocks", type=int, default=8)
    ap.add_argument("--gen-samples", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=12)
    ap.add_argument("--gen-depth", type=int, default=2)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--resume", default="auto")
    return ap.parse_args()


def order_for(epoch, n, seed):
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


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
    lora_params, n_targets = inject_lora(backbone, 8, 19, args.lora_r)
    model = L1Qwen35(backbone, args.B)
    model.cuda()
    gate_params = list(model.l1gate.parameters())
    trainables = gate_params + lora_params
    optimizer = torch.optim.AdamW(
        [{"params": gate_params, "lr": args.gate_lr},
         {"params": lora_params, "lr": args.lora_lr}])
    n_gate = sum(p.numel() for p in gate_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"trainable: gate+switch {n_gate:,} + lora {n_lora:,} "
          f"({n_targets} modules r={args.lora_r})", flush=True)

    wiki_blocks, n_wiki = build_wiki_blocks(tok, "train", args.ctx)
    gsm_blocks, n_gsm = build_gsm_blocks(tok, args.ctx, "train")
    dev_all, n_dev = build_wiki_blocks(tok, "validation", args.ctx)
    dev_blocks = [dev_all[i].unsqueeze(0)
                  for i in range(min(args.dev_blocks, n_dev))]
    gsm_probe = build_gsm_probe(tok)
    print(f"pools: wiki {n_wiki} / gsm {n_gsm} blocks | dev {len(dev_blocks)} "
          f"| task probe {len(gsm_probe)} items", flush=True)

    step, sup_tokens, turns = 0, 0, []
    ckpt_latest = run_dir / "ckpt_latest.pt"
    ckpt_prev = run_dir / "ckpt_prev.pt"
    if args.resume != "none" and ckpt_latest.exists():
        step, sup_tokens, turns = load_ckpt(ckpt_latest, model, optimizer)
        if history_path.exists():          # turns are facts of THIS run's evals
            for _line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    _r = json.loads(_line)
                except Exception:
                    continue
                if _r.get("event") == "turn" and _r.get("name") not in turns:
                    turns = turns + [_r["name"]]
        print(f"resumed from step {step} ({sup_tokens:.0f} sup tokens, "
              f"turns {turns})", flush=True)

    def append_history(rec):
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # deterministic data position: replay the source pattern up to `step`
    cw = cg = 0
    for s0 in range(step):
        if src_is_gsm(s0):
            cg += 1
        else:
            cw += 1

    d4_hist = []
    if history_path.exists():
        for _line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                _r = json.loads(_line)
            except Exception:
                continue
            if _r.get("event") == "eval":
                _v = _r.get("dev_ce", {}).get("d4")
                if _v is not None:
                    d4_hist.append(_v)
        d4_hist = d4_hist[-6:]

    gen_cfg = {"samples": args.gen_samples, "tokens": args.gen_tokens,
               "depth": args.gen_depth}
    if step == 0:
        append_history({"event": "start", "args": vars(args)})
        if not args.no_baseline:
            append_history({"event": "dev_baseline",
                            "dev_ce": dev_ce_l1(model, dev_blocks,
                                                args.final_depths, tok, gen_cfg)})

    t0, batches_done = time.perf_counter(), 0
    ep_w, ord_w = -1, None
    ep_g, ord_g = -1, None
    while sup_tokens < args.target_supervised_tokens and step < args.max_batches:
        beta = min(1.0, 0.3 + 0.7 * step / max(1, args.beta_warmup_steps))
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
        src = "gsm" if src_is_gsm(step) else "wiki"
        logs = train_step_l1(model, x, args.B, args.K, beta, optimizer, trainables)
        step += 1
        batches_done += 1
        sup_tokens += args.K * args.ctx
        append_history({"step": step, "sup_tokens": sup_tokens, "src": src,
                        "ts": round(time.time(), 2), "logs": logs})

        if step % args.ckpt_every == 0:
            if ckpt_latest.exists():
                if ckpt_prev.exists():
                    ckpt_prev.unlink()
                os.replace(ckpt_latest, ckpt_prev)
            save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer,
                      args, turns)
        if args.eval_every and step % args.eval_every == 0:
            entry = {"event": "eval", "step": step, "sup_tokens": sup_tokens,
                     "dev_ce": dev_ce_l1(model, dev_blocks, args.depths,
                                         tok, gen_cfg),
                     "sec_per_batch": round((time.perf_counter() - t0)
                                            / max(batches_done, 1), 2)}
            if step % args.probe_every == 0:         # pathology + task probes
                txts = []
                for i in range(2):
                    xb = dev_blocks[i]
                    _, tx = gen_l1(model, tok, xb[0].tolist()[:400], 2, 48)
                    txts.append(tx)
                entry["rep_probe"] = {f"p{i}": rep_metrics(t) for i, t in enumerate(txts)}
                entry["gsm8k_ce"] = {f"d{d}": task_ce(model, gsm_probe, d)
                                     for d in (1, 4)}
            _dc = entry["dev_ce"]
            d4_hist.append(_dc.get("d4"))
            d4_hist = [x for x in d4_hist if x is not None][-6:]
            t1_ok = ("t1_inversion" not in turns
                     and _dc.get("d4", 9.0) < _dc.get("d1", 0.0) - 0.005)
            t2_ok = ("t1_inversion" in turns
                     and "t2_mature" not in turns
                     and sup_tokens >= 1_500_000 and len(d4_hist) >= 4
                     and d4_hist[-4] - d4_hist[-1] < 0.02)
            for name, ok, fname in (
                    ("t1_inversion", t1_ok, "ckpt_turn1_inversion.pt"),
                    ("t2_mature", t2_ok, "ckpt_turn2_mature.pt")):
                if ok:
                    turns = turns + [name]
                    save_ckpt(run_dir / fname, step, sup_tokens, model,
                              optimizer, args, turns)
                    append_history({"event": "turn", "name": name,
                                    "step": step, "sup_tokens": sup_tokens})
                    print(f"TURNING-POINT {name} saved @ step {step}",
                          flush=True)
            append_history(entry)
            save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer,
                      args, turns)
            print("EVAL " + json.dumps(entry), flush=True)
            t0, batches_done = time.perf_counter(), 0

    save_ckpt(ckpt_latest, step, sup_tokens, model, optimizer, args, turns)
    final = {"event": "final", "step": step, "sup_tokens": sup_tokens,
             "dev_ce": dev_ce_l1(model, dev_blocks, args.final_depths, tok, gen_cfg)}
    append_history(final)
    print("FINAL " + json.dumps(final), flush=True)
    print(f"done: {step} batches, {sup_tokens:.0f} supervised tokens", flush=True)


if __name__ == "__main__":
    main()
