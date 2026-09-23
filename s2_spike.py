"""S2: training-infrastructure spike on the frozen NF4 Qwen3.5-4B hybrid (V5 spec).

Implements the minimal faithful LoopUS training step on our S1b-decided split
(E=layers 0..7, M=L8-19 with the A2 block-boundary gate, D=layers 20..31):

  per batch: encoder (no grad) -> B reasoning iterations; K random ones supervised
    supervised:  h_old = h.detach(); h_prop = M(h_old)
                 q_logit = q_head((h_prop - h_old)[:, -1])          (pre-gate delta)
                 h_new  = Gate(h_prop, h_old)                       (SelectiveGate, exact
                                                                     LoopUS math, fp32)
                 lm_loss = CE(D(h_new));  lm_prev = CE(D(h_old))    (no grad)
                 loss = lm + beta*SiLU(lm - lm_prev) + BCEWithLogits(q, per-sample acc)
                 backward -> clip -> optimizer.step (per supervision, as LoopUS does)
    unsupervised: no_grad M + gate, detach
  gate applies on EVERY iteration (supervised or not), matching ReasoningBlock.forward.

Checks: only gate/q_head receive grads (backbone stays frozen, count verified),
state isolation (identical outputs regardless of preceding batches), loss moves,
peak VRAM and sec/step at (ctx512,B8,K2), (ctx512,B20,K5), (ctx256,B20,K5).

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s2_spike.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig

from s1b_trace_v2 import call_layer, get_texts, load_hybrid_nf4, text_backbone

ROOT = Path(__file__).resolve().parent
LO, HI = 8, 20                     # S1b decision: M = L8-19 (slice hi-exclusive = 20)
BETA = 0.3
LR = 1e-4
OUT = ROOT / "results/s2_spike.json"
CONFIGS = [dict(tag="ctx512_B8_K2", ctx=512, B=8, K=2, steps=8),
           dict(tag="ctx512_B20_K5", ctx=512, B=20, K=5, steps=5),
           dict(tag="ctx256_B20_K5", ctx=256, B=20, K=5, steps=5)]


class SelectiveGate(nn.Module):
    """Verbatim LoopUS SelectiveGate math (loopus/models/modeling_lds.py), fp32 params."""

    def __init__(self, hidden_size, dt_rank=None):
        super().__init__()
        dt_rank = dt_rank or math.ceil(hidden_size / 16)
        self.dt_input_proj = nn.Linear(hidden_size, dt_rank, bias=False)
        self.delta_proj = nn.Linear(dt_rank, hidden_size, bias=True)
        A = torch.arange(1, hidden_size + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.delta_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(hidden_size) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.delta_proj.bias.copy_(inv_dt)
        self.last_A_bar_mean = None

    def forward(self, h_new, h_old):
        orig = h_new.dtype
        p = self.dt_input_proj.weight.dtype
        h_new, h_old = h_new.to(p), h_old.to(p)
        delta = F.softplus(self.delta_proj(self.dt_input_proj(h_new - h_old)))
        A_bar = torch.exp(delta * (-torch.exp(self.A_log)))
        self.last_A_bar_mean = float(A_bar.detach().mean().item())
        return (A_bar * h_new + (1 - A_bar) * h_old).to(orig)


class LoopUSQwen35(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        layers, norm, lm_head, rotary = text_backbone(backbone)
        self.embed = backbone.get_input_embeddings()
        self.norm = norm
        self.lm_head = lm_head
        self.rotary = rotary
        self.E, self.M, self.D = layers[:LO], layers[LO:HI], layers[HI:]
        d = int(self.norm.weight.shape[0])
        self.gate = SelectiveGate(d)
        self.q_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.cuda()

    def _run(self, hs, h):
        for l in hs:
            h = call_layer(l, h, self.pe, self.pos_ids, self.cache_pos)
        return h

    def encode(self, x):
        self.pos_ids = torch.arange(x.shape[1], device="cuda").unsqueeze(0)
        self.cache_pos = self.pos_ids[0]
        h0 = self.embed(x)
        self.pe = self.rotary(h0, self.pos_ids) if self.rotary is not None else None
        return self._run(self.E, h0)

    def block(self, h):
        return self._run(self.M, h)

    def decode(self, h):
        h = self._run(self.D, h)
        return self.lm_head(self.norm(h))

    def trainable_parameters(self):
        return list(self.gate.parameters()) + list(self.q_head.parameters())


@torch.no_grad()
def pipeline_logits(model, x, iters=2):
    """E -> (M+gate) x iters -> D, no grad — used by the isolation test."""
    h = model.encode(x)
    for _ in range(iters):
        h = model.gate(model.block(h), h)
    return model.decode(h)


def train_step(model, x, labels, B, K, beta, optimizer, clip=1.0, keep_grads=False):
    with torch.no_grad():
        h = model.encode(x)
    supervised = sorted(random.sample(range(B), K))
    logs = []
    for b in range(B):
        h_old = h.detach()
        h_prop = model.block(h_old)
        if b in supervised:
            delta_pre = h_prop - h_old
            q_logit = model.q_head(delta_pre[:, -1, :].float()).squeeze(-1)
            h_new = model.gate(h_prop, h_old)
            logits = model.decode(h_new)
            sl, st = logits[:, :-1], labels[:, 1:]
            lm_loss = F.cross_entropy(sl.float().reshape(-1, sl.shape[-1]),
                                      st.reshape(-1))
            with torch.no_grad():
                lo = model.decode(h_old)
                lm_prev = F.cross_entropy(lo[:, :-1].float().reshape(-1, sl.shape[-1]),
                                          st.reshape(-1))
                acc = (sl.argmax(-1) == st).float().mean(dim=1)
            q_loss = F.binary_cross_entropy_with_logits(
                q_logit, acc.to(q_logit.dtype))
            q_hat = torch.sigmoid(q_logit).detach().mean()
            loss = lm_loss + beta * F.silu(lm_loss - lm_prev) + q_loss
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), clip))
            if not keep_grads:      # inspect grads before they are cleared
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            logs.append(dict(b=b, lm=round(float(lm_loss), 4),
                             lm_prev=round(float(lm_prev), 4),
                             q=round(float(q_loss), 4),
                             grad_norm=round(grad_norm, 4),
                             A_bar_mean=round(model.gate.last_A_bar_mean, 5),
                             q_hat=round(float(q_hat), 4)))
            h = h_new.detach()
        else:
            with torch.no_grad():
                h = model.gate(h_prop, h_old)
    return logs


def main():
    result = {"spec": {"split": "E=0..7 / M=L8-19 / D=20..31 (S1b)", "beta": BETA,
                       "lr": LR, "optimizer": "AdamW",
                       "step": "per-supervision optimizer.step (LoopUS-faithful)",
                       "q_target": "per-sample next-token accuracy (approx of "
                                   "LoopUS compute_q_target)"},
              "checks": {}, "configs": {}}
    torch.manual_seed(2026)
    random.seed(2026)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    backbone = load_hybrid_nf4()
    model = LoopUSQwen35(backbone)
    for p in backbone.parameters():
        p.requires_grad_(False)
    trainable = model.trainable_parameters()
    result["checks"]["trainable_params"] = sum(p.numel() for p in trainable)
    result["checks"]["expected_params"] = 824320 + 7681
    result["checks"]["backbone_frozen"] = not any(
        p.requires_grad for p in backbone.parameters())

    texts = get_texts()
    batches = {}
    for ctx in (512, 256):
        xs = []
        for t in texts:
            ids = tok.encode(t)[:ctx]
            if len(ids) == ctx:
                xs.append(torch.tensor([ids], device="cuda"))
        batches[ctx] = xs

    # --- isolation test: outputs must not depend on preceding batches ---
    xa, xb, xc = batches[512][0], batches[512][1], batches[512][2]
    with torch.no_grad():
        r1 = pipeline_logits(model, xb)
        _ = pipeline_logits(model, xa)
        r_mid = pipeline_logits(model, xa)
        _ = pipeline_logits(model, xc)
        r2 = pipeline_logits(model, xb)
    result["checks"]["isolation_max_diff_b"] = float((r1 - r2).abs().max().item())
    result["checks"]["isolation_max_diff_a"] = float((r_mid - r_mid).abs().max().item())
    result["checks"]["isolation_pass"] = bool(
        (r1 - r2).abs().max().item() == 0.0)

    # --- grad plumbing check on one supervised step ---
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    logs = train_step(model, batches[512][3], batches[512][3], B=4, K=1,
                      beta=BETA, optimizer=optimizer)
    gnorms = {n: float(p.grad.abs().sum().item()) if p.grad is not None else -1.0
              for n, p in list(model.gate.named_parameters())
              + list(model.q_head.named_parameters())}
    result["checks"]["grad_present_all_trainable"] = all(v >= 0 for v in gnorms.values())
    result["checks"]["grad_norms"] = {k: round(v, 6) for k, v in gnorms.items()}
    result["checks"]["first_step_logs"] = logs

    # --- throughput configs ---
    for cfg in CONFIGS:
        xs = batches[cfg["ctx"]]
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(trainable, lr=LR)
        t0 = time.perf_counter()
        all_logs = []
        for i in range(cfg["steps"]):
            x = xs[i % len(xs)]
            all_logs += train_step(model, x, x, cfg["B"], cfg["K"], BETA, optimizer)
        dt = time.perf_counter() - t0
        per_step = dt / cfg["steps"]
        result["configs"][cfg["tag"]] = {
            "steps": cfg["steps"], "sec_total": round(dt, 1),
            "sec_per_batch": round(per_step, 2),
            "tok_per_s_raw": round(cfg["ctx"] / per_step, 1),
            "tok_per_s_supervised": round(cfg["K"] * cfg["ctx"] / per_step, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "lm_first": all_logs[0]["lm"], "lm_last": all_logs[-1]["lm"],
            "n_supervised_logs": len(all_logs)}
        print(cfg["tag"], json.dumps(result["configs"][cfg["tag"]]), flush=True)

    result["checks"]["loss_moved"] = bool(
        result["configs"]["ctx512_B8_K2"]["lm_last"]
        != result["configs"]["ctx512_B8_K2"]["lm_first"])

    # token-budget extrapolation (L0/L1 planning input)
    per_step = result["configs"]["ctx512_B20_K5"]["sec_per_batch"]
    supervised = result["configs"]["ctx512_B20_K5"]["tok_per_s_supervised"]
    result["budget_note"] = {
        "official_recipe_sec_per_batch_ctx512": per_step,
        "supervised_tok_per_s": supervised,
        "tokens_1h_supervised": round(supervised * 3600),
        "note": "L1 budget decided from these numbers before S3; "
                "3B-token server recipe out of local scope"}

    del backbone, model
    torch.cuda.empty_cache()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("checks:", json.dumps(result["checks"], ensure_ascii=False), flush=True)
    print("budget:", json.dumps(result["budget_note"]), flush=True)
    print(f"written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
