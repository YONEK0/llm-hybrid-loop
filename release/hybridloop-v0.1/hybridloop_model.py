"""HybridLoop v0.1 — adaptive-depth hybrid-attention language model.

Self-contained inference module. Load with:

    from hybridloop_model import load_release, generate
    model, router, tok, ck = load_release("hybridloop-v0.1.pt")
    text = generate(model, router, tok, "Once upon a time", max_new_tokens=128)

Architecture (see README.md): Qwen3.5-4B-Base backbone (NF4), layers 8-19 run
as a recurrent loop with identity-first pass, h0 reinjection, a wide
SelectiveGate, and a per-iteration MLP router that halts each token at its own
depth (monotonic mask, soft state decay). Trained components inherit:
LoopUS-style selective gating (arXiv 2605.11011), identity-first retrofit
(arXiv 2608.11233), AdaPonderLM-style routing (arXiv 2603.01914), plus
original dynamics-based block localization and soft-decay correction.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
LO, HI = 8, 20                    # recurrent block = layers 8..19 (S1b)
N_SWITCH = 20                     # depth-switch resolution (v3 lineage)
SIG = {}                          # layer-id -> kwargs profile cache
SOFT_EPS = 0.05                   # halted tokens keep refining at 5%


def unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def call_layer(layer, h, pe, pos_ids, cache_pos):
    profile = SIG.get(id(layer))
    attempts = []
    if profile == "full":
        attempts = [dict(attention_mask=None, position_ids=pos_ids,
                         position_embeddings=pe, cache_position=cache_pos)]
    elif profile == "mid":
        attempts = [dict(attention_mask=None, position_ids=pos_ids,
                         cache_position=cache_pos)]
    elif profile == "min":
        attempts = [dict(attention_mask=None)]
    else:
        attempts = [dict(attention_mask=None, position_ids=pos_ids,
                         position_embeddings=pe, cache_position=cache_pos),
                    dict(attention_mask=None, position_ids=pos_ids,
                         cache_position=cache_pos),
                    dict(attention_mask=None)]
    last = None
    for kw in attempts:
        try:
            out = layer(h, use_cache=False, **kw)
            SIG[id(layer)] = ("full" if "position_embeddings" in kw else
                              "mid" if "cache_position" in kw else "min")
            return unwrap(out)
        except TypeError as e:
            last = e
    raise last


def inject_lora(backbone, lo=8, hi=19, r=8):
    """peft in-place LoRA on projection linears of loop-block layers."""
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
    return len(targets)


def text_backbone(model):
    cands = []
    mm = getattr(model, "model", None)
    if mm is not None:
        cands += [getattr(mm, "language_model", None), mm]
    cands += [getattr(model, "language_model", None), model]
    for c in cands:
        if c is not None and hasattr(c, "layers"):
            return (c.layers, getattr(c, "norm", None),
                    model.get_output_embeddings(),
                    getattr(c, "rotary_emb", None))
    raise RuntimeError("cannot locate text backbone layers")


def load_backbone_nf4():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": 0}).eval()


class SelectiveGate(nn.Module):
    """LoopUS-style selective gate (verbatim math from arXiv 2605.11011)."""

    def __init__(self, hidden_size):
        super().__init__()
        dt_rank = int(torch.tensor(hidden_size / 16).ceil())
        self.dt_input_proj = nn.Linear(hidden_size, dt_rank, bias=False)
        self.delta_proj = nn.Linear(dt_rank, hidden_size, bias=True)
        A = torch.arange(1, hidden_size + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

    def forward(self, h_new, h_old):
        orig = h_new.dtype
        p = self.dt_input_proj.weight.dtype
        h_new, h_old = h_new.to(p), h_old.to(p)
        delta = F.softplus(self.delta_proj(self.dt_input_proj(h_new - h_old)))
        A_bar = torch.exp(delta * (-torch.exp(self.A_log)))
        return (A_bar * h_new + (1 - A_bar) * h_old).to(orig)


class L1Gate(nn.Module):
    """Per-depth identity switch over the SelectiveGate."""

    def __init__(self, hidden, n_switch):
        super().__init__()
        self.gate = SelectiveGate(hidden)
        init = [-4.0] * n_switch
        init[0] = 4.0                       # identity-preserving first loop
        self.s = nn.Parameter(torch.tensor(init))

    def forward(self, h_prop, h, d):
        w = torch.sigmoid(self.s[d - 1])
        g = self.gate(h_prop, h)
        return w * h_prop + (1 - w) * g


class IterGate(nn.Module):
    """Per-iteration gate (matches training-time state-dict layout)."""

    def __init__(self, hidden, h_mult):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, h_mult), nn.ReLU(),
                                 nn.Linear(h_mult, 1))
        nn.init.constant_(self.net[-1].bias, 2.0)   # start "keep going"

    def forward(self, h):
        return torch.sigmoid(self.net(h.float())).squeeze(-1)


class V4Router(nn.Module):
    """Per-iteration MLP router over the full hidden state (AdaPonderLM)."""

    def __init__(self, hidden, n_iters, h_mult=256):
        super().__init__()
        self.iters = n_iters
        self.gates = nn.ModuleList([IterGate(hidden, h_mult)
                                    for _ in range(n_iters)])

    def forward(self, h, t):
        return self.gates[t - 1](h)


class HybridLoopModel(nn.Module):
    """E (layers 0-7) -> M-loop (8-19, gated+routed) -> D (20-31) -> head."""

    def __init__(self, backbone, n_switch=N_SWITCH):
        super().__init__()
        layers, norm, lm_head, rotary = text_backbone(backbone)
        self.backbone = backbone
        self.embed = backbone.get_input_embeddings()
        self.norm = norm
        self.lm_head = lm_head
        self.rotary = rotary
        self.E, self.M, self.D = layers[:LO], layers[LO:HI], layers[HI:]
        d = int(self.norm.weight.shape[0])
        self.l1gate = L1Gate(d, n_switch)

    def _run(self, hs, h):
        pos_ids = self.pos_ids
        for l in hs:
            h = call_layer(l, h, self.pe, pos_ids, pos_ids[0])
        return h

    def encode(self, x):
        self.pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h0 = self.embed(x)
        self.pe = (self.rotary(h0, self.pos_ids)
                   if self.rotary is not None else None)
        return self._run(self.E, h0)

    def block(self, h):
        return self._run(self.M, h)

    def decode(self, h):
        h = self._run(self.D, h)
        return self.lm_head(self.norm(h))


class V4RouterStateless(V4Router):
    pass                                # router is stateless across tokens


def load_release(ckpt_path, device="cuda"):
    """Load the full model: NF4 backbone + gate/switch/LoRA/router."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = load_backbone_nf4()
    inject_lora(backbone, 8, 19, int(ck["args"].get("lora_r", 8)))
    for p in backbone.parameters():
        p.requires_grad_(False)
    model = HybridLoopModel(backbone, n_switch=ck["switch"].numel()).to(device)
    model.l1gate.gate.load_state_dict(ck["gate"])
    with torch.no_grad():
        model.l1gate.s.copy_(ck["switch"].to(model.l1gate.s.device))
        named = dict(model.backbone.named_parameters())
        for name, p in ck["lora"].items():
            named[name].copy_(p.to(named[name].device))
    ck_args = ck.get("args", {})
    r_B = int(ck_args.get("B", 8))
    r_h = int(ck_args.get("gate_h", 256))
    router = V4Router(int(model.norm.weight.shape[0]), r_B, r_h).to(device)
    router.load_state_dict(ck["router"])
    model.eval()
    router.eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    return model, router, tok, ck


@torch.no_grad()
def decode_routed(model, router, x, B, tau=1e-4, eps=SOFT_EPS):
    """Adaptive forward: each token halts at its own depth (soft decay)."""
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
def generate(model, router, tok, prompt, max_new_tokens=128, B=8,
             tau=1e-4):
    """Greedy continuation with adaptive per-token depth."""
    ids = tok.encode(prompt)
    eos = tok.eos_token_id
    new = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], device="cuda")
        logits = decode_routed(model, router, x, B, tau)
        t = int(logits[0, -1].argmax(-1).item())
        ids.append(t)
        new.append(t)
        if t == eos:
            break
    return tok.decode(new, skip_special_tokens=True)
