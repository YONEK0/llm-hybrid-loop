"""HybridLoop — adaptive-depth hybrid-attention LM (HF-native modeling file).

This file is copied into the exported model directory. Load with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(DIR, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(DIR)
    print(tok.decode(model.generate(**tok("Once upon a time", return_tensors="pt"),
                                    max_new_tokens=64)[0]))

Architecture: a Qwen3.5-4B text backbone whose layers are split
E(0-7) / M(8-19, weight-tied recurrent loop) / D(20-31). After each loop
iteration a per-iteration MLP router scores every token; tokens below tau halt
(monotonic mask, soft state decay keeps 5% refinement). Set config.adaptive_exit
= False (or model.set_exit_depth(N)) for fixed-depth decoding.

No KV cache: every generated token re-runs the full loop (documented
limitation). Requires transformers with qwen3_5 support.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


class HybridLoopConfig(PretrainedConfig):
    model_type = "hybridloop"

    def __init__(self, text_config=None, loop_lo=8, loop_hi=20, n_switch=20,
                 max_depth=8, gate_hidden=256, soft_eps=0.05, tau=1e-4,
                 adaptive_exit=True, **kwargs):
        if isinstance(text_config, dict):
            text_config = Qwen3_5TextConfig(**text_config)
        self.text_config = text_config
        self.loop_lo = loop_lo
        self.loop_hi = loop_hi
        self.n_switch = n_switch
        self.max_depth = max_depth
        self.gate_hidden = gate_hidden
        self.soft_eps = soft_eps
        self.tau = tau
        self.adaptive_exit = adaptive_exit
        super().__init__(**kwargs)


class SelectiveGate(nn.Module):
    """LoopUS selective gate (arXiv 2605.11011), fp32 params."""

    def __init__(self, hidden_size):
        super().__init__()
        dt_rank = math.ceil(hidden_size / 16)
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
    def __init__(self, hidden, n_switch):
        super().__init__()
        self.gate = SelectiveGate(hidden)
        init = [-4.0] * n_switch
        init[0] = 4.0
        self.s = nn.Parameter(torch.tensor(init))

    def forward(self, h_prop, h, d):
        w = torch.sigmoid(self.s[d - 1])
        return w * h_prop + (1 - w) * self.gate(h_prop, h)


class IterGate(nn.Module):
    def __init__(self, hidden, h_mult):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, h_mult), nn.ReLU(),
                                 nn.Linear(h_mult, 1))

    def forward(self, h):
        return torch.sigmoid(self.net(h.to(self.net[0].weight.dtype))).squeeze(-1).to(h.dtype)


class V4Router(nn.Module):
    def __init__(self, hidden, n_iters, h_mult=256):
        super().__init__()
        self.gates = nn.ModuleList([IterGate(hidden, h_mult)
                                    for _ in range(n_iters)])

    def forward(self, h, t):
        return self.gates[t - 1](h)


class HybridLoopForCausalLM(PreTrainedModel):
    config_class = HybridLoopConfig

    def __init__(self, config):
        super().__init__(config)
        tc = config.text_config
        self.model = Qwen3_5TextModel(tc)
        self.vocab_size = tc.vocab_size
        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
        d = tc.hidden_size
        self.l1gate = L1Gate(d, config.n_switch)
        self.router = V4Router(d, config.max_depth, config.gate_hidden)
        self._fixed_depth = None
        self.post_init()

    # ---- helpers -------------------------------------------------------
    def _run(self, layers, h, position_embeddings, position_ids):
        for l in layers:
            try:
                out = l(h, attention_mask=None, position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        cache_position=position_ids[0], use_cache=False)
            except TypeError:
                out = l(h, attention_mask=None, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def set_exit_depth(self, n):
        """Fixed-depth decoding (disables adaptive routing). n=None -> adaptive."""
        self._fixed_depth = n

    def _loop(self, h, depth, eps, tau):
        h0 = h
        mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        for t in range(1, depth + 1):
            inp = h if t == 1 else h + h0
            h_prop = self._run(self.M, inp, self._pe, self._pos)
            h_new = self.l1gate(h_prop, h, t)
            s = self.router(h_new, t)
            keep = ((s >= tau).float() * mask).to(h.dtype)
            h = h + keep.unsqueeze(-1) * (h_new - h) + (
                1 - keep.unsqueeze(-1)) * eps * (h_new - h)
            mask = keep
        return h

    # ---- forward -------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=0,
                **kw):  # attention_mask accepted: causal, no padding
        _ = attention_mask  # causal attention, no padding in this model
        cfg = self.config
        layers = self.model.layers
        self.E, self.M, self.D = (layers[:cfg.loop_lo], layers[cfg.loop_lo:cfg.loop_hi],
                                  layers[cfg.loop_hi:])
        h = (self.model.embed_tokens(input_ids) if inputs_embeds is None
             else inputs_embeds)
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        self._pos = position_ids
        self._pe = (self.model.rotary_emb(h, position_ids)
                    if getattr(self.model, "rotary_emb", None) is not None
                    else None)
        h = self._run(self.E, h, self._pe, position_ids)
        depth = self._fixed_depth or cfg.max_depth
        h = self._loop(h, depth, cfg.soft_eps, cfg.tau)
        h = self._run(self.D, h, self._pe, position_ids)
        h = self.model.norm(h)
        logits = self.lm_head(h)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
            labels = labels[:, -logits_to_keep:] if labels is not None else None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None,
                                      **kw):
        # no KV cache: always feed the full prefix; mask is causal, unused
        inputs = {"input_ids": input_ids, "use_cache": False}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        return inputs


def register():
    """Optional: register for AutoModelForCausalLM without trust_remote_code."""
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register(HybridLoopConfig.model_type, HybridLoopConfig)
    AutoModelForCausalLM.register(HybridLoopConfig, HybridLoopForCausalLM)
