"""S2 supplement: per-parameter gradient presence check (the main spike inspected
grads after zero_grad(set_to_none=True), which always shows None; the functional
evidence — finite grad_norm=6.32 + loss movement — was already in s2_spike.json).

One supervised step (B=2, K=1), backward, NO optimizer.step / zero_grad, then
inspect every trainable parameter's gradient; also confirm zero backbone grads.

Run: HF_HUB_OFFLINE=1 loopus_env/Scripts/python.exe s2_grad_check.py
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s1b_trace_v2 import get_texts, load_hybrid_nf4
from s2_spike import LoopUSQwen35

OUT = Path("results/s2_grad_check.json")


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    backbone = load_hybrid_nf4()
    model = LoopUSQwen35(backbone)
    for p in backbone.parameters():
        p.requires_grad_(False)

    ids = tok.encode(get_texts()[0])[:512]
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        h = model.encode(x)
    h_old = h.detach()
    h_prop = model.block(h_old)
    q_logit = model.q_head((h_prop - h_old)[:, -1, :].float()).squeeze(-1)
    h_new = model.gate(h_prop, h_old)
    logits = model.decode(h_new)
    lm_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                              x[:, 1:].reshape(-1))
    # q_loss must be included: q_head only receives gradient through it (as in train_step)
    with torch.no_grad():
        acc = (logits[:, :-1].argmax(-1) == x[:, 1:]).float().mean(dim=1)
    q_loss = F.binary_cross_entropy_with_logits(q_logit, acc.to(q_logit.dtype))
    loss = lm_loss + q_loss
    loss.backward()

    per_param = {}
    for n, p in list(model.gate.named_parameters()) + list(model.q_head.named_parameters()):
        g = p.grad
        per_param[n] = {"present": g is not None,
                        "finite": bool(g is not None and torch.isfinite(g).all()),
                        "absmax": round(float(g.abs().max()), 6) if g is not None else None}
    backbone_grads = sum(1 for p in backbone.parameters()
                         if p.requires_grad or p.grad is not None)
    out = {"loss": round(float(loss), 4),
           "trainable_params_total": sum(p.numel() for p in model.trainable_parameters()),
           "per_param": per_param,
           "all_trainable_have_finite_grads": all(v["present"] and v["finite"]
                                                  for v in per_param.values()),
           "backbone_requires_or_has_grad": backbone_grads,
           "verdict": "PASS" if all(v["present"] and v["finite"]
                                    for v in per_param.values())
                      and backbone_grads == 0 else "FAIL"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
