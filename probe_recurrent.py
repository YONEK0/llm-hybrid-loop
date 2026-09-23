import json
from pathlib import Path
import time
import traceback
from collections import defaultdict

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, read_jsonl, setup

setup(0)
result = {"technique": "recurrent depth: weight-tied block looped K times, per-iteration low-rank deltas, deep supervision"}
try:
    start = time.perf_counter()
    model = RecurrentDepthQwen(train_depth=4, block_start=24, block_len=6, rank=8)
    result["load_seconds"] = round(time.perf_counter() - start, 1)

    # --- parameter accounting ---
    groups = defaultdict(int)
    trainable = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
            groups[name.split(".")[0] + "." + (name.split(".")[1] if "." in name else "")] += parameter.numel()
    result["trainable_total_m"] = round(trainable / 1e6, 3)
    result["trainable_by_group"] = {k: round(v / 1e6, 3) for k, v in sorted(groups.items())[:10]}
    result["n_delta_sets"] = len(model.deltas)
    result["block_layers"] = len(model.block)
    result["stack_len"] = len(model.all_layers)

    rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[:2]
    ids, prefix, target = model.batch(rows[0])
    result["sample"] = {"ids": len(ids), "prefix": len(prefix), "target": len(target)}

    # --- forward at several depths ---
    for depth in (1, 2, 4):
        input_ids = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            logits, marks = model.forward(input_ids, depth=depth, collect_iterations=True)
        result[f"forward_depth{depth}"] = {"logits": list(logits.shape), "iteration_marks": len(marks)}

    # --- one training step at depth 4 ---
    params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    total, report = model.loss(rows[0], aux_weight=0.2, depth=4)
    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    result["train_step"] = {"ok": True, "loss": round(float(total.detach()), 4), "report": report,
                            "grad_norm": round(grad_norm, 4),
                            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                            "seconds": round(time.perf_counter() - start, 1)}

    # --- do the deltas actually move? ---
    before = [p.detach().clone() for p in params[:4]]
    total2, _ = model.loss(rows[1], aux_weight=0.2, depth=4)
    total2.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    moved = [float((b - p.detach()).abs().max()) for b, p in zip(before, params[:4])]
    result["delta_change_after_step"] = [round(m, 8) for m in moved]
    result["weights_moved"] = all(m > 0 for m in moved)

    # --- inference depth extrapolation (trained at 4) ---
    curve = model.answer_at_depth(rows[0]["question"], depths=(1, 2, 4, 8), max_new_tokens=16)
    result["depth_curve"] = curve
    result["gold"] = rows[0]["answer"]
except Exception as exc:
    result["error"] = repr(exc)
    result["traceback"] = traceback.format_exc()

print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "probe_recurrent.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
