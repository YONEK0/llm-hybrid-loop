"""Decisive control: at depth 1, do the trained deltas matter at all?

If zeroed-deltas at depth 1 scores the same as trained at depth 1, then the depth-1
accuracy comes from the frozen base model reading the scaffold, not from anything we
trained.  This distinguishes "our training helped" from "the base model was already
able to finish the problem once the first k steps were given".
"""

import json
import random

import torch
import torch.nn.functional as F

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup


@torch.no_grad()
def score(model, row, keep_steps, depth):
    prompt = model.prefix(row["question"])
    scaffold = model.scaffold_tokens(row, keep_steps)
    answer = [int(t) for t in model.tokenizer.encode(
        "</think>\n\n#### " + row["answer"], add_special_tokens=False)] + [model.tokenizer.eos_token_id]
    logits, _ = model.forward_split(prompt, scaffold, answer[:-1], depth=depth)
    absolute_start = len(prompt) if scaffold else len(prompt) - 1
    span_len = (len(scaffold) - 1 if scaffold else 0) + len(answer)
    labels = torch.tensor([scaffold[1:] + answer], dtype=torch.long, device="cuda").reshape(-1)
    lg = logits[:, absolute_start:absolute_start + span_len].reshape(-1, logits.shape[-1])
    ce = float(F.cross_entropy(lg, labels))
    text = model.tokenizer.decode(lg[-len(answer):].argmax(-1).tolist(), skip_special_tokens=True)
    return ce, answer_from_text(text) == str(row["answer"]).strip()


setup(11)
model = RecurrentDepthQwen(checkpoint=ROOT / "runs/cur-v2/checkpoint")
rows = read_jsonl(ROOT / "data/gsm8k/test.jsonl")
order = list(range(len(rows)))
random.Random(11).shuffle(order)
rows = [rows[i] for i in order[:30]]

saved = model.delta_snapshot()
saved_inj = {k: v.detach().clone() for k, v in model.injection.state_dict().items()}

result = {}
for keep in (3, 0):
    for depth in (1, 4):
        ces, hits = [], 0
        for row in rows:
            ce, ok = score(model, row, keep, depth)
            ces.append(ce)
            hits += int(ok)
        result[f"trained_k{keep}_d{depth}"] = {"ce": sum(ces) / len(ces), "em": hits / len(rows)}
        print(f"trained  keep={keep} d={depth}: CE={sum(ces)/len(ces):.3f} EM={hits/len(rows):.3f}",
              flush=True)

# disable deltas AND reset injection to its initialisation (previous-state passthrough)
with torch.no_grad():
    for wrapper in model.deltas.values():
        for parameter in list(wrapper.B):
            parameter.zero_()
        for parameter in list(wrapper.A):
            parameter.zero_()
    hidden = model.model.config.hidden_size
    model.injection.proj.weight.zero_()
    model.injection.proj.weight[:, :hidden] = torch.eye(hidden, device="cuda",
                                                        dtype=model.injection.proj.weight.dtype)

for keep in (3, 0):
    for depth in (1, 4):
        ces, hits = [], 0
        for row in rows:
            ce, ok = score(model, row, keep, depth)
            ces.append(ce)
            hits += int(ok)
        result[f"disabled_k{keep}_d{depth}"] = {"ce": sum(ces) / len(ces), "em": hits / len(rows)}
        print(f"disabled keep={keep} d={depth}: CE={sum(ces)/len(ces):.3f} EM={hits/len(rows):.3f}",
              flush=True)

model.restore_deltas(saved)
model.injection.load_state_dict(saved_inj)

print("\n=== trained vs disabled ===")
for key in sorted(result):
    row = result[key]
    print(f"  {key:<22} CE={row['ce']:.3f}  EM={row['em']:.3f}")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "cur-v2_trained_vs_disabled.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
