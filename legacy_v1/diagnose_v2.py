"""Diagnose why training loss falls to 0.6 but development accuracy stays at ~5%.

Three hypotheses, each testable:

  H1 teacher-forcing gap - training feeds the gold scaffold as input, so the model never
     practises generating it.  Test: score the model *given* the gold scaffold.
  H2 the loop is not learning - the deltas did nothing and the base model answers alone.
     Test: compare against a zeroed-delta run and against the frozen base model.
  H3 the checkpoint is bad / mis-loaded - Test: verify deltas are non-zero and that
     depth 1 still equals the base model.
"""

import json

import torch

from recurrent_qwen import RecurrentDepthQwen, ROOT, answer_from_text, read_jsonl, setup

setup(0)
model = RecurrentDepthQwen(checkpoint=ROOT / "runs/cur-v1/checkpoint")
rows = read_jsonl(ROOT / "data/gsm8k/train.jsonl")[:6]

print("=" * 72)
print("H3: checkpoint sanity")
print("=" * 72)
nonzero = 0
total = 0
for name, tensor in model.deltas.state_dict().items():
    if ".B." in name:
        total += 1
        if float(tensor.abs().max()) > 0:
            nonzero += 1
print(f"  delta B matrices with non-zero entries: {nonzero}/{total}")
inject_delta = float(model.injection.proj.weight[:, model.model.config.hidden_size:].abs().max())
print(f"  injection 'initial-state' half max |w|: {inject_delta:.4f}  "
      f"({'learned' if inject_delta > 1e-4 else 'STILL ZERO - injection never used'})")

print()
print("=" * 72)
print("H1: teacher-forced scaffold (does the model score well when given the scaffold?)")
print("=" * 72)
for keep in (2, 1, 0):
    total_loss, n = 0.0, 0
    for row in rows:
        loss, report = model.scaffold_loss(row, keep_steps=keep, aux_weight=0.0, depth=4)
        total_loss += report["ce"]
        n += 1
    print(f"  keep_steps={keep}: mean CE = {total_loss / n:.3f}")

print()
print("=" * 72)
print("H2/H1: greedy generation, with and without the loop")
print("=" * 72)
saved = {k: v.detach().clone() for k, v in model.deltas.state_dict().items()}


def greedy(question, depth, max_new_tokens=24):
    ids = model.prefix(question)
    out_tokens = []
    for _ in range(max_new_tokens):
        logits, _ = model.forward(torch.tensor([ids], dtype=torch.long, device="cuda"), depth=depth)
        token = int(logits[0, -1].argmax(-1).item())
        out_tokens.append(token)
        ids.append(token)
        if token == model.tokenizer.eos_token_id:
            break
    return model.tokenizer.decode(out_tokens, skip_special_tokens=True)


for row in rows[:3]:
    print(f"\n  gold={row['answer']}  Q: {row['question'][:70]}...")
    print(f"    ref steps: {model.cot_steps(row)[:2]}")
    for depth in (1, 4):
        text = greedy(row["question"], depth)
        print(f"    depth{depth}: pred={answer_from_text(text)!r}")
        print(f"      {text[:150]!r}")

print()
print("=" * 72
      )
print("zeroed-delta control (is the loop doing anything at all?)")
print("=" * 72)
with torch.no_grad():
    for wrapper in model.deltas.values():
        for parameter in list(wrapper.B):
            parameter.zero_()
for row in rows[:3]:
    text = greedy(row["question"], 4)
    print(f"  gold={row['answer']}: zeroed-loop pred={answer_from_text(text)!r}")
model.deltas.load_state_dict(saved)
print("\nrestored deltas")
