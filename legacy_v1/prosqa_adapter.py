"""ProsQA adapter: synthetic DAG logic reasoning, the task Coconut reports gains on.

Why this task instead of GSM8K: the answer is a specific invented entity name
("Tom is a zhorpus"), so guessing cannot score.  On GSM8K a 4B model already answers many
problems from the prompt plus a couple of gold steps, which made the depth-1 number a
measurement of the scaffold rather than of latent computation.

Data format (Coconut, Meta):
    question : a list of logical statements
    steps    : the gold derivation as an ordered list of statements
    answer   : final statement, e.g. "Tom is a zhorpus."
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from recurrent_qwen import RecurrentDepthQwen, ROOT

PROSQA_DIR = ROOT / "data" / "prosqa"
SYSTEM = "Determine the logical relationship by reasoning step by step."


def load_prosqa(split, count=None, seed=2026):
    rows = json.loads((PROSQA_DIR / f"prosqa_{split}.json").read_text(encoding="utf-8"))
    if count is not None:
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        rows = [rows[i] for i in order[:count]]
    return rows


def prosqa_row_to_gsm8k_shape(row):
    """Adapt to the field names the training scripts already expect."""
    return {
        "question": row["question"],
        "rationale": "\n".join(row["steps"]),
        "answer": row["answer"],
    }


class ProsQAModel(RecurrentDepthQwen):
    """RecurrentDepthQwen with the prompt and target shaped for ProsQA."""

    def prefix(self, question):
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
        if not isinstance(encoded, list):
            encoded = list(encoded["input_ids"])
        return [int(t) for t in encoded]

    def target(self, row):
        """Answer text for ProsQA is a full statement, not a bare number."""
        text = "</think>\n\n#### " + row["answer"].strip()
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)] \
               + [self.tokenizer.eos_token_id]

    def cot_steps(self, row):
        if isinstance(row.get("steps"), list):
            return [s.strip() for s in row["steps"] if str(s).strip()]
        return super().cot_steps(row)

    def scaffold_loss(self, row, keep_steps=1, aux_weight=0.2, depth=None, depth_sampling=False):
        if depth_sampling:
            depth = random.randint(1, self.train_depth)
        depth = depth or self.train_depth
        prompt = self.prefix(row["question"])
        scaffold = self.scaffold_tokens(row, keep_steps)
        answer = self.target(row)
        absolute_start = len(prompt) if scaffold else len(prompt) - 1
        span_len = (len(scaffold) - 1 if scaffold else 0) + len(answer)
        import torch
        import torch.nn.functional as F
        labels = torch.tensor([scaffold[1:] + answer], dtype=torch.long, device="cuda").reshape(-1)
        logits, marks = self.forward_split(prompt, scaffold, answer[:-1], depth=depth,
                                           collect_iterations=aux_weight > 0,
                                           mark_offset=absolute_start, mark_length=span_len)
        span = slice(absolute_start, absolute_start + span_len)
        main = F.cross_entropy(logits[:, span].reshape(-1, logits.shape[-1]), labels)
        aux = main.new_zeros(())
        if aux_weight and marks:
            pieces = []
            for hidden in marks[:-1]:
                piece_logits = self.model.lm_head(self.core.norm(hidden)).float()
                pieces.append(F.cross_entropy(piece_logits.reshape(-1, piece_logits.shape[-1]), labels))
            if pieces:
                aux = torch.stack(pieces).mean()
        total = main + aux_weight * aux
        return total, {"ce": float(main.detach()), "aux_ce": float(aux.detach()),
                       "depth": depth, "keep_steps": keep_steps,
                       "scaffold": len(scaffold), "answer": len(answer)}


def prosqa_accuracy(pred_text, gold):
    """ProsQA scores by exact final statement, normalised for case/spacing."""
    if not pred_text:
        return False
    pred = pred_text.strip().rstrip(".").strip().lower()
    want = gold.strip().rstrip(".").strip().lower()
    return pred == want


def extract_prosqa_answer(text):
    """The answer clause after ####, or the last sentence that looks like a statement."""
    import re
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    else:
        text = text.split("</think>")[-1]
    text = text.replace("<|im_end|>", "").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None
    # prefer a sentence of the form "<Name> is a <thing>."
    for line in reversed(lines):
        m = re.search(r"([A-Z][A-Za-z]* is a [a-z]+)", line)
        if m:
            return m.group(1)
    return lines[-1].rstrip(".").strip()
