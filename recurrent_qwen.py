"""Recurrent-depth (looped-layer) latent reasoning on a 4-bit Qwen3 decoder.

Technique
---------
A pretrained decoder is converted into a looped model.  Its transformer layers are
split into  prelude | recurrent block | coda.  In the forward pass the *same* block
weights are applied K times in place: the residual stream is refined iteratively,
so the model can spend more latent computation on a hard problem without emitting
more tokens.  This is the recurrent-depth / looped-LM formulation (Geiping et al.
2025; Zhu et al. 2025) applied as a post-training conversion (Bae et al. 2025).

Each loop iteration carries its own low-rank delta on the block's projection
matrices ("relaxed" weight tying), so iteration k is not forced to be identical to
iteration k-1 while the bulk of the parameters stay shared.

Training objective: cross entropy on the answer tokens, plus deep supervision --
the answer is also read out after every iteration, which makes inference depth a
measurable test-time-compute knob instead of a fixed constant.

The forward pass is written explicitly (no layer-list surgery) so that the loop is
plain Python and every intermediate hidden state is available.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen3-4B-Thinking-2507-bnb-4bit"
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_PROMPT = "Solve the math problem. Give the final numeric answer in the format #### number."
SUBMODULES = (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
              ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"))


def setup(seed=2026):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_number(text):
    from decimal import Decimal, InvalidOperation
    try:
        value = Decimal(text.replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return None
    return str(value.quantize(Decimal(1))) if value == value.to_integral_value() else str(value.normalize())


def answer_from_text(text):
    """Extract the final numeric answer.

    Note: generation is cut off at a token budget, so the model is frequently still
    inside its reasoning block.  We deliberately do NOT require a closing tag -- the
    grader should look at the last number produced, otherwise every truncated-but-
    correct answer would be scored wrong.
    """
    text = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    elif "####" in text:
        text = text.rsplit("####", 1)[-1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else None


class LatentInjection(nn.Module):
    """Re-inject the prelude output at the start of every loop iteration.

    Geiping et al. concatenate the *initial* representation with the previous loop's
    hidden state and project the pair back down, so each iteration can consult the
    original problem encoding instead of only the running refinement.  Without this
    the loop tends to drift away from the input as depth grows.

    Initialised as an identity on the previous-state half and zero on the initial half,
    which keeps depth=1 exactly equal to the stock forward pass (see check_equivalence).
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.proj.weight.zero_()
            eye = torch.eye(dim, dtype=torch.float32)
            self.proj.weight[:, :dim] = eye      # previous loop state -> passthrough
            # the initial-state half stays zero until training moves it

    def forward(self, previous, initial):
        pair = torch.cat([previous.float(), initial.float()], dim=-1)
        return self.proj(pair).to(previous.dtype)


class DeltaSet(nn.Module):
    """Per-iteration low-rank deltas added to one frozen projector.

    The loop iteration is *state*, not an argument, because the wrapped projector is
    called by the transformer layer's own code (self.q_proj(...)), which we do not
    control.  Setting `.iteration` before each block pass selects which delta applies.
    """

    def __init__(self, linear, iters, rank, alpha=None, device=None):
        super().__init__()
        self.linear = linear
        # Use in/out_features attributes: touching `.weight` on a bitsandbytes 4-bit
        # module can trigger a dequantisation, which would materialise a full fp32 copy.
        in_features = int(linear.in_features)
        out_features = int(linear.out_features)
        self.rank = rank
        self.scaling = (alpha or rank * 2) / rank
        self.iters = iters
        self.iteration = 0
        device = device or torch.device("cuda:0")
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for _ in range(iters):
            a = torch.empty(rank, in_features, dtype=torch.float32, device=device)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.A.append(nn.Parameter(a))
            self.B.append(nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32,
                                                   device=device)))

    def forward(self, x):
        out = self.linear(x)
        index = min(self.iteration, self.iters - 1)
        delta = F.linear(F.linear(x.float(), self.A[index].float()),
                         self.B[index].float()) * self.scaling
        return out + delta.to(out.dtype)


class RecurrentDepthQwen(nn.Module):
    def __init__(self, checkpoint=None, train_depth=4, block_start=24, block_len=6, rank=8,
                 lr=1e-4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa",
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if checkpoint:
            config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
            train_depth = config["train_depth"]
            block_start = config["block_start"]
            block_len = config["block_len"]
            rank = config["rank"]
        self.train_depth = train_depth
        self.block_start = block_start
        self.block_len = block_len
        self.rank = rank
        self.device = torch.device("cuda:0")
        self.core = self.model.model
        # Re-injection layer for the recurrent loop (Geiping et al. style): each
        # iteration sees the prelude output again alongside its running state.
        self.injection = LatentInjection(self.model.config.hidden_size).to(self.device)
        if checkpoint:
            inject_path = Path(checkpoint) / "injection.pt"
            if inject_path.exists():
                self.injection.load_state_dict(
                    torch.load(inject_path, map_location="cuda", weights_only=True))
        self.all_layers = list(self.core.layers)
        if block_start + block_len > len(self.all_layers):
            raise ValueError("block extends past the end of the stack")
        self.prelude = self.all_layers[:block_start]
        self.block = self.all_layers[block_start:block_start + block_len]
        self.coda = self.all_layers[block_start + block_len:]
        self.deltas = nn.ModuleDict()
        for layer_index, layer in enumerate(self.block):
            attached = []
            for parent_name, name in SUBMODULES:
                parent = getattr(layer, parent_name)
                key = f"b{layer_index}_{parent_name}_{name}"
                wrapper = DeltaSet(getattr(parent, name), train_depth, rank, device=self.device)
                setattr(parent, name, wrapper)
                self.deltas[key] = wrapper
                attached.append(wrapper)
            # Plain tuple (not a Module) so it stays out of state_dict; _run_layer uses
            # it to select which delta is active for the current loop iteration.
            object.__setattr__(layer, "_delta_sets", tuple(attached))
        if checkpoint:
            state = torch.load(Path(checkpoint) / "deltas.pt", map_location="cuda", weights_only=True)
            state = {k: v for k, v in state.items() if ".A." in k or ".B." in k}
            self.deltas.load_state_dict(state, strict=False)
        # Train only the low-rank deltas.  Two traps here:
        #  * NF4 weights are stored as packed uint8, so an is_floating_point() guard
        #    would skip them and leave the whole base model trainable;
        #  * each DeltaSet also *holds* the frozen projector it wraps, so
        #    self.deltas.parameters() would re-enable those weights.
        # Freeze everything that can be frozen, then switch on A/B explicitly.
        skipped = 0
        for parameter in self.model.parameters():
            try:
                parameter.requires_grad_(False)
            except RuntimeError:
                skipped += 1
                parameter.requires_grad = False
        self.frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        self.unfreezable = skipped
        for wrapper in self.deltas.values():
            for parameter in list(wrapper.A) + list(wrapper.B):
                parameter.requires_grad_(True)
        # The injection projection is learned too, but starts as an exact passthrough.
        for parameter in self.injection.parameters():
            parameter.requires_grad_(True)

    # ---------- forward ----------
    def _embed(self, input_ids):
        return self.core.embed_tokens(input_ids)

    def _prepare(self, input_ids):
        hidden = self._embed(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.core.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        return hidden, positions, causal_mask

    def forward_split(self, prompt_ids, scaffold_ids=None, target_ids=None, depth=None,
                      collect_iterations=False, mark_offset=None, mark_length=None):
        """Run prelude over prompt+scaffold, loop the block, then run the coda.

        The latent loop sits *between* the visible scaffold and the answer, so any
        reasoning the model does not spell out has to be carried by the recurrent
        iterations.  Per-iteration hidden states are collected over the supervised
        span (mark_offset/mark_length) when collect_iterations is set.
        """
        depth = depth or self.train_depth
        body = list(prompt_ids) + list(scaffold_ids or []) + list(target_ids or [])
        input_ids = torch.tensor([body], dtype=torch.long, device="cuda")
        hidden, positions, causal_mask = self._prepare(input_ids)
        position_embeddings = self.core.rotary_emb(hidden, positions)
        for layer in self.prelude:
            hidden = self._run_layer(layer, hidden, position_embeddings, causal_mask, positions, 0)
        if mark_offset is None:
            mark_offset = len(prompt_ids) + len(scaffold_ids or [])
        if mark_length is None:
            mark_length = len(target_ids or [])
        initial = hidden
        marks = []
        for iteration in range(depth):
            # Re-inject the prelude output before each pass through the shared block.
            hidden = self.injection(hidden, initial)
            for layer in self.block:
                hidden = self._run_layer(layer, hidden, position_embeddings, causal_mask,
                                         positions, iteration)
            if collect_iterations and mark_length > 0:
                marks.append(hidden[:, mark_offset:mark_offset + mark_length])
        for layer in self.coda:
            hidden = self._run_layer(layer, hidden, position_embeddings, causal_mask, positions, 0)
        logits = self.model.lm_head(self.core.norm(hidden))
        return logits, marks

    def _run_layer(self, layer, hidden, position_embeddings, mask, position_ids, iteration):
        for wrapper in layer.__dict__.get("_delta_sets", ()):
            wrapper.iteration = iteration
        out = layer(hidden, attention_mask=mask, position_embeddings=position_embeddings,
                    position_ids=position_ids, past_key_values=None, use_cache=False)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, input_ids, depth=None, collect_iterations=False):
        """prelude -> block x depth -> coda.  Returns logits and optional per-iteration states."""
        depth = depth or self.train_depth
        hidden = self._embed(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        # Reuse the model's own mask builder: hand-rolling the mask (or omitting
        # position_ids) silently changes attention and breaks equivalence with the
        # stock forward pass.
        causal_mask = create_causal_mask(
            config=self.core.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        position_embeddings = self.core.rotary_emb(hidden, positions)
        for layer in self.prelude:
            hidden = self._run_layer(layer, hidden, position_embeddings, causal_mask, positions, 0)
        marks = []
        for iteration in range(depth):
            for layer in self.block:
                hidden = self._run_layer(layer, hidden, position_embeddings, causal_mask,
                                         positions, iteration)
            if collect_iterations:
                marks.append(hidden)
        for layer in self.coda:
            hidden = self._run_layer(layer, hidden, position_embeddings, causal_mask, positions, 0)
        logits = self.model.lm_head(self.core.norm(hidden))
        return logits, marks

    # ---------- parameters / io ----------
    def trainable_parameters(self):
        """Low-rank deltas plus the loop re-injection projection.

        The base 4-bit weights stay frozen.  The injection term must be included here or
        the optimiser never sees it: an earlier version returned only the deltas, and the
        injection matrix stayed at its exact zero initialisation for the whole run, so the
        re-injection mechanism was silently inert.
        """
        out = []
        for wrapper in self.deltas.values():
            out.extend(list(wrapper.A))
            out.extend(list(wrapper.B))
        out.extend(list(self.injection.parameters()))
        return out

    def deltas_state_dict(self):
        """Only the trainable A/B deltas.

        DeltaSet holds the frozen projector as a submodule, so a full state_dict would
        also drag in bitsandbytes quantisation metadata (`.weight.absmax`, `.quant_map`,
        `.quant_state.*`).  Feeding that back into load_state_dict raises on unexpected
        keys, so filter to the delta parameters only.
        """
        return {k: v.detach().cpu() for k, v in self.deltas.state_dict().items()
                if ".A." in k or ".B." in k}

    def delta_snapshot(self):
        """In-memory snapshot for restoring after ablations."""
        return {k: v.detach().clone() for k, v in self.deltas_state_dict().items()}

    def restore_deltas(self, snapshot):
        current = self.deltas_state_dict()
        with torch.no_grad():
            for name, tensor in snapshot.items():
                if name in current:
                    self.deltas.state_dict()[name].copy_(tensor.to(self.device))

    def save(self, directory, extra=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.deltas_state_dict(), directory / "deltas.pt")
        torch.save({k: v.detach().cpu() for k, v in self.injection.state_dict().items()},
                   directory / "injection.pt")
        payload = {
            "base_model": "Qwen/Qwen3-4B-Thinking-2507",
            "quantized_weights": str(MODEL_DIR),
            "method": "recurrent depth (weight-tied block looped K times with per-iteration low-rank deltas) + deep supervision",
            "train_depth": self.train_depth,
            "block_start": self.block_start,
            "block_len": self.block_len,
            "rank": self.rank,
            "trainable": "low-rank deltas on 7 projections of the 6 looped layers",
            "scope": "GSM8K answer-only training on one 8GB laptop GPU; inference depth is a runtime knob",
        }
        payload.update(extra or {})
        (directory / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                               encoding="utf-8")

    # ---------- data ----------
    def prefix(self, question):
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
        if not isinstance(encoded, list):
            encoded = list(encoded["input_ids"])
        return [int(t) for t in encoded]

    def target(self, row):
        """Answer target with no scaffolding (kept for the hardest curriculum stage)."""
        text = "</think>\n\n#### " + row["answer"]
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)] \
               + [self.tokenizer.eos_token_id]

    def cot_steps(self, row):
        """Split the reference solution into reasoning steps.

        Each GSM8K rationale line is one step; the calculator annotations are stripped
        but the line structure is preserved because the curriculum removes whole steps.
        """
        rationale = re.sub(r"<<[^>]*>>", "", row["rationale"]).strip()
        steps = [line.strip() for line in rationale.split("\n") if line.strip()]
        return steps

    def scaffolded_target(self, row, keep_steps=None):
        """Target that keeps some explicit reasoning steps, then the answer.

        This is the middle ground between a full chain of thought and no reasoning at
        all: the kept steps act as a visible scaffold while the *remaining* computation
        has to happen inside the recurrent latent iterations.  keep_steps=None keeps
        everything (pure text CoT), keep_steps=0 is the answer-only setting.

        Returns (token_ids, n_prompt_positions) where the prompt positions are the
        prefix + scaffold, i.e. everything whose hidden state may be trained on.
        """
        steps = self.cot_steps(row)
        if keep_steps is not None:
            steps = steps[:keep_steps]
        body = "".join(step + "\n" for step in steps)
        text = body + "</think>\n\n#### " + row["answer"]
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)] \
               + [self.tokenizer.eos_token_id]

    def scaffold_prefix_len(self, row, keep_steps):
        steps = self.cot_steps(row)
        if keep_steps is not None:
            steps = steps[:keep_steps]
        body = "".join(step + "\n" for step in steps)
        return len(self.tokenizer.encode(body, add_special_tokens=False))

    def batch(self, row):
        prefix = self.prefix(row["question"])
        target = self.target(row)
        return prefix + target, prefix, target

    # ---------- training ----------
    def loss(self, row, aux_weight=0.2, depth=None):
        ids, prefix, target = self.batch(row)
        return self.loss_from_ids(ids, len(prefix), target, aux_weight=aux_weight, depth=depth)

    def loss_from_sample(self, sample, aux_weight=0.2, depth=None):
        return self.loss_from_ids(sample["ids"], sample["prefix"], sample["target"],
                                  aux_weight=aux_weight, depth=depth)

    def loss_from_ids(self, ids, prefix_len, target, aux_weight=0.2, depth=None):
        depth = depth or self.train_depth
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        logits, marks = self.forward(input_ids, depth=depth, collect_iterations=aux_weight > 0)
        answer_slice = slice(prefix_len - 1, len(ids) - 1)
        targets = torch.tensor(target, dtype=torch.long, device="cuda")
        main = F.cross_entropy(logits[:, answer_slice].reshape(-1, logits.shape[-1]), targets)
        aux = main.new_zeros(())
        if aux_weight and marks:
            pieces = []
            for hidden in marks[:-1]:
                piece_logits = self.model.lm_head(self.core.norm(hidden[:, answer_slice])).float()
                pieces.append(F.cross_entropy(piece_logits.reshape(-1, piece_logits.shape[-1]), targets))
            if pieces:
                aux = torch.stack(pieces).mean()
        total = main + aux_weight * aux
        return total, {"ce": float(main.detach()), "aux_ce": float(aux.detach()),
                       "ids": len(ids), "depth": depth}

    def scaffold_loss(self, row, keep_steps=1, aux_weight=0.2, depth=None,
                      depth_sampling=False):
        """Train with a visible scaffold: keep `keep_steps` CoT steps, hide the rest.

        Scored on the full scaffold+answer continuation, so the model must both write
        the steps it is given and produce the answer -- with the intervening reasoning
        carried by the latent iterations instead of emitted tokens.

        With depth_sampling the loop count is drawn uniformly from 1..train_depth, which
        is what makes a single set of weights usable at any inference depth (Geiping et
        al. vary the loop count during training for exactly this reason).
        """
        if depth_sampling:
            depth = random.randint(1, self.train_depth)
        depth = depth or self.train_depth
        prompt = self.prefix(row["question"])
        scaffold = self.scaffold_tokens(row, keep_steps)
        answer = [int(t) for t in self.tokenizer.encode(
            "</think>\n\n#### " + row["answer"], add_special_tokens=False)] + [self.tokenizer.eos_token_id]
        # Supervised positions, in absolute body coordinates.
        #
        # body = prompt + scaffold + answer[:-1], and the logits at position p predict
        # body[p+1].  With a scaffold, scaffold[1:] is predicted from positions
        # len(prompt).. and answer[0] onwards from len(prompt)+len(scaffold)-1, so the
        # window starts at len(prompt).  With no scaffold the first answer token is
        # predicted from the last prompt position, so the window starts one earlier.
        # Getting this wrong shifts every label by a token AND still yields equal
        # element counts, so cross entropy would silently score the wrong positions.
        absolute_start = len(prompt) if scaffold else len(prompt) - 1
        span_len = (len(scaffold) - 1 if scaffold else 0) + len(answer)
        labels = torch.tensor([scaffold[1:] + answer], dtype=torch.long,
                              device="cuda").reshape(-1)
        logits, marks = self.forward_split(prompt, scaffold, answer[:-1], depth=depth,
                                           collect_iterations=aux_weight > 0,
                                           mark_offset=absolute_start, mark_length=span_len)
        span = slice(absolute_start, absolute_start + span_len)
        main = F.cross_entropy(logits[:, span].reshape(-1, logits.shape[-1]), labels)
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

    def scaffold_tokens(self, row, keep_steps):
        steps = self.cot_steps(row)
        if keep_steps is not None:
            steps = steps[:keep_steps]
        return [int(t) for t in self.tokenizer.encode("".join(s + "\n" for s in steps),
                                                      add_special_tokens=False)]

    # ---------- inference ----------
    @torch.no_grad()
    def generate(self, question, depth=None, max_new_tokens=24):
        depth = depth or self.train_depth
        ids = self.prefix(question)
        start = time.perf_counter()
        generated = []
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            logits, _ = self.forward(input_ids, depth=depth)
            token = int(logits[0, -1].argmax(-1).item())
            generated.append(token)
            ids.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        torch.cuda.synchronize()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {"answer": answer_from_text(text), "text": text, "depth": depth,
                "seconds": time.perf_counter() - start, "generated_tokens": len(generated),
                "hit_length_limit": bool(generated) and generated[-1] != self.tokenizer.eos_token_id}

    @torch.no_grad()
    def answer_at_depth(self, question, depths=(1, 2, 4, 8, 16), max_new_tokens=24):
        """Greedy answer for several inference depths: the test-time compute curve."""
        results = {}
        for depth in depths:
            ids = self.prefix(question)
            generated = []
            start = time.perf_counter()
            for _ in range(max_new_tokens):
                input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                logits, _ = self.forward(input_ids, depth=depth)
                token = int(logits[0, -1].argmax(-1).item())
                generated.append(token)
                ids.append(token)
                if token == self.tokenizer.eos_token_id:
                    break
            torch.cuda.synchronize()
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results[depth] = {"answer": answer_from_text(text), "text": text,
                              "seconds": round(time.perf_counter() - start, 2),
                              "generated_tokens": len(generated)}
        return results
