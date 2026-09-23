from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import time
from decimal import Decimal, InvalidOperation

import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen3-4B-Thinking-2507-bnb-4bit"
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_PROMPT = "Solve the math problem. Give the final numeric answer in the format #### number."
BOUNDARY = "</think>\n\n"


def setup(seed=2026):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_number(text):
    try:
        value = Decimal(text.replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return str(value.normalize())


def answer_from_text(text, skip_think=True):
    if skip_think:
        if "<think>" in text and "</think>" not in text:
            return None  # generation stopped mid-reasoning: no answer yet
        text = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    elif "####" in text:
        text = text.rsplit("####", 1)[-1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else None


class LatentBridge(nn.Module):
    """Map a residual-stream hidden state to a token-embedding-scaled vector.

    Near-identity at init (up-projection zeroed), so the first latent step is the
    model's own normalised hidden state rather than an arbitrary vector.
    """

    def __init__(self, dim, rank=256, embedding_scale=0.03):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.log_scale = nn.Parameter(torch.tensor(float(embedding_scale)).log())

    def forward(self, hidden):
        x = self.norm(hidden.float())
        x = x + self.up(F.silu(self.down(x)))
        return (x * self.log_scale.exp()).to(hidden.dtype)


class LatentQwen(nn.Module):
    def __init__(self, checkpoint=None, lora_rank=8, latent_steps=4, bridge_rank=256,
                 kd_layers=(17, 24, 30, 36)):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa",
        )
        self.kd_layers = tuple(kd_layers)
        self.lora_rank = lora_rank
        self.bridge_rank = bridge_rank
        if checkpoint:
            config = json.loads((Path(checkpoint) / "latent_config.json").read_text(encoding="utf-8"))
            latent_steps = config["latent_steps"]
            self.kd_layers = tuple(config.get("kd_layers", self.kd_layers))
            self.llm = PeftModel.from_pretrained(base, Path(checkpoint) / "adapter", is_trainable=True)
        else:
            self.llm = get_peft_model(base, LoraConfig(
                r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            ))
        self.latent_steps = latent_steps
        with torch.no_grad():
            scale = self.base.get_input_embeddings().weight[:4096].float().pow(2).mean().sqrt().item()
        self.bridge = LatentBridge(self.base.config.hidden_size, rank=bridge_rank,
                                   embedding_scale=scale).to("cuda")
        if checkpoint:
            self.bridge.load_state_dict(
                torch.load(Path(checkpoint) / "bridge.pt", map_location="cuda", weights_only=True))
        self.boundary_ids = [int(t) for t in self.encode(BOUNDARY)]

    # ---------- basics ----------
    @property
    def base(self):
        return self.llm.get_base_model()

    @property
    def backbone(self):
        return self.base.model

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device="cuda")

    def prefix(self, question):
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
        if not isinstance(encoded, list):
            encoded = list(encoded["input_ids"])
        return [int(t) for t in encoded]

    def target_tokens(self, row):
        return [int(t) for t in self.encode("#### " + row["answer"])] + [self.tokenizer.eos_token_id]

    def cot_tokens(self, row):
        body = re.sub(r"<<[^>]*>>", "", row["rationale"]).strip()
        return [int(t) for t in self.encode(body + "\n" + BOUNDARY)]

    def _picked(self, hidden_states):
        return [hidden_states[i].float() for i in self.kd_layers]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------- forward paths ----------
    def _latent_forward(self, prefix_ids, suffix_ids, steps=None, ablation=None, cosine_cache=None):
        """Run prefix -> N continuous latent steps -> suffix, with KV cache.

        The latent steps feed the bridged last hidden state back as the next input
        embedding.  Returns the suffix output and the latent tensors.
        """
        steps = self.latent_steps if steps is None else steps
        mask = torch.ones((1, len(prefix_ids)), dtype=torch.long, device="cuda")
        out = self.llm(input_ids=self.tensor(prefix_ids), attention_mask=mask,
                       use_cache=True, output_hidden_states=True)
        past, hidden = out.past_key_values, out.hidden_states[-1][:, -1:]
        latents = []
        for _ in range(steps):
            if ablation == "zero":
                z = torch.zeros_like(self.bridge(hidden))
            elif ablation == "repeat":
                z = self.base.get_input_embeddings()(self.tensor(self.encode(".")))[:, :1].to(hidden.dtype)
            else:
                z = self.bridge(hidden)
            latents.append(z)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long, device="cuda")], dim=1)
            out = self.llm(inputs_embeds=z, attention_mask=mask, past_key_values=past,
                           use_cache=True, output_hidden_states=True)
            past, hidden = out.past_key_values, out.hidden_states[-1][:, -1:]
        mask = torch.cat([mask, torch.ones((1, len(suffix_ids)), dtype=torch.long, device="cuda")], dim=1)
        out = self.llm(input_ids=self.tensor(suffix_ids), attention_mask=mask, past_key_values=past,
                       use_cache=True, output_hidden_states=True)
        return out, latents

    # ---------- losses ----------
    def losses(self, row, kd_weight=1.0, steps=None, ablation=None, mode="latent"):
        """Teacher (explicit CoT, no grad) hidden-state distillation + student (latent) CE.

        Both branches run the same weights with adapters active; the teacher passes
        its rationale explicitly, the student replaces it with continuous latents.
        Targets are taken at the answer-prediction positions of each branch, computed
        from each branch's own prefix length, then detached.
        """
        prefix = self.prefix(row["question"])
        target = self.target_tokens(row)
        n_target = len(target)

        kd_targets = None
        if kd_weight:
            cot = self.cot_tokens(row)
            teacher_ids = prefix + cot + target[:-1]
            first_t = len(prefix) + len(cot) - 1
            with torch.no_grad():
                t_out = self.llm(input_ids=self.tensor(teacher_ids), use_cache=False,
                                 output_hidden_states=True)
                hidden = t_out.hidden_states
                try:
                    kd_targets = [h[:, first_t:first_t + n_target].detach().clone()
                                  for h in self._picked(hidden)]
                finally:
                    del t_out, hidden
                    torch.cuda.empty_cache()

        if mode == "direct":
            ids = prefix + target
            s_out = self.llm(input_ids=self.tensor(ids), use_cache=False, output_hidden_states=True)
            hidden_states = s_out.hidden_states
            logits = s_out.logits.float()
            first_s = len(prefix) - 1
            first_h = len(prefix) - 1
        else:
            s_out, _ = self._latent_forward(prefix, self.boundary_ids + target[:-1],
                                            steps=steps, ablation=ablation)
            hidden_states = s_out.hidden_states
            logits = s_out.logits.float()
            first_s = len(self.boundary_ids) - 1
            first_h = len(self.boundary_ids) - 1

        student_ce = F.cross_entropy(
            logits[:, first_s:first_s + n_target - 1].reshape(-1, logits.shape[-1]),
            self.tensor(target[:-1]).reshape(-1))

        kd = student_ce.new_zeros(())
        if kd_weight and kd_targets is not None:
            picked = self._picked(hidden_states)
            pieces = []
            for index in range(len(self.kd_layers)):
                student_h = picked[index][:, first_h:first_h + n_target]
                teacher_h = kd_targets[index]
                diff = F.smooth_l1_loss(student_h, teacher_h, reduction="none")
                pieces.append(diff.mean() / teacher_h.std().clamp_min(1e-3))
            kd = torch.stack(pieces).mean()

        report = {"ce": float(student_ce.detach()), "kd": float(kd.detach()),
                  "prefix_len": len(prefix), "cot_len": len(self.cot_tokens(row)),
                  "target_len": n_target}
        return student_ce, kd, report

    # ---------- io ----------
    def save(self, directory, extra=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(directory / "adapter", safe_serialization=True)
        torch.save(self.bridge.state_dict(), directory / "bridge.pt")
        payload = {
            "base_model": "Qwen/Qwen3-4B-Thinking-2507",
            "quantized_weights": str(MODEL_DIR),
            "latent_steps": self.latent_steps,
            "kd_layers": [int(i) for i in self.kd_layers],
            "lora_rank": self.lora_rank,
            "bridge_rank": self.bridge_rank,
            "method": ("continuous hidden-state feedback (Coconut-style) trained with same-model "
                       "explicit-CoT hidden-state distillation (CODI/iCoT-style) on a 4-bit QLoRA "
                       "Qwen3-4B-Thinking"),
            "scope": "GSM8K math-reasoning pilot on a single 8GB laptop GPU",
        }
        payload.update(extra or {})
        (directory / "latent_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- inference ----------
    @torch.no_grad()
    def generate(self, question, mode="latent", max_new_tokens=48, steps=None, ablation=None):
        was_training = self.training
        self.eval()
        prefix = self.prefix(question)
        start = time.perf_counter()
        generated = []
        if mode == "latent":
            out, _ = self._latent_forward(prefix, self.boundary_ids, steps=steps, ablation=ablation)
            past, hidden = out.past_key_values, out.hidden_states[-1][:, -1:]
        else:
            ids = prefix + (self.boundary_ids if mode == "direct" else [])
            out = self.llm(input_ids=self.tensor(ids), use_cache=True, output_hidden_states=True)
            past, hidden = out.past_key_values, out.hidden_states[-1][:, -1:]
        for _ in range(max_new_tokens):
            token = int(self.base.lm_head(hidden[:, -1]).argmax(-1).item())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                break
            mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device="cuda")
            out = self.llm(input_ids=self.tensor([token]), attention_mask=mask,
                           past_key_values=past, use_cache=True, output_hidden_states=True)
            past, hidden = out.past_key_values, out.hidden_states[-1][:, -1:]
        torch.cuda.synchronize()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if was_training:
            self.train()
        return {
            "answer": answer_from_text(text, skip_think=(mode == "cot")),
            "text": text, "mode": mode, "ablation": ablation,
            "seconds": time.perf_counter() - start, "generated_tokens": len(generated),
            "latent_steps": (self.latent_steps if steps is None else steps) if mode == "latent" else 0,
            "hit_length_limit": bool(generated) and generated[-1] != self.tokenizer.eos_token_id,
        }
