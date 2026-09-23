"""Load Ouro-1.4B and test depth-accuracy on ProsQA. Self-contained."""

import json, os, sys, re, time, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Patch: Ouro expects layer_type_validation from transformers.configuration_utils
import transformers.configuration_utils as _cu
if not hasattr(_cu, "layer_type_validation"):
    _cu.layer_type_validation = lambda *a, **kw: None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = str(ROOT / "models" / "Ouro-1.4B")
SYSTEM = "Solve the logic problem. Give the final answer."


def extract_answer(text):
    lines = [ln.strip() for ln in text.replace("</think>", "").split("\n") if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"([A-Z][A-Za-z]*\s+is\s+a\s+[a-z]+)", ln)
        if m:
            return m.group(1)
    return lines[-1].rstrip(".") if lines else None


@torch.no_grad()
def generate(model, tokenizer, ids, max_new_tokens=16):
    generated = []
    current = torch.tensor([ids], dtype=torch.long, device="cuda")
    past = None
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        kwargs = {"input_ids": current}
        if past is not None:
            kwargs["past_key_values"] = past
        out = model(**kwargs, use_cache=True)
        past = out.past_key_values
        token = int(out.logits[0, -1].argmax(-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        current = torch.tensor([[token]], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated), time.perf_counter() - start


def main():
    setup_seed = 11
    random.seed(setup_seed)
    torch.manual_seed(setup_seed)

    print("Loading Ouro-1.4B ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    print(f"Loaded. type={type(model).__name__} layers={model.config.num_hidden_layers}", flush=True)

    # Check loop-related attributes
    for attr in ("num_loop_steps", "recurrent_steps", "loop_steps", "num_recurrence"):
        val = getattr(model.config, attr, None)
        if val is not None:
            print(f"  config.{attr} = {val}")

    # Load ProsQA
    data_path = ROOT / "data" / "prosqa" / "prosqa_test.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    random.Random(11).shuffle(rows)
    rows = rows[:30]
    print(f"Evaluating {len(rows)} questions", flush=True)

    correct = 0
    total_tokens = 0
    records = []
    for i, row in enumerate(rows):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(encoded, list):
            encoded = list(encoded["input_ids"])
        ids = [int(t) for t in encoded]

        text, ntok, sec = generate(model, tokenizer, ids, 16)
        pred = answer_from_text(text)
        gold = row["answer"].strip()
        ok = pred == gold
        correct += int(ok)
        total_tokens += ntok
        records.append({"idx": i, "gold": gold, "pred": pred, "ok": ok, "text": text[:150]})

        if i < 3:
            print(f"  [{i}] gold={gold!r} pred={pred!r} ok={ok}")
            print(f"      text: {text[:120]!r}")

    acc = correct / len(rows)
    print(f"\nOuro-1.4B ProsQA accuracy: {acc:.3f} ({correct}/{len(rows)})", flush=True)
    print(f"mean tokens: {total_tokens/len(rows):.1f}", flush=True)

    out = ROOT / "results" / "ouro1b_prosqa.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"accuracy": acc, "n": len(rows), "records": records}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
