import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download, HfApi
import pyarrow.parquet as pq

if __name__ == "__main__":
    folder = ROOT / "data" / "gsm8k"
    folder.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info("openai/gsm8k", timeout=20)
    for split in ("train", "test"):
        filename = f"main/{split}-00000-of-00001.parquet"
        path = hf_hub_download("openai/gsm8k", filename, repo_type="dataset", revision=info.sha, local_dir=folder / "source")
        rows = pq.read_table(path).to_pylist()
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                rationale, answer = row["answer"].rsplit("####", 1)
                record = {"id": f"gsm8k-{split}-{i}", "question": row["question"], "rationale": rationale.strip(), "answer": answer.strip()}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(split, len(rows), flush=True)
    (folder / "source.json").write_text(json.dumps({"repository": "openai/gsm8k", "revision": info.sha, "license": "mit", "source": "https://huggingface.co/datasets/openai/gsm8k", "train_test_separation": "Official test split never used for training or checkpoint selection"}, indent=2), encoding="utf-8")
