import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    dest = ROOT / "models" / args.repo.split("/")[-1]
    print(f"Downloading {args.repo} -> {dest}", flush=True)
    start = time.perf_counter()
    snapshot_download(args.repo, revision=args.revision, local_dir=dest, max_workers=3,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"])
    result = {"repository": args.repo, "revision": args.revision, "path": str(dest),
              "seconds": time.perf_counter() - start,
              "bytes": sum(p.stat().st_size for p in dest.glob("*") if p.is_file())}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"download_{dest.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
