#!/usr/bin/env python
"""HybridLoop v0.1 inference CLI.

Usage:
  python infer.py "Once upon a time" --max-new 128
  python infer.py --interactive
"""
import argparse

from hybridloop_model import generate, load_release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="Once upon a time")
    ap.add_argument("--ckpt", default="hybridloop-v0.1.pt")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    model, router, tok, ck = load_release(args.ckpt)
    print(f"HybridLoop v0.1 loaded (base ckpt step {ck['step']}, "
          f"adaptive exit B={ck['args'].get('B', 8)})", flush=True)
    if args.interactive:
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            out = generate(model, router, tok, prompt, args.max_new)
            print(prompt + out)
    else:
        out = generate(model, router, tok, args.prompt, args.max_new)
        print(args.prompt + out)


if __name__ == "__main__":
    main()
