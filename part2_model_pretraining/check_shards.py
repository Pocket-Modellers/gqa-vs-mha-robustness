"""
Sanity-check Santosh's shards before training.

    python check_shards.py --data-dir data/fineweb --tokenizer tokenizer_32000.json

Checks every .bin file: token count, max id (< vocab size), how many documents,
and prints a decoded snippet so you can eyeball that it's real English.
"""

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/fineweb")
    ap.add_argument("--tokenizer", default="tokenizer_32000.json")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--eos-id", type=int, default=0)
    args = ap.parse_args()

    tok = None
    if os.path.exists(args.tokenizer):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        print(f"tokenizer vocab size: {tok.get_vocab_size()}")
        for t in ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]:
            print(f"  {t:15s} -> id {tok.token_to_id(t)}")
    else:
        print(f"(tokenizer not found at {args.tokenizer}; skipping decode)")

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
    if not paths:
        print(f"No .bin files in {args.data_dir}")
        return

    total, ok = 0, True
    for p in paths:
        a = np.memmap(p, dtype=np.uint16, mode="r")
        mx = int(a.max())
        n_docs = int((a == args.eos_id).sum())
        total += len(a)
        status = "OK" if mx < args.vocab_size else "BAD: id >= vocab size!"
        ok &= mx < args.vocab_size
        print(f"{os.path.basename(p):25s} {len(a):>14,} tokens  {n_docs:>10,} docs  max id {mx:>6}  {status}")
    print(f"TOTAL: {total:,} tokens in {len(paths)} files")

    if tok is not None:
        a = np.memmap(paths[0], dtype=np.uint16, mode="r")
        print("\n--- first 150 tokens of", os.path.basename(paths[0]), "---")
        print(tok.decode(a[:150].tolist(), skip_special_tokens=False))
    print("\nALL GOOD" if ok else "\nPROBLEM FOUND: tell Santosh")


if __name__ == "__main__":
    main()
