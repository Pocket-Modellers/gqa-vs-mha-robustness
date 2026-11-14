"""
Download TinyStories and tokenize it with Santosh's tokenizer, in the SAME format
as his FineWeb shards (uint16, documents separated by id 0).

    pip install datasets
    python prepare_tinystories.py --tokenizer tokenizer_32000.json

Writes data/tinystories/train.bin and data/tinystories/val.bin.
By default it uses 300k stories (~60M tokens), which is plenty for a smoke test.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer_32000.json")
    ap.add_argument("--out-dir", default="data/tinystories")
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--eos-id", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")

    for split, n, fname in [("train", args.n_train, "train.bin"), ("validation", None, "val.bin")]:
        data = ds[split]
        if n is not None:
            data = data.select(range(min(n, len(data))))
        texts = [t for t in data["text"] if t.strip()]
        ids = []
        for i in range(0, len(texts), 1000):
            for enc in tok.encode_batch(texts[i : i + 1000], add_special_tokens=False):
                ids.extend(enc.ids)
                ids.append(args.eos_id)  # document separator, same as FineWeb shards
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(args.out_dir, fname)
        arr.tofile(path)
        print(f"{path}: {len(arr):,} tokens from {len(texts):,} stories")


if __name__ == "__main__":
    main()
