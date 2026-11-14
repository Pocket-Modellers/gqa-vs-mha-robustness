"""Step 2: stream FineWeb-Edu, tokenize, write uint16 .bin shards (nanoGPT-style).

Usage: python 02_make_shards.py tokenizer_32000.json [TARGET_TOKENS]
Output (in data/): val_000.bin (5M tokens), train_000.bin, train_001.bin, ... (100M tokens each)
Load later with np.memmap(path, dtype=np.uint16, mode="r").
Needs vocab size < 65536 (uint16).
"""
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

tok_path = sys.argv[1]
TARGET = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_500_000_000
SHARD = 100_000_000
VAL = 5_000_000
BATCH = 1000

tok = Tokenizer.from_file(tok_path)
assert tok.get_vocab_size() < 65536
EOT = tok.token_to_id("<|endoftext|>")

out = Path("data")
out.mkdir(exist_ok=True)


class Writer:
    def __init__(self):
        self.name, self.cap = "val_000.bin", VAL
        self.buf = np.empty(SHARD, dtype=np.uint16)
        self.fill = 0
        self.train_idx = 0
        self.train_tokens = 0

    def add(self, ids):
        pos = 0
        while pos < len(ids):
            take = min(len(ids) - pos, self.cap - self.fill)
            self.buf[self.fill:self.fill + take] = ids[pos:pos + take]
            self.fill += take
            pos += take
            if not self.name.startswith("val"):
                self.train_tokens += take
            if self.fill == self.cap:
                self.flush()

    def flush(self):
        if self.fill == 0:
            return
        self.buf[:self.fill].tofile(out / self.name)
        print(f"wrote {self.name} ({self.fill:,} tokens); train total {self.train_tokens:,}")
        self.fill = 0
        self.name = f"train_{self.train_idx:03d}.bin"
        self.train_idx += 1
        self.cap = SHARD


w = Writer()
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                  split="train", streaming=True)

batch = []
done = False
for ex in ds:
    batch.append(ex["text"])
    if len(batch) < BATCH:
        continue
    for e in tok.encode_batch(batch):
        w.add(np.array(e.ids + [EOT], dtype=np.uint16))
    batch = []
    if w.train_tokens >= TARGET:
        break

w.flush()  # last partial shard
print("done")
