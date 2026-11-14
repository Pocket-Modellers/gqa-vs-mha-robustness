"""
Reads Santosh's token shards and produces training batches.

Shard format (agreed with Part 1):
  - raw uint16 arrays, read with np.memmap(path, dtype=np.uint16, mode="r")
  - documents separated by token id 0 (<|endoftext|>)
  - vocab size 32000

We treat all shards as one long stream of tokens and cut random windows of
length seq_len + 1 out of it (input = first seq_len, target = shifted by one).

Determinism: the batch at a given step depends ONLY on (data_seed, step).
So two runs with the same data_seed see exactly the same data in the same
order, and resuming from a checkpoint continues the exact same stream.
"""

import glob
import os

import numpy as np
import torch


def find_shards(data_dir, pattern):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No shards matching '{pattern}' in '{data_dir}'. "
            f"Check data_dir / train_pattern / val_pattern in your config."
        )
    return paths


class TokenShards:
    def __init__(self, paths, vocab_size=None):
        self.paths = paths
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.lengths = np.array([len(a) for a in self.arrays], dtype=np.int64)
        self.total_tokens = int(self.lengths.sum())
        # probability of picking each shard = its share of all tokens
        self.shard_probs = self.lengths / self.lengths.sum()
        self.vocab_size = vocab_size

    def _sample_windows(self, rng, batch_size, seq_len):
        shard_ids = rng.choice(len(self.arrays), size=batch_size, p=self.shard_probs)
        rows = []
        for s in shard_ids:
            arr = self.arrays[s]
            start = rng.integers(0, len(arr) - seq_len - 1)
            rows.append(np.asarray(arr[start : start + seq_len + 1], dtype=np.int64))
        chunk = torch.from_numpy(np.stack(rows))
        return chunk[:, :-1], chunk[:, 1:]  # inputs, targets

    def get_batch(self, data_seed, step, micro_step, batch_size, seq_len):
        """Training batch: a deterministic function of (data_seed, step, micro_step)."""
        rng = np.random.default_rng([data_seed, step, micro_step])
        return self._sample_windows(rng, batch_size, seq_len)

    def fixed_batches(self, n_batches, batch_size, seq_len, seed=1234):
        """
        Validation batches. The seed is fixed and NOT tied to the run seed,
        so every run (MHA, GQA, every seed) is evaluated on identical tokens.
        """
        rng = np.random.default_rng(seed)
        return [self._sample_windows(rng, batch_size, seq_len) for _ in range(n_batches)]
