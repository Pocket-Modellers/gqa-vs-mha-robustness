"""Step 1: train a byte-level BPE tokenizer on a FineWeb-Edu sample.

Usage: python 01_train_tokenizer.py [VOCAB_SIZE]   (default 32000)
Output: tokenizer_<VOCAB_SIZE>.json
Re-run with 8000 / 16000 for the tokenizer-size extension.
"""
import itertools
import sys

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
N_DOCS = 300_000  # ~ 300M characters, plenty for BPE

# Order matters: the ids are fixed by this order (0=eot, 1=pad, 2=im_start, 3=im_end)
SPECIALS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]


def text_iter():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    for ex in itertools.islice(ds, N_DOCS):
        yield ex["text"]


tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=SPECIALS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)
tok.train_from_iterator(text_iter(), trainer=trainer)

out = f"tokenizer_{VOCAB}.json"
tok.save(out)

# Sanity checks
s = "The quick brown fox jumps over 13 lazy dogs.\n\nNew paragraph, café."
ids = tok.encode(s).ids
assert tok.decode(ids) == s, "round-trip failed"
print(f"saved {out}; vocab={tok.get_vocab_size()}; example -> {len(ids)} tokens")
print("special ids:", {t: tok.token_to_id(t) for t in SPECIALS})
