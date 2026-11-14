"""Step 3 (v2): build the SFT (supervised fine-tuning) dataset with response-only loss masks.

Usage: python 03_prepare_sft.py tokenizer_32000.json [sft_closed.jsonl]
Output (in sft/):
  train.jsonl    {"input_ids": [...], "labels": [...], "category": str}; labels = -100 on prompt tokens
  heldout.jsonl  1,000 Dolly rows, original keys (instruction, context, response, category); NOT in train.jsonl

SFT mix = Dolly (minus held-out) + Alpaca-cleaned + sft_closed.jsonl (closed classification, Dolly format).

Chat template (see sft_template.txt):
<|im_start|>user\n{instruction}\n\n{context}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
CLOSED_PATH = sys.argv[2] if len(sys.argv) > 2 else "sft_closed.jsonl"
MAX_LEN = 512
N_HELDOUT = 1000
random.seed(0)

out = Path("sft")
out.mkdir(exist_ok=True)

# ---- Dolly: hold out 1,000 rows BEFORE anything else, keeping original keys
dolly = [dict(ex) for ex in load_dataset("databricks/databricks-dolly-15k", split="train")]
dolly = [r for r in dolly if r["instruction"].strip() and r["response"].strip()]
random.shuffle(dolly)
heldout, dolly_train = dolly[:N_HELDOUT], dolly[N_HELDOUT:]

# ---- Alpaca-cleaned, normalised to the same keys
alpaca = [
    {"instruction": ex["instruction"], "context": ex["input"],
     "response": ex["output"], "category": "alpaca"}
    for ex in load_dataset("yahma/alpaca-cleaned", split="train")
]
alpaca = [r for r in alpaca if r["instruction"].strip() and r["response"].strip()]

# ---- closed-classification examples from the Part 3 teammate (Dolly format)
closed = []
with open(CLOSED_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ex = json.loads(line)
            closed.append({"instruction": ex["instruction"], "context": ex.get("context", ""),
                           "response": ex["response"], "category": ex.get("category", "closed_clf")})
print(f"closed examples loaded: {len(closed)}")

train = dolly_train + alpaca + closed

# ---- safety net: drop any train row whose (instruction, context) matches a held-out row
held_keys = {(r["instruction"].strip(), r["context"].strip()) for r in heldout}
n_before = len(train)
train = [r for r in train if (r["instruction"].strip(), r["context"].strip()) not in held_keys]
print(f"dropped {n_before - len(train)} train rows that duplicate held-out prompts")
random.shuffle(train)


def prompt_text(instr, ctx):
    user = instr.strip() + (f"\n\n{ctx.strip()}" if ctx.strip() else "")
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def encode(instr, ctx, resp):
    p_ids = tok.encode(prompt_text(instr, ctx)).ids
    r_ids = tok.encode(resp.strip() + "<|im_end|>").ids
    return p_ids + r_ids, [-100] * len(p_ids) + r_ids


kept, too_long = Counter(), Counter()
with open(out / "train.jsonl", "w", encoding="utf-8") as f:
    for r in train:
        ids, labels = encode(r["instruction"], r["context"], r["response"])
        if len(ids) > MAX_LEN:
            too_long[r["category"] == "closed_clf"] += 1
            continue
        f.write(json.dumps({"input_ids": ids, "labels": labels, "category": r["category"]}) + "\n")
        kept["closed_clf" if r["category"] == "closed_clf" else "other"] += 1

with open(out / "heldout.jsonl", "w", encoding="utf-8") as f:
    for r in heldout:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train kept: {sum(kept.values())} (closed_clf {kept['closed_clf']}, other {kept['other']}); "
      f"held-out: {len(heldout)}")

# ---- verification: decode one normal and one closed example, show masked vs trained parts
lines = [json.loads(l) for l in open(out / "train.jsonl", encoding="utf-8")]
for want in (False, True):
    ex = next(e for e in lines if (e["category"] == "closed_clf") == want)
    masked = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100]
    trained = [l for l in ex["labels"] if l != -100]
    print(f"\n===== example ({'closed_clf' if want else 'general'}) =====")
    print("--- MASKED (no loss) ---\n" + tok.decode(masked, skip_special_tokens=False))
    print("--- TRAINED ON (loss) ---\n" + tok.decode(trained, skip_special_tokens=False))
