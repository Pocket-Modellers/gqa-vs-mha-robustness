"""
Text generation. Two ways to use it:

1) Eyeball a checkpoint (e.g. after the TinyStories smoke test):

    python generate.py sample --ckpt ckpts/tinystories_gqa/final.pt \
        --tokenizer tokenizer_32000.json --prompt "Once upon a time"

2) Make the predictions file Sree (Part 3) asked for:

    python generate.py preds --ckpt ckpts/gqa_s0_sft/final.pt \
        --tokenizer tokenizer_32000.json --eval-set eval_set.jsonl \
        --out preds_gqa_s0.jsonl

   Spec from Sree: greedy decoding, max_new_tokens from each row, stop at EOS,
   one {"id", "output"} per line, output = only the newly generated text.
   Use this SAME script, unchanged, for every model.
"""

import argparse
import json

import torch

from chat_format import format_prompt
from checkpoint import load_model


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new_tokens, stop_ids=(3,), device="cpu"):
    """
    Greedy decoding: at every step pick the single most likely next token.
    Stops early if the model produces any token in stop_ids.
    Returns only the NEW token ids (not the prompt).
    """
    ctx_len = model.cfg.max_seq_len
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -ctx_len:])
        next_id = int(logits[0, -1].argmax())
        if next_id in stop_ids:
            break
        new_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return new_ids


def get_device(arg):
    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_sample(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    device = get_device(args.device)
    model, info = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} (step {info['step']}, {info['tokens_seen']:,} tokens)\n")
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    new = greedy_generate(model, ids, args.max_new_tokens, stop_ids=(0,), device=device)
    print(args.prompt + tok.decode(new))


def cmd_preds(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    device = get_device(args.device)
    model, info = load_model(args.ckpt, device)
    stop_ids = tuple(int(s) for s in args.stop_ids.split(","))
    print(f"loaded {args.ckpt}; stop ids {stop_ids}")

    with open(args.eval_set) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(args.out, "w") as out:
        for i, row in enumerate(rows):
            # Use the ready-made prompt if the row has one; otherwise build it from
            # instruction/context with the SAME template used in SFT (chat_format.py).
            prompt = row[args.prompt_field] if args.prompt_field in row else format_prompt(row)
            prompt_ids = tok.encode(prompt).ids  # exactly as in 03_prepare_sft.py
            new = greedy_generate(model, prompt_ids, int(row[args.max_tokens_field]),
                                  stop_ids=stop_ids, device=device)
            text = tok.decode(new, skip_special_tokens=True)
            out.write(json.dumps({"id": row[args.id_field], "output": text}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)}")
    print(f"wrote {len(rows)} predictions to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="generate text from a prompt")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--tokenizer", default="tokenizer_32000.json")
    s.add_argument("--prompt", default="Once upon a time")
    s.add_argument("--max-new-tokens", type=int, default=150)
    s.add_argument("--device", default=None)
    s.set_defaults(func=cmd_sample)

    p = sub.add_parser("preds", help="make preds_<arch>_<seed>.jsonl for Part 3")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer_32000.json")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stop-ids", default="3", help="<|im_end|>=3 (per sft_template.txt)")
    p.add_argument("--id-field", default="id")
    p.add_argument("--prompt-field", default="prompt")
    p.add_argument("--max-tokens-field", default="max_new_tokens")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_preds)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
