"""
Supervised fine-tuning (SFT): teaches a pretrained model to answer instructions.

    python sft.py configs/sft.yaml ckpts/gqa_s0/final.pt

Input:   a pretrained checkpoint + Santosh's sft/train.jsonl
Output:  ckpts/gqa_s0_sft/final.pt   (this is the model Sree's preds come from)

How it works:
  Each example is   [prompt tokens][response tokens][<|im_end|>]
  The model sees the whole thing, but the loss is ONLY computed on the response
  and <|im_end|>. Prompt positions get label -100, which the loss ignores.
  So the model learns to WRITE answers, not to predict the questions.

Fairness rules (same as pretraining): every model gets the exact same SFT
settings, the same examples in the same order (sft_seed is fixed in
configs/sft.yaml), and the same number of steps. Run the same command for
every pretrained model and change nothing but the checkpoint path.

Accepted rows in train.jsonl:
  - Dolly style: {"instruction", "context", "response", ...}, formatted with chat_format.py
  - pre-tokenized: {"input_ids": [...], "labels": [...]} with -100 on prompt positions
"""

import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

from chat_format import PAD_ID, format_prompt, format_response
from checkpoint import load_model, save_checkpoint
from pretrain import evaluate, load_config, lr_at, make_optimizer, pick_precision


# ---------------------------------------------------------------- data
def tokenize_rows(rows, tok, max_len):
    """Returns a list of (input_ids, labels) lists, already shifted by one."""
    examples, n_skipped = [], 0
    for row in rows:
        if "input_ids" in row and "labels" in row:
            ids, labels = list(row["input_ids"]), list(row["labels"])
        else:
            p = tok.encode(format_prompt(row)).ids  # as in 03_prepare_sft.py
            r = tok.encode(format_response(row)).ids
            ids = p + r
            labels = [-100] * len(p) + r
        ids, labels = ids[: max_len + 1], labels[: max_len + 1]
        if all(l == -100 for l in labels[1:]):  # answer cut off entirely
            n_skipped += 1
            continue
        # next-token prediction: position t reads ids[t] and must predict labels[t+1]
        examples.append((ids[:-1], labels[1:]))
    return examples, n_skipped


def collate(batch):
    """Pad a list of examples to the same length. Padding is ignored by the loss."""
    T = max(len(x) for x, _ in batch)
    xs = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    ys = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : len(x)] = torch.tensor(x)
        ys[i, : len(y)] = torch.tensor(y)
    return xs, ys


# ---------------------------------------------------------------- main
def main(config_path, pretrained_path):
    c = load_config(config_path)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(c["tokenizer"])

    device = c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype, use_scaler = pick_precision(device)
    sft_seed = int(c["sft_seed"])
    torch.manual_seed(sft_seed)

    model, info = load_model(pretrained_path, device)
    model.train()
    mcfg = model.cfg
    run_name = os.path.basename(os.path.dirname(pretrained_path)) + "_sft"
    run_dir = os.path.join(c["out_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ----- data: identical split and order for every model
    with open(c["train_file"]) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    examples, n_skipped = tokenize_rows(rows, tok, mcfg.max_seq_len)
    order = np.random.default_rng(1234).permutation(len(examples))
    n_val = int(len(examples) * float(c["val_fraction"]))
    val_ex = [examples[i] for i in order[:n_val]]
    train_ex = [examples[i] for i in order[n_val:]]

    B, accum, epochs = int(c["batch_size"]), int(c["grad_accum"]), int(c["epochs"])
    steps_per_epoch = math.ceil(len(train_ex) / (B * accum))
    total_steps = steps_per_epoch * epochs
    val_batches = [collate(val_ex[i : i + B]) for i in range(0, len(val_ex), B)]

    print(f"SFT {pretrained_path} -> {run_dir}")
    print(f"{len(rows):,} rows -> {len(train_ex):,} train / {len(val_ex):,} val "
          f"({n_skipped} skipped: too long)")
    print(f"{steps_per_epoch} steps/epoch x {epochs} epochs = {total_steps} steps")
    n_resp = sum(sum(l != -100 for l in y) for _, y in train_ex[:200])
    n_all = sum(len(y) for _, y in train_ex[:200])
    print(f"loss is on {100 * n_resp / n_all:.0f}% of tokens (the responses)")

    optimizer = make_optimizer(model, c)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    log_file = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "sec"])

    step, t0 = 0, time.time()
    val_loss = evaluate(model, val_batches, device, dtype)
    print(f"before SFT: val loss {val_loss:.4f}")
    for epoch in range(epochs):
        perm = np.random.default_rng([sft_seed, epoch]).permutation(len(train_ex))
        for s in range(steps_per_epoch):
            lr = lr_at(step, total_steps, c)
            for g in optimizer.param_groups:
                g["lr"] = lr
            idx = perm[s * B * accum : (s + 1) * B * accum]
            micro_batches = [idx[i : i + B] for i in range(0, len(idx), B)]
            n_tokens = sum(sum(l != -100 for l in train_ex[j][1]) for j in idx)
            loss_sum = 0.0
            for mb in micro_batches:
                x, y = collate([train_ex[j] for j in mb])
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    _, loss = model(x, y)
                # weight each micro-batch by its share of answer tokens in the step
                weight = (y != -100).sum().item() / max(1, n_tokens)
                scaler.scale(loss * weight).backward()
                loss_sum += loss.item() * weight
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            val_loss = None
            if step % int(c["eval_every"]) == 0 or step == total_steps:
                val_loss = evaluate(model, val_batches, device, dtype)
            if step % int(c["log_every"]) == 0 or val_loss is not None:
                msg = f"step {step:5d}/{total_steps} | epoch {epoch} | loss {loss_sum:.4f} | lr {lr:.2e}"
                if val_loss is not None:
                    msg += f" | VAL {val_loss:.4f}"
                print(msg + f" | {time.time() - t0:.0f}s", flush=True)
                logger.writerow([step, epoch, f"{loss_sum:.5f}",
                                 "" if val_loss is None else f"{val_loss:.5f}",
                                 f"{lr:.3e}", f"{time.time() - t0:.1f}"])
                log_file.flush()

    final_path = os.path.join(run_dir, "final.pt")
    save_checkpoint(final_path, model, mcfg, step, info["tokens_seen"],
                    run_config={"sft": c, "pretrained_from": pretrained_path})
    print(f"done. SFT model: {final_path}")
    log_file.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python sft.py configs/sft.yaml ckpts/<run>/final.pt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
