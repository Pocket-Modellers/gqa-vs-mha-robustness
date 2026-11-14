"""
Pretraining script. One run = one YAML config file.

    python pretrain.py configs/gqa_s0.yaml

What it does:
  1. Reads the config (per-run file + the base.yaml it points to).
  2. Builds the model. For GQA/MQA with match_params: true, it widens the
     feed-forward layer so the parameter count matches MHA.
  3. Trains with AdamW, linear warmup + cosine decay.
  4. Every eval_every steps: computes validation loss on a FIXED set of
     tokens (identical for every run) and logs it.
  5. Saves weight-only checkpoints at fixed token counts (same for every run,
     for Part 4) and a latest.pt for resuming.
  6. If latest.pt already exists, it resumes from there automatically.

Logs go to ckpts/<run_name>/log.csv, and to Weights & Biases if use_wandb: true.

Two GPUs (e.g. Kaggle "GPU T4 x2"):
    torchrun --standalone --nproc_per_node=2 pretrain.py configs/gqa_s0.yaml
Each GPU processes half of the micro-batches of every step, and the gradients
are averaged, so a step sees exactly the same 524,288 tokens as on one GPU.
The training is the same; it just runs about twice as fast. kaggle_run.py
does this automatically when it sees two GPUs.

Time limit: if the environment variable STOP_AT_UNIX is set (a unix time),
training stops cleanly before then and saves latest.pt.
"""

import csv
import math
import os
import signal
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint import load_for_resume, save_checkpoint
from data import TokenShards, find_shards
from model import ModelConfig, Transformer, matched_ffn_hidden


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:  # per-run file points at a shared base file
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base"))
        base = load_config(base_path)
        base.update(cfg)  # per-run values override base values
        cfg = base
    return cfg


def build_model_config(c):
    mcfg = ModelConfig(
        vocab_size=int(c["vocab_size"]),
        d_model=int(c["d_model"]),
        n_layers=int(c["n_layers"]),
        n_heads=int(c["n_heads"]),
        n_kv_heads=int(c["n_kv_heads"]),
        ffn_hidden=int(c["ffn_hidden"]),
        max_seq_len=int(c["seq_len"]),
    )
    if c.get("match_params", True) and mcfg.n_kv_heads != mcfg.n_heads:
        mcfg.ffn_hidden = matched_ffn_hidden(mcfg, int(c["ffn_hidden"]))
    return mcfg


# ---------------------------------------------------------------- helpers
def lr_at(step, total_steps, c):
    """Linear warmup, then cosine decay down to min_lr_ratio * lr."""
    lr, warmup = float(c["lr"]), int(c["warmup_steps"])
    min_lr = lr * float(c["min_lr_ratio"])
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model, c):
    # weight decay on matrices only, not on norms
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": float(c["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=float(c["lr"]), betas=(0.9, 0.95), eps=1e-8,
                             fused=fused)


def pick_precision(device):
    if device != "cuda":
        return torch.float32, False
    # Native bf16 needs compute capability 8.0+ (A100, L4, H100, RTX 30xx/40xx).
    # Newer PyTorch says the T4 "supports" bf16, but it is only emulated there
    # (slow), so check the hardware generation instead.
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16, False
    return torch.float16, True          # T4, P100 (Colab/Kaggle): needs GradScaler


@torch.no_grad()
def evaluate(model, batches, device, dtype):
    model.eval()
    losses = []
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"   # "cuda:1" -> "cuda"
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=dev_type, dtype=dtype, enabled=dev_type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------- main
def main(config_path):
    c = load_config(config_path)
    run_name = c["run_name"]
    run_dir = os.path.join(c["out_dir"], run_name)

    # ----- one GPU, or several GPUs launched with torchrun
    ddp = "RANK" in os.environ
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device_str = f"cuda:{local_rank}"
        else:
            device_str = "cpu"
    else:
        rank, world = 0, 1
        device_str = c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if device_str.startswith("cuda") else "cpu"   # for autocast
    main_proc = rank == 0

    def log(*a, **k):
        if main_proc:
            print(*a, **k, flush=True)

    if main_proc:
        os.makedirs(run_dir, exist_ok=True)
    dtype, use_scaler = pick_precision(device)
    seed = int(c["seed"])
    data_seed = int(c.get("data_seed", seed))
    torch.manual_seed(seed)  # controls model initialisation (same on every GPU)

    # ----- data
    train = TokenShards(find_shards(c["data_dir"], c["train_pattern"]))
    val = TokenShards(find_shards(c["data_dir"], c["val_pattern"]))
    B, T, accum = int(c["batch_size"]), int(c["seq_len"]), int(c["grad_accum"])
    assert accum % world == 0, f"grad_accum ({accum}) must divide by the number of GPUs ({world})"
    local_accum = accum // world
    tokens_per_step = B * T * accum
    total_steps = int(float(c["train_tokens"]) // tokens_per_step)
    val_batches = val.fixed_batches(int(c["eval_batches"]), B, T)

    # ----- model
    mcfg = build_model_config(c)
    model = Transformer(mcfg).to(device_str)
    optimizer = make_optimizer(model, c)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ----- resume if possible (every GPU loads the same file)
    step, tokens_seen = 0, 0
    latest_path = os.path.join(run_dir, "latest.pt")
    if c.get("resume", True) and os.path.exists(latest_path):
        step, tokens_seen = load_for_resume(latest_path, model, optimizer, device_str)
        log(f"Resumed from {latest_path} at step {step}")

    train_model = torch.compile(model) if c.get("compile", False) else model
    if ddp:
        train_model = DDP(train_model, device_ids=[local_rank] if device == "cuda" else None)

    # ----- checkpoint schedule (identical for every run)
    ckpt_tokens = sorted(set(
        [int(float(t)) for t in c.get("ckpt_extra_tokens", [])]
        + list(range(int(float(c["ckpt_every_tokens"])), int(float(c["train_tokens"])) + 1,
                     int(float(c["ckpt_every_tokens"]))))
    ))
    ckpt_tokens = [t for t in ckpt_tokens if t > tokens_seen]

    # ----- logging
    log(f"run={run_name} device={device_str} gpus={world} dtype={dtype}")
    log(f"params: {model.num_params():,} total, "
        f"{model.num_params(include_embeddings=False):,} without embeddings")
    log(f"config: {mcfg}")
    log(f"train tokens available: {train.total_tokens:,}  |  "
        f"tokens per step: {tokens_per_step:,}  |  total steps: {total_steps:,}")
    if float(c["train_tokens"]) > train.total_tokens:
        log("WARNING: train_tokens is more than the data has; data will repeat.")

    wandb, logger, log_file = None, None, None
    if main_proc:
        if c.get("use_wandb", False):
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=c["wandb_project"], name=run_name, config=c,
                       id=run_name, resume="allow")
        log_path = os.path.join(run_dir, "log.csv")
        new_log = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        logger = csv.writer(log_file)
        if new_log:
            logger.writerow(["step", "tokens_seen", "train_loss", "val_loss", "lr", "grad_norm", "sec"])

    # ----- stop cleanly on Ctrl+C / notebook Stop button / time limit:
    # finish the current step, save latest.pt, then exit. Nothing is lost.
    stop = {"requested": False}

    def request_stop(signum, frame):
        if stop["requested"]:
            raise KeyboardInterrupt  # pressed twice: stop immediately
        stop["requested"] = True
        log("\nstop requested: finishing this step and saving latest.pt ...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_at = float(os.environ.get("STOP_AT_UNIX", "inf"))

    def should_stop():
        """True if any GPU wants to stop. All GPUs must agree, or they'd hang."""
        want = stop["requested"] or time.time() > stop_at
        if ddp:
            flag = torch.tensor([1.0 if want else 0.0], device=device_str)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            want = flag.item() > 0
        return want

    # ----- training loop
    model.train()
    t0 = time.time()
    stopping = False
    while step < total_steps and not stopping:
        lr = lr_at(step, total_steps, c)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_sum = torch.zeros(1, device=device_str)
        t_step, t_data = time.time(), 0.0
        for i in range(local_accum):
            micro = i * world + rank   # GPU r takes micro-batches r, r+world, r+2*world, ...
            t_d = time.time()
            x, y = train.get_batch(data_seed, step, micro, B, T)
            t_data += time.time() - t_d   # time spent reading data (should be tiny)
            x, y = x.to(device_str, non_blocking=True), y.to(device_str, non_blocking=True)
            # only sync gradients between GPUs on the last micro-batch (faster)
            sync = (not ddp) or i == local_accum - 1
            ctx = train_model.no_sync() if (ddp and not sync) else _null()
            with ctx:
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    _, loss = train_model(x, y)
                scaler.scale(loss / local_accum).backward()   # DDP then averages over GPUs
            loss_sum += loss.detach() / local_accum
        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        loss_val = loss_sum.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"])).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += tokens_per_step
        stopping = should_stop()
        step_sec = time.time() - t_step

        # ----- log / eval (validation runs on the main GPU only)
        val_loss = None
        is_last = step == total_steps
        if main_proc and (step % int(c["eval_every"]) == 0 or is_last):
            val_loss = evaluate(model, val_batches, device_str, dtype)
        if main_proc and (step % int(c["log_every"]) == 0 or val_loss is not None):
            dt = time.time() - t0
            msg = f"step {step:6d}/{total_steps} | loss {loss_val:.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f}"
            if val_loss is not None:
                msg += f" | VAL {val_loss:.4f}"
            log(msg + f" | {step_sec:.1f}s/step (data {t_data:.1f}s) | {dt:.0f}s")
            logger.writerow([step, tokens_seen, f"{loss_val:.5f}",
                             "" if val_loss is None else f"{val_loss:.5f}",
                             f"{lr:.3e}", f"{grad_norm:.4f}", f"{dt:.1f}"])
            log_file.flush()
            if wandb:
                d = {"train/loss": loss_val, "lr": lr, "grad_norm": grad_norm, "tokens_seen": tokens_seen}
                if val_loss is not None:
                    d["val/loss"] = val_loss
                wandb.log(d, step=step)

        # ----- checkpoints (main GPU writes; all GPUs hold identical weights)
        while ckpt_tokens and tokens_seen >= ckpt_tokens[0]:
            ckpt_tokens.pop(0)
            if main_proc:
                path = os.path.join(run_dir, f"step_{step:06d}.pt")
                save_checkpoint(path, model, mcfg, step, tokens_seen, run_config=c)
                log(f"saved {path}")
        if main_proc and (step % int(c["save_latest_every"]) == 0 or is_last or stopping):
            save_checkpoint(latest_path, model, mcfg, step, tokens_seen,
                            optimizer=optimizer, run_config=c)
        if ddp:
            dist.barrier()   # nobody runs ahead while a checkpoint is being written

    if step < total_steps:
        log(f"stopped at step {step}/{total_steps}. latest.pt saved; "
            f"run the same command again to resume.")
    else:
        if main_proc:
            final_path = os.path.join(run_dir, "final.pt")
            save_checkpoint(final_path, model, mcfg, step, tokens_seen, run_config=c)
            log(f"done. final model: {final_path}")
    if log_file:
        log_file.close()
    if wandb:
        wandb.finish()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python pretrain.py configs/<run>.yaml")
        sys.exit(1)
    main(sys.argv[1])
