"""
Runs one job on Kaggle. Teammates never edit this file; they only change the
config line in the notebook.

    python kaggle_run.py --config configs/gqa_s0.yaml --hours 11.5               # pretraining
    python kaggle_run.py --config configs/gqa_s0.yaml --task sft                 # SFT + preds

--task sft: attach (1) the notebook output that contains ckpts/<run>/final.pt,
(2) a dataset with Santosh's sft/train.jsonl, and (3) Sree's eval_set.jsonl.
It fine-tunes the model with configs/sft.yaml and writes, as notebook output:
    ckpts/<run>_sft/final.pt      the fine-tuned model
    preds_<run>.jsonl             Sree's predictions file

What it does, in order:
  1. Finds our code, Santosh's tokenizer and shards, and any checkpoints from
     an earlier session, anywhere under /kaggle/input (so dataset folder names
     don't matter).
  2. Copies the code to /tmp/part2 and links the data there. Only the
     checkpoints folder lives in /kaggle/working, so only checkpoints are saved
     as the notebook's output (not 4GB of data again).
  3. If an earlier session's checkpoints are attached, copies them in, so
     training RESUMES instead of starting over.
  4. Trains, and stops itself before Kaggle's 12-hour limit so the output
     always gets saved. Run it again next session to continue.
"""

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml


def find_one(root, pattern, what):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        sys.exit(f"ERROR: could not find {what} ('{pattern}') under {root}. "
                 f"Did you attach the right dataset with 'Add Input'?")
    return hits[0]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)


def last_logged_step(run_dir):
    try:
        with open(os.path.join(run_dir, "log.csv")) as f:
            return int(f.readlines()[-1].split(",")[0])
    except (OSError, ValueError, IndexError):
        return -1


def copy_shards_local(src_dir, dst_dir, c):
    """
    /kaggle/input is slow for the many small random reads training does, so copy
    the shards to the machine's local disk first (one fast sequential copy).
    If there isn't enough local space, fall back to reading them in place, but
    pre-read every file once so the operating system caches it in memory.
    """
    files = sorted(set(glob.glob(os.path.join(src_dir, c["train_pattern"]))
                       + glob.glob(os.path.join(src_dir, c["val_pattern"]))))
    need = sum(os.path.getsize(p) for p in files)
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    t0 = time.time()
    if shutil.disk_usage(os.path.dirname(dst_dir)).free > need * 1.2:
        os.makedirs(dst_dir, exist_ok=True)
        for p in files:
            shutil.copy(p, os.path.join(dst_dir, os.path.basename(p)))
        print(f"copied {need / 1e9:.1f} GB of shards to local disk in {time.time() - t0:.0f}s")
    else:
        link(src_dir, dst_dir)
        for p in files:
            with open(p, "rb") as f:
                while f.read(64 << 20):
                    pass
        print(f"not enough local disk; pre-read {need / 1e9:.1f} GB into memory cache "
              f"in {time.time() - t0:.0f}s")


def quick_data_check(shard_dir, c):
    """Fast sanity check of Santosh's shards before spending GPU hours on them."""
    import numpy as np
    train = sorted(glob.glob(os.path.join(shard_dir, c["train_pattern"])))
    val = sorted(glob.glob(os.path.join(shard_dir, c["val_pattern"])))
    if not val:
        sys.exit(f"ERROR: no validation file matching '{c['val_pattern']}' in {shard_dir}")
    n_train = sum(os.path.getsize(p) // 2 for p in train)
    n_val = sum(os.path.getsize(p) // 2 for p in val)
    print(f"data:      {len(train)} train files, {n_train:,} tokens | "
          f"{len(val)} val files, {n_val:,} tokens")
    for p in train[:1] + val[:1]:  # check the first 5M tokens of one train and one val file
        a = np.memmap(p, dtype=np.uint16, mode="r")[:5_000_000]
        if int(a.max()) >= int(c["vocab_size"]):
            sys.exit(f"ERROR: {os.path.basename(p)} has token id {int(a.max())} >= vocab size")
    if n_train < float(c["train_tokens"]):
        print(f"WARNING: train_tokens ({float(c['train_tokens']):,.0f}) is more than the data; "
              f"some data will be seen twice")


def run_sft(args, run_name):
    """SFT one pretrained model, then generate Sree's predictions with it."""
    hits = glob.glob(os.path.join(args.input_root, "**", "ckpts", run_name, "final.pt"),
                     recursive=True)
    if not hits:
        sys.exit(f"ERROR: no ckpts/{run_name}/final.pt found. Attach the notebook output "
                 f"(version) where pretraining of {run_name} FINISHED.")
    pretrained = hits[0]
    train_jsonl = [p for p in glob.glob(os.path.join(args.input_root, "**", "train.jsonl"),
                                        recursive=True)]
    if not train_jsonl:
        sys.exit("ERROR: no train.jsonl found. Attach the dataset with Santosh's sft/train.jsonl.")
    train_jsonl = sorted(train_jsonl, key=lambda p: "sft" not in p)[0]  # prefer .../sft/train.jsonl
    eval_set = glob.glob(os.path.join(args.input_root, "**", "eval_set.jsonl"), recursive=True)
    print(f"pretrained: {pretrained}")
    print(f"sft data:   {train_jsonl}")
    print(f"eval set:   {eval_set[0] if eval_set else 'NOT FOUND (preds will be skipped)'}")

    link(train_jsonl, os.path.join(args.work_dir, "sft", "train.jsonl"))
    out_ckpts = os.path.join(args.working, "ckpts")
    os.makedirs(out_ckpts, exist_ok=True)
    link(out_ckpts, os.path.join(args.work_dir, "ckpts"))
    # sft.py names its output after the folder the pretrained model is in: <run>_sft
    local_pre = os.path.join(args.work_dir, "pretrained", run_name, "final.pt")
    link(pretrained, local_pre)

    try:
        import torch
        print(f"GPU:        {torch.cuda.device_count()} x "
              f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none!'}")
    except ImportError:
        pass
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    print(f"\n=== SFT {run_name} ===", flush=True)
    r = subprocess.run([sys.executable, "-u", "sft.py", "configs/sft.yaml", local_pre],
                       cwd=args.work_dir, env=env)
    sft_model = os.path.join(out_ckpts, f"{run_name}_sft", "final.pt")
    if r.returncode != 0 or not os.path.exists(sft_model):
        sys.exit("ERROR: SFT failed, scroll up for the error.")

    if eval_set:
        out_preds = os.path.join(args.working, f"preds_{run_name}.jsonl")
        print(f"\n=== predictions -> {os.path.basename(out_preds)} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", "generate.py", "preds", "--ckpt", sft_model,
                            "--tokenizer", "tokenizer_32000.json", "--eval-set", eval_set[0],
                            "--out", out_preds], cwd=args.work_dir, env=env)
        if r.returncode != 0:
            sys.exit("ERROR: generating predictions failed, scroll up for the error.")

    print("\n=========== STATUS ===========")
    print(f"{run_name}: SFT FINISHED -> ckpts/{run_name}_sft/final.pt")
    if eval_set:
        print(f"predictions: preds_{run_name}.jsonl (in this version's Output; send it to Sree)")


def read_config(code_dir, config):
    with open(os.path.join(code_dir, config)) as f:
        c = yaml.safe_load(f)
    base = {}
    if "base" in c:
        with open(os.path.join(code_dir, os.path.dirname(config), c["base"])) as f:
            base = yaml.safe_load(f)
    base.update(c)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. configs/gqa_s0.yaml")
    ap.add_argument("--hours", type=float, default=11.5, help="stop after this long")
    ap.add_argument("--task", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--work-dir", default="/tmp/part2")
    args = ap.parse_args()
    t_start = time.time()

    # ---- 1. code
    src_code = os.path.dirname(find_one(args.input_root, "pretrain.py", "our code"))
    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    shutil.copytree(src_code, args.work_dir)
    print(f"code:      {src_code}")

    # Kaggle sometimes flattens folders when it unpacks an upload, so the
    # configs/ folder can go missing. Collect every .yaml from our code dataset,
    # wherever it ended up, back into configs/.
    cfg_dir = os.path.join(args.work_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    dataset_root = src_code
    while os.path.dirname(dataset_root) not in (args.input_root, "/", dataset_root) and \
            not glob.glob(os.path.join(dataset_root, "**", "base.yaml"), recursive=True):
        dataset_root = os.path.dirname(dataset_root)  # look a level up if needed
    for y in glob.glob(os.path.join(dataset_root, "**", "*.yaml"), recursive=True):
        dst = os.path.join(cfg_dir, os.path.basename(y))
        if not os.path.exists(dst):
            shutil.copy(y, dst)
    wanted = os.path.join(args.work_dir, "configs", os.path.basename(args.config))
    if not os.path.exists(wanted):
        sys.exit(f"ERROR: config '{os.path.basename(args.config)}' is not in the code dataset. "
                 f"Found: {sorted(os.listdir(cfg_dir))}")
    args.config = os.path.join("configs", os.path.basename(args.config))

    c = read_config(args.work_dir, args.config)
    run_name = c["run_name"]

    # ---- 2. tokenizer and data
    tok = find_one(args.input_root, "tokenizer_32000.json", "the tokenizer")
    link(tok, os.path.join(args.work_dir, "tokenizer_32000.json"))
    print(f"tokenizer: {tok}")

    if args.task == "sft":
        return run_sft(args, run_name)

    data_dir = os.path.join(args.work_dir, c["data_dir"])
    if c["data_dir"].rstrip("/").endswith("tinystories"):
        if not os.path.exists(os.path.join(data_dir, "train.bin")):
            print("building TinyStories shards (needs Internet ON in notebook settings)...")
            subprocess.run([sys.executable, "prepare_tinystories.py", "--out-dir", data_dir],
                           cwd=args.work_dir, check=True)
    else:
        shard = find_one(args.input_root, c["train_pattern"], "the training shards")
        src_dir = os.path.dirname(shard)
        quick_data_check(src_dir, c)
        print(f"data:      {src_dir}")
        copy_shards_local(src_dir, data_dir, c)

    # ---- 3. checkpoints: output folder + restore from an earlier session
    out_ckpts = os.path.join(args.working, "ckpts")
    os.makedirs(out_ckpts, exist_ok=True)
    link(out_ckpts, os.path.join(args.work_dir, c["out_dir"]))
    run_out = os.path.join(out_ckpts, run_name)
    previous = glob.glob(os.path.join(args.input_root, "**", "ckpts", run_name, "latest.pt"),
                         recursive=True)
    if previous and not os.path.exists(os.path.join(run_out, "latest.pt")):
        # if several earlier sessions are attached, take the one that got furthest
        prev_dir = max((os.path.dirname(p) for p in previous), key=last_logged_step)
        shutil.copytree(prev_dir, run_out, dirs_exist_ok=True)
        print(f"restored earlier session from {prev_dir} -> will RESUME")
    elif os.path.exists(os.path.join(run_out, "final.pt")):
        print("this run already has final.pt")
    else:
        print("no earlier session found -> starting from scratch")

    # ---- 4. GPU check
    n_gpus = 0
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        if n_gpus:
            print(f"GPU:       {n_gpus} x {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: no GPU! Set Accelerator to 'GPU T4' in notebook settings.")
    except ImportError:
        pass

    # ---- 5. train, stopping before the time limit
    budget = args.hours * 3600 - (time.time() - t_start)
    print(f"\ntraining {run_name} on {max(n_gpus, 1)} GPU(s) for at most "
          f"{budget / 3600:.1f} more hours\n", flush=True)
    env = dict(os.environ,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",   # less memory waste
               STOP_AT_UNIX=str(time.time() + budget))               # training stops itself then
    if n_gpus >= 2:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={n_gpus}", "pretrain.py", args.config]
    else:
        cmd = [sys.executable, "-u", "pretrain.py", args.config]
    proc = subprocess.Popen(cmd, cwd=args.work_dir, env=env)
    try:
        proc.wait(timeout=budget + 600)   # normally training stops itself at the budget
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not os.path.exists(os.path.join(run_out, "final.pt")):
        saved = os.path.exists(os.path.join(run_out, "latest.pt"))
        print("\nSTOPPED BEFORE THE END." + (" Progress is saved in latest.pt." if saved else
                                            " Training had not saved anything yet."))

    # ---- 6. status
    print("\n=========== STATUS ===========")
    if os.path.exists(os.path.join(run_out, "final.pt")):
        print(f"{run_name}: FINISHED. final.pt is in this notebook's output.")
    else:
        print(f"{run_name}: NOT finished yet. Run again next session with this")
        print("notebook's output attached as input; it will resume automatically.")
    log = os.path.join(run_out, "log.csv")
    if os.path.exists(log):
        with open(log) as f:
            print("last log line:", f.readlines()[-1].strip())
    if proc.returncode not in (0, None, -signal.SIGINT) and not os.path.exists(
            os.path.join(run_out, "final.pt")):
        print(f"training exited with code {proc.returncode}: scroll up for the error.")


if __name__ == "__main__":
    main()
