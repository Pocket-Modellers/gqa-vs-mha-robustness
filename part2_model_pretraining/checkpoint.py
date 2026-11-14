"""
Saving and loading checkpoints. Parts 3 and 4 only ever need `load_model`.

Two kinds of files are written per run (in ckpts/<run_name>/):

  step_000500.pt   Model weights + config + step + tokens_seen.
                   Small. These are what you hand to Parts 3 and 4.
  latest.pt        Same, PLUS optimizer state. Only used to resume training
                   if it crashes or Colab/Kaggle disconnects.

Usage for teammates:
    from checkpoint import load_model
    model, info = load_model("ckpts/gqa_s0/step_004000.pt")
    print(info["config"], info["tokens_seen"])
"""

import os

import torch

from model import ModelConfig, Transformer


def _strip_compile_prefix(state_dict):
    # torch.compile wraps the model and adds "_orig_mod." to every key
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def save_checkpoint(path, model, cfg: ModelConfig, step, tokens_seen,
                    optimizer=None, run_config=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model": _strip_compile_prefix(model.state_dict()),
        "config": cfg.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "run_config": run_config,   # the full training YAML, for the record
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a broken file


def load_model(path, device="cpu"):
    """Returns (model, info). The model is in eval mode on `device`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    info = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer")}
    return model, info


def load_for_resume(path, model, optimizer, device="cpu"):
    """Loads weights and optimizer state in place. Returns (step, tokens_seen)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["tokens_seen"]
