# Part 2: Model, Pretraining and Fine-tuning

This folder holds the code that built and trained the six models compared in this project: three with multi-head attention (MHA) and three with grouped-query attention (GQA). Attention is the only deliberate difference between them. The same code also fine-tuned all six models (SFT) and generated the prediction files that Part 3 scores. The code here is exactly the version that produced every checkpoint.

## Results

All six runs trained stably to comparable loss. GQA is slightly but consistently worse: every GQA seed is above every MHA seed.

| Run | Pretrain val loss | SFT train loss | SFT val loss |
|-----|------------------:|---------------:|-------------:|
| mha_s0 | 3.4627 | 1.965 | 2.2603 |
| mha_s1 | 3.4611 | 1.965 | 2.2626 |
| mha_s2 | 3.4687 | 1.973 | 2.2678 |
| gqa_s0 | 3.4774 | 1.980 | 2.2778 |
| gqa_s1 | 3.4774 | 1.976 | 2.2740 |
| gqa_s2 | 3.4872 | 1.985 | 2.2872 |
| **MHA mean** | **3.4642** | **1.968** | **2.2636** |
| **GQA mean** | **3.4807** | **1.980** | **2.2797** |

- Pretrain val loss: final loss after 1B tokens, on the same fixed validation tokens for every run.
- SFT train loss: mean of the logged steps 1920–2000. The last step (2018) is a small leftover batch and is left out because its loss is noisy.
- SFT val loss: final loss on the same fixed 2% held-back split of the SFT data (1,316 examples).
- gqa_s0 and gqa_s1 finishing at the same pretraining loss is a coincidence; they are separate runs (val loss at step 1,900: 3.4774 vs 3.4790).

The same numbers are in `results.csv`.

## Model

A Llama-style decoder-only transformer: RMSNorm (pre-norm), rotary position embeddings (RoPE), SwiGLU feed-forward layers, no biases, and tied input/output embeddings.

| | MHA | GQA |
|---|---|---|
| Layers / width / context | 12 / 768 / 1,024 | 12 / 768 / 1,024 |
| Query heads (dim 64) | 12 | 12 |
| Key/value heads | 12 | 3 (each shared by 4 query heads) |
| Feed-forward hidden size | 2,048 | 2,432 |
| Vocabulary | 32,000 (Part 1 tokenizer) | 32,000 |
| Parameters | 109,529,856 | 109,529,856 |

GQA has smaller key/value projections, so its feed-forward layer is widened to give exactly the same parameter count. GQA therefore differs from MHA in head sharing plus a slightly wider feed-forward layer.

## Training

**Pretraining:** FineWeb-Edu shards from Part 1, 1B tokens per run (1,907 steps × 524,288 tokens). AdamW (β 0.9, 0.95, weight decay 0.1), peak learning rate 6e-4 with 200 warmup steps and cosine decay to 6e-5, gradient clipping at 1.0, fp16 mixed precision on 2× NVIDIA T4 (Kaggle), data-parallel. About 10.5 hours per run.

**Controls:** runs with the same seed see exactly the same batches in the same order, validation uses the same fixed tokens for every run, and all runs share `configs/base.yaml`. Three seeds per architecture.

**Fine-tuning (SFT):** Part 1's `sft/train.jsonl` (65,832 rows), ChatML template, loss on the response tokens only. Full fine-tuning for 2 epochs (2,018 steps, 64 examples per step, max 512 tokens), learning rate 1e-4 with 50 warmup steps and cosine decay to 1e-5, no weight decay, fixed data order. Identical for all six models. About 1 hour per run, including generation.

**Predictions for Part 3:** greedy decoding on Part 3's 3,000 prompts, stopping at `<|im_end|>` (id 3), written as `preds_<arch>_<seed>.jsonl`.

**Checkpoints:** every run saved weights at 25M, 50M, 100M, 200M, 400M, 600M and 800M tokens plus the final model (files `step_000048.pt` … `step_001526.pt`, `final.pt`), used by Part 4.

## Contents

| Path | Description |
|------|-------------|
| `model.py` | The model. `Attention` is the only part that differs between MHA and GQA. Each layer exposes a `head_out` hook on the per-head outputs; `ablate_heads()` is a zero-ablation example |
| `data.py` | Reads the uint16 token shards and cuts deterministic batches |
| `checkpoint.py` | `save_checkpoint`, and `load_model(path)` to rebuild any checkpoint |
| `pretrain.py` | Pretraining loop (single GPU or both T4s via torchrun), with automatic resume |
| `sft.py` | Supervised fine-tuning with response-only loss |
| `chat_format.py` | ChatML template, identical to Part 1's `sft_template.txt` |
| `generate.py` | `sample` to generate text; `preds` to make Part 3's prediction files |
| `kaggle_run.py`, `kaggle_notebook.ipynb` | Kaggle launcher for pretraining (`TASK = "pretrain"`) and SFT + predictions (`TASK = "sft"`) |
| `configs/` | `base.yaml` (shared settings), one file per run, `sft.yaml`, and the TinyStories smoke test |
| `count_params.py`, `check_shards.py`, `prepare_tinystories.py` | Parameter counts, shard checks, smoke-test data |
| `tests/test_part2.py` | 11 checks, e.g. GQA equals MHA with copied K/V weights, no attention to future tokens, exact checkpoint reload |
| `results.csv` | The results table above |

## Reproducing

```
pip install -r requirements.txt
python -m pytest tests -q        # 11 passed
python count_params.py           # MHA and GQA both 109,529,856

# pretrain one model (needs Part 1's shards in data/fineweb and tokenizer_32000.json)
python pretrain.py configs/gqa_s0.yaml
# on two GPUs
torchrun --standalone --nproc_per_node=2 pretrain.py configs/gqa_s0.yaml

# fine-tune it and generate Part 3's predictions
python sft.py configs/sft.yaml ckpts/gqa_s0/final.pt
python generate.py preds --ckpt ckpts/gqa_s0_sft/final.pt --tokenizer tokenizer_32000.json \
    --eval-set eval_set.jsonl --out preds_gqa_s0.jsonl
```

On Kaggle, upload this folder as a dataset (Kaggle flattens folders; `kaggle_run.py` rebuilds `configs/`), import `kaggle_notebook.ipynb`, attach the code and shard datasets, set the accelerator to GPU T4 x2, and set the `RUN` and `TASK` lines.

## Checkpoints

The trained weights are too large for GitHub (about 440 MB per checkpoint). They are stored as private Kaggle datasets under `pranavkalla`: `mha-s0-ckpts`, `mha-s1-ckpts` (version 1 is mha_s1; the latest version is gqa_s2), `mha-s2-ckpts`, `gqa-s0-ckpts`, `gqa-s1-ckpts` for the pretrained runs, and `sft-runner-{mha,gqa}-s{0,1,2}-datset` for the fine-tuned models. Load any of them with:

```python
from checkpoint import load_model
model, info = load_model("ckpts/gqa_s0_sft/final.pt", device="cuda")
```

## Notes

- gqa_s0 hit Kaggle's 12-hour session limit at step 1,848 and was resumed from its saved state for the last 59 steps; the fp16 loss scaler restarted at its default value for those steps. All other runs finished in one session.
- The configs for a third arm, multi-query attention (`n_kv_heads=1`), were written but not run for compute reasons, so they are not included.
- Compute: about 63 GPU-hours of pretraining and 6 GPU-hours of fine-tuning, all on free Kaggle T4s.
