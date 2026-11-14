# gqa-vs-mha-robustness
# Does sharing keys make transformers fragile? MHA vs GQA

**Team Pocket Modellers.** AFML course project.

Grouped-query attention (GQA) lets several query heads share one key/value head. It cuts the KV cache and is now standard in large language models. We ask whether that sharing has a hidden cost: is a GQA model more sensitive to how a prompt is worded, and does it solve tasks with fewer, more concentrated internal components?

To test this we trained **six ~110M-parameter language models from scratch**, three with multi-head attention (MHA) and three with GQA. Attention is the only deliberate difference between them. We then fine-tuned all six identically, measured how much their answers change under prompt perturbations, and ablated attention heads to see how concentrated their circuits are.

## Headline results

| Question | Result |
|---|---|
| Does GQA train as well as MHA? | Almost. Every GQA seed ends slightly above every MHA seed: pretraining validation loss 3.481 vs 3.464 (mean of 3 seeds). |
| Is GQA more brittle to prompt changes? | **No detectable difference.** On AG News, MHA vs GQA mean accuracy drop is 17.7 vs 15.3 points (permutation p = 0.60). Dolly gives the same verdict. |
| Are GQA circuits more concentrated? | **Not consistently.** One GQA seed relies on a single head group (91% of the IOI margin), but the other two GQA seeds look like the MHA models. |
| Does concentration predict brittleness? | No clear link. The correlations change sign between tasks, and with 6 models the analysis is exploratory. |

## Pipeline

```
Part 1  data + tokenizer      FineWeb-Edu shards, 32k BPE tokenizer, SFT mix
   │
Part 2  model + training      6 models (MHA / GQA × 3 seeds), 1B tokens each, then SFT
   │                          checkpoints ──────────────────────────────┐
Part 3  brittleness           3,000 prompts × 6 models, 4 metrics       │
   │                                                                    │
Part 4  circuit audit         IOI task, head and group ablation ◄───────┘
                              joined with Part 3 drops
```

| Folder | What it contains | Owner |
|---|---|---|
| [`part1_data_tokenizer/`](part1_data_tokenizer) | Tokenizer training, FineWeb-Edu sharding, SFT data mix and chat template | Balla Santhosh |
| [`part2_model_pretraining/`](part2_model_pretraining) | Llama-style model, pretraining, SFT, prediction generation, loss curves | Pranav Kalla |
| [`part3/`](part3) | Evaluation set, scoring, brittleness metrics, statistical tests, final verdict | Bhogala Srika |
| [`part4_circuit_audit/`](part4_circuit_audit) | IOI circuit audit, ablation results, concentration vs brittleness | Nainika Batchu |

Each folder has its own README with full details and reproduction steps.

## The six models

| | MHA | GQA |
|---|---|---|
| Layers / width / context | 12 / 768 / 1,024 | 12 / 768 / 1,024 |
| Query heads | 12 | 12 |
| Key/value heads | 12 | 3 (each shared by 4 query heads) |
| Feed-forward hidden size | 2,048 | 2,432 (widened to match parameters) |
| Parameters | 109,529,856 | 109,529,856 |
| Seeds | 0, 1, 2 | 0, 1, 2 |

Architecture: RMSNorm, RoPE, SwiGLU, tied embeddings, no biases. Each model was pretrained on 1B tokens and fine-tuned for 2 epochs with an identical recipe. Runs with the same seed saw exactly the same batches. Everything ran on free Kaggle 2× T4 GPUs (about 70 GPU-hours in total).

## Datasets

All data is public:

| Dataset | Used for | Link |
|---|---|---|
| FineWeb-Edu (sample-10BT) | Tokenizer and pretraining | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| Databricks Dolly 15k | SFT, plus a 1,000-row held-out slice for evaluation | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| Alpaca (cleaned) | SFT | https://huggingface.co/datasets/yahma/alpaca-cleaned |
| SST-2 (GLUE) | SFT and evaluation | https://huggingface.co/datasets/nyu-mll/glue |
| AG News | SFT and evaluation | https://huggingface.co/datasets/fancyzhx/ag_news |
| IOI prompts | Circuit audit | Generated synthetically in the Part 4 notebook |

The final SFT set has 65,832 rows (`part3/provenance/sft_artifacts/train.jsonl.gz`). The frozen evaluation set has 3,000 prompts (`part3/eval_set.jsonl`): 3 tasks × 200 items × 5 prompt formats.

## Reproducing

```
# Part 2: check the model code and parameter counts (CPU, seconds)
cd part2_model_pretraining
pip install -r requirements.txt
python -m pytest tests -q        # 11 passed
python count_params.py           # both 109,529,856

# Part 3: rerun the full brittleness analysis from the committed predictions (CPU)
cd ../part3
pip install -r requirements.txt
python collect_all.py
python compare_arch.py eval_set.jsonl
python three_metrics.py eval_set.jsonl
python extra_metrics.py eval_set.jsonl
python logprob_metrics.py eval_set.jsonl
python final_tables.py           # writes final_summary.txt
```

Pretraining, SFT and the circuit audit need GPUs. See the Part 2 and Part 4 READMEs. The checkpoints are about 440 MB each, too large for GitHub, so they are kept as Kaggle datasets (listed in the Part 2 README).

## Limitations

- Only three seeds per architecture. The smallest possible permutation p-value is 0.10, so only large, consistent effects could be detected.
- Matching the parameter counts widened the GQA feed-forward layer. GQA therefore differs from MHA in head sharing plus slightly more feed-forward capacity.
- SST-2 was excluded from the brittleness verdict because all six models scored only 53–57% on clean prompts, below the 65% baseline gate.
- Small models (110M parameters, 1B tokens) are far from the scale where GQA is used in practice.
- The circuit audit measured concentration on the pretrained models, while brittleness was measured after SFT.
