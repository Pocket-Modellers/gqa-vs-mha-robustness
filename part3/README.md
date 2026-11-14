# Part 3: Behavioral (Brittleness) Evaluation

This folder measures how much the instruction-following of six small language models degrades when the prompt is reworded or reformatted. The six models are three trained with multi-head attention (MHA) and three with grouped-query attention (GQA), identical in size, data and training settings. Everything here runs on a CPU from the committed files. Only the optional log-probability step needs a GPU.

## Quick start

```
pip install -r requirements.txt
python collect_all.py
python compare_arch.py eval_set.jsonl
python three_metrics.py eval_set.jsonl
python extra_metrics.py eval_set.jsonl
python logprob_metrics.py eval_set.jsonl
python final_tables.py
```

Run the analysis scripts from this folder in the order shown. All random draws are seeded, and results are written to `final_summary.txt` and the `final_*.csv` files. The frozen evaluation set is stored here; `make_data.py` is kept in `provenance/` and writes regenerated data there.

## Contents

| Path | Description |
|------|-------------|
| `eval_set.jsonl` | Frozen evaluation set: 3,000 prompts (3 tasks x 200 items x 5 prompt formats). sha256 starts with `dd20736fbc32` |
| `preds_<arch>_<seed>.jsonl` | Greedy generations of each final SFT model on `eval_set.jsonl` (`{"id", "output"}` per line) |
| `logp_<arch>_<seed>.jsonl` | Label log-probabilities for the two closed tasks, one file per model (2,000 rows each) |
| `score.py` | Shared scoring functions (exact match, ROUGE-L) |
| `check_preds.py`, `check_eval.py` | Checks for prediction-file format and evaluation-file integrity |
| `collect_all.py`, `compare_arch.py`, `three_metrics.py`, `extra_metrics.py`, `logprob_metrics.py`, `final_tables.py` | The analysis chain |
| `brittleness_table.csv` | Per-model clean scores and percentage-point drops by perturbation type (Part 3 hand-off table) |
| `final_summary.txt` | Results, verdict, caveats, and input hashes |
| `final_metrics_seed.csv`, `final_metrics_arch.csv` | Every metric per model, and MHA versus GQA for every metric |
| `final_per_perturbation_seed.csv`, `final_per_perturbation_arch.csv` | Drops per perturbation, with bootstrap intervals and a Bonferroni-adjusted p |
| `final_per_model_ci.csv`, `final_ancova.csv` | Per-model scores with 95% bootstrap intervals, and the ANCOVA adjusted for clean accuracy |
| `kaggle/logp_cell.py` | Kaggle GPU cell that produced the `logp_*` files |
| `provenance/` | Scripts and inputs that built the evaluation set |

## Data

| Task | Source | Use |
|------|--------|-----|
| SST-2 (sentiment) | https://huggingface.co/datasets/nyu-mll/glue (config `sst2`) | 200 validation items for evaluation; 1,000 training items in the fine-tuning set |
| AG News (topic) | https://huggingface.co/datasets/fancyzhx/ag_news | 200 test items for evaluation; 1,000 training items in the fine-tuning set |
| Dolly 15k | https://huggingface.co/datasets/databricks/databricks-dolly-15k | 200 items from a 1,000-row held-out slice that was excluded from fine-tuning |

Perturbation conditions: clean, paraphrase, distractor (a neutral sentence prepended), and two reformatted layouts. The 200 Dolly paraphrases were drafted with a language model and checked manually. Evaluation templates and neutral distractors are recorded in `provenance/make_data.py` and the Dolly paraphrase file in `provenance/`.

## Metrics

All four decision metrics are "higher = more brittle".

| Metric | Definition |
|--------|------------|
| M1 mean drop | Mean over four perturbations of (clean score minus perturbed score), in points |
| M2 format spread | Highest minus lowest task score across the five formats |
| M3 answer change | Share of items whose predicted label changes (Dolly: 100 x (1 - ROUGE-L between clean and perturbed output)) |
| M4 log-probability shift | Mean fall in the margin between the correct label and the best wrong label, in nats (closed tasks only) |
| `S_*` columns | Supplementary: loss and gain rates, balanced accuracy, majority baseline, answer-collapse shares, first-token M4, and agreement checks |

Scoring: exact match on the first label word for the closed tasks, and ROUGE-L for Dolly. Uncertainty: item-level bootstrap intervals, plus exact permutation tests across seeds.

**Decision rule.** An architecture difference is claimed for a task only if all decision metrics available for it have the same sign of (MHA minus GQA) and the seeds separate completely on at least two of them. M4 exists for the closed tasks only, so the rule uses four metrics for AG News and three for Dolly. SST-2 is excluded because all six models scored 53.5 to 57.0 percent against a 50 percent chance level, below the 65 percent baseline gate.

**Chronology.** M1 had been seen before the rule was written. The three-metric rule was declared before M2 and M3 were computed. M4 and the `S_` metrics were added after the M1 to M3 results were seen, so they are reported in full and cannot overturn the verdict.

## Results (AG News, the primary task)

| Metric | MHA | GQA | MHA minus GQA | Permutation p | Seeds separate |
|--------|-----|-----|---------------|---------------|----------------|
| M1 mean drop | 17.67 | 15.29 | +2.38 | 0.60 | no |
| M2 format spread | 26.83 | 23.17 | +3.67 | 0.50 | no |
| M3 answer change | 28.46 | 28.38 | +0.08 | 1.00 | no |
| M4 log-probability shift (nats) | 0.662 | 0.537 | +0.125 | 0.40 | no |

Verdict under the rule: no detectable difference between MHA and GQA. Dolly gives the same verdict on M1 to M3. See `final_summary.txt` for every other metric, the ANCOVA, and the multiplicity check.

## Limitations

- Seeds are independent pretraining runs; fine-tuning used one fixed data order, so fine-tuning variation is not measured.
- With three seeds per architecture, the smallest possible permutation p-value is 0.10, and the four metrics are correlated.
- GQA has a wider feed-forward layer (2,432 versus 2,048) to match parameter counts, and a slightly higher pretraining loss. Higher clean accuracy goes with larger absolute drops, so absolute and relative drops are both reported.
- Closed-task scoring is exact match on the first label word; ROUGE-L is a crude measure for open-ended answers. A single score carries about 3.5 points of sampling noise with 200 items.

## Regenerating the inputs

- **Log-probabilities (GPU, needs the six final SFT checkpoints and Part 2's model code):** run `kaggle/logp_cell.py` in a Kaggle notebook with the checkpoints attached. Its AG News label strings must match the evaluation set (`technology`).
- **Evaluation set and fine-tuning additions:** `cd provenance`, then run `python make_sft_closed.py` and `python make_data.py` (needs internet). They write into `provenance/`. The regenerated `eval_set.jsonl` should hash to `dd20736fbc32`.
- **Predictions:** generated by Part 2's generation script (greedy decoding, stop at `<|im_end|>`, `max_new_tokens` from each row).



## SFT provenance files for Part 1 review

The `provenance/` folder includes the tokenizer used with the SFT outputs, Part 3's `sft_closed.jsonl` input, and the held-out data used by `make_data.py`. `sft_artifacts/train.jsonl.gz` is the compressed copy of the SFT `train.jsonl` output received with the Part 2 files. The held-out file has SHA-256 `C547A730C5C1BA3D1BD57D51287F88F6DB3F14F7A35C1FDA78C0DF38A2CBBE50`; the uncompressed train file has SHA-256 `8613ADC1C2DFBE41DCAC4940CCB7072FF061B7FA15D1F19F2B636CF56C909730`. The included tokenizer has SHA-256 `C7039AE373DB120B7F42EE139C17A8469EEB20552370CCD50456D920E12AA3A0`, and the included `sft_closed.jsonl` has SHA-256 `5B5FCAD066B5625BACCD25B9BCF712EE8EBF8E6794A8357F6767C84DC42D8553`.

These files let Part 1 compare the received SFT artifacts with the output of that script version; their inclusion alone does not establish which script version generated the training data.


