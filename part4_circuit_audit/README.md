# Part 4 — Circuit Audit of MHA and GQA

This audit tests how much six pretrained models rely on particular attention components to solve a synthetic Indirect Object Identification (IOI) task. The comparison covers three Multi-Head Attention (MHA) models and three Grouped-Query Attention (GQA) models, one per training seed for each architecture.

## IOI task and ablation method

The IOI generator builds prompts with two names, an object, and a location. The model scores two candidate continuations. For example, after “Tom gave the book to Bob. Then Tom gave another book to”, the correct continuation is **Bob** and the distractor is **Tom**. We measure accuracy and the mean logit margin between the correct and distractor tokens. The notebook also includes an early IOI check on a mid-training MHA checkpoint.

During inference, the ablation hooks replace selected query-head activations with their mean activations. The grouped audit tests blocks of four query heads per layer. For GQA, each block corresponds to query heads that share a key/value group. MHA uses the same block size for comparison.

The cumulative audit ranks groups by the margin drop from ablating each group alone, then ablates them together in that order. In the exported CSVs, `heads_to_80` counts how many ranked **groups** were needed to reduce the baseline mean margin by at least 80%. `top_head_score` is the fraction of the baseline mean margin removed by ablating the strongest group alone.

## Pretrained results

| Model | Baseline IOI accuracy | Groups to 80% margin reduction | Margin reduction from strongest group |
|---|---:|---:|---:|
| MHA seed 0 | 99.4% | 3 | 58.8% |
| MHA seed 1 | 98.0% | 2 | 55.7% |
| MHA seed 2 | 94.6% | 3 | 57.0% |
| GQA seed 0 | 99.6% | 1 | 91.4% |
| GQA seed 1 | 94.8% | 3 | 60.7% |
| GQA seed 2 | 97.4% | 3 | 58.7% |

GQA seed 0 had the strongest dependence on a single group. Ablating that group reduced its mean margin by 91.4%. The other two GQA seeds each needed three groups to reach the 80% threshold, so the one-group result did not repeat across GQA seeds.

See the [pretrained results summary](final_pretrained_results_summary.csv), [concentration scores](six_model_concentration_summary.csv), [cumulative damage plot](cumulative_damage_all_models.png), [groups-to-80% comparison](components_to_80_comparison.png), and [strongest-group comparison](top_component_damage_comparison.png).

## Inference sanity check and SFT

In the targeted seed-0 verification runs, ablating layer 6 head 8 reduced MHA accuracy from 99% to 76%, a drop of 23 percentage points. Ablating layer 10 head 4 reduced GQA accuracy from 100% to 94%, a drop of 6 percentage points. These runs measure individual query heads. See the [targeted comparison](mha_vs_gqa_comparison.csv) and [key ablation results](key_ablation_results.csv).

The six supervised fine-tuned models reached **99.0–100.0%** baseline IOI accuracy. See the [SFT baseline results](sft_baseline_results.csv), [pretrained vs. SFT accuracy plot](accuracy_pretrained_vs_sft.png), and [pretrained vs. SFT margin plot](margin_pretrained_vs_sft.png).

## Concentration and brittleness

The concentration–brittleness correlation is pending. It requires joining the circuit-audit scores with Part 3’s perturbation-induced accuracy drops by architecture and seed.

## Reproducing the audit

Open the [circuit-audit notebook](part4-circuit-audit-mha-vs-gqa%20%284%29.ipynb) and attach the project model code, tokenizer, and checkpoints referenced in its Kaggle inputs. The notebook contains the prompt generator, baseline evaluation, ablation hooks, and result export steps.
