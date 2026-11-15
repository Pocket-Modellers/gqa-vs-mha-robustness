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

We joined the pretrained circuit-audit scores to Part 3’s **post-SFT** perturbation accuracy drops by architecture and seed. Across the six models, the AG News `ALL` summary gave a Spearman correlation of **+0.51** for groups needed to reach an 80% margin reduction and **−0.31** for the strongest-group score. These estimates are exploratory given the six-model sample. See the AG News rows in the [joined results](concentration_vs_brittleness.csv).

Part 3 excluded SST-2 because all models scored below the 65% clean-accuracy gate. The reported correlations therefore use AG News only.

**Limitation.** Concentration and brittleness were measured at different training stages. The correlations describe how pretrained circuit concentration relates to post-SFT brittleness across corresponding model runs. A same-stage comparison would require concentration measurements from the SFT checkpoints.

## Reproducing the audit

Open [part4_circuit_audit.ipynb](part4_circuit_audit.ipynb) and start with its run guide. It identifies the cells for checkpoint verification and plotting from saved results, and labels the historical cells that previously stopped with errors or interruptions. The revised notebook has not been rerun end to end.

For `mha_s1`, attach **version 1** of the Kaggle dataset `mha-s1-ckpts` and run the checkpoint inventory before starting the audit. The saved inventory found `ckpts/mha_s1/final.pt` with 12 query heads and 12 KV heads. The separate `gqa_s2` checkpoint had 3 KV heads. The original notebook did not record the Kaggle dataset version, so the saved output verifies the architecture and path, while the exact version used remains unverified.
