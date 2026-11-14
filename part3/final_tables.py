"""Merge every metric into one set of final tables and evaluate the decision rule.

Usage:  python final_tables.py
Run the other scripts first. Needs in the current folder:
  three_metrics_seed.csv, extra_metrics_seed.csv, logprob_metrics_seed.csv,
  extra_metrics_ancova.csv, seed_table.csv, arch_comparison.csv, brittleness_table.csv
Writes:
  final_metrics_seed.csv            one row per model and task, every metric
  final_metrics_arch.csv            MHA vs GQA for every metric
  final_per_perturbation_seed.csv   drop and relative drop per model, task and perturbation
  final_per_perturbation_arch.csv   MHA vs GQA per perturbation (bootstrap CI, permutation p)
  final_per_model_ci.csv            per-model scores and drops with 95% bootstrap intervals
  final_ancova.csv                  ANCOVA: MHA vs GQA adjusted for clean accuracy (from extra_metrics.py)
  final_summary.txt                 coverage, gate, results, verdict, caveats

Naming: M1..M4 are the decision metrics, M4b is the probability version of M4, and columns
starting with S_ are supplementary. All are reported. All are "higher = more brittle"
unless stated, and diff = MHA - GQA (the hypothesis predicts diff < 0).

DECISION RULE (v2, declared before M4 was computed; it is stricter than v1):
  Claim an architecture difference for a task only if all decision metrics available for it
  (M1 mean drop, M2 format spread, M3 answer change, M4 log-probability shift) have the same sign
  of (MHA - GQA) AND the seeds separate completely (every seed of one architecture beyond every
  seed of the other) on at least two of them. Otherwise: no detectable difference.
  M4 is computed for the closed tasks only, so the rule uses four metrics for AG News and
  three (M1-M3) for Dolly.
  Adding a condition can only make a claim harder, so v2 cannot turn the earlier
  "not claimable" verdict on M1-M3 into a claim.
"""
import glob, hashlib, os, shutil, sys
from itertools import combinations
import numpy as np
import pandas as pd

A, B = "mha", "gqa"
GATES = {"sst2": 65.0, "agnews": 45.0}
DECISION = ["M1_mean_drop", "M2_format_spread", "M3_answer_change", "M4_logprob_shift"]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def perm_p(a, b):
    v = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    n, k = len(v), len(a)
    obs = abs(v[:k].mean() - v[k:].mean())
    hits = total = 0
    for idx in combinations(range(n), k):
        mask = np.zeros(n, bool)
        mask[list(idx)] = True
        total += 1
        hits += abs(v[mask].mean() - v[~mask].mean()) >= obs - 1e-9
    return hits / total


for f in ["three_metrics_seed.csv", "extra_metrics_seed.csv", "extra_metrics_ancova.csv", "seed_table.csv",
          "arch_comparison.csv", "brittleness_table.csv"]:
    if not os.path.exists(f):
        sys.exit(f"Missing {f}. Run the earlier scripts first.")

seed = pd.read_csv("three_metrics_seed.csv")
extra = pd.read_csv("extra_metrics_seed.csv").drop(columns=["arch", "clean", "net_drop"])
extra = extra.rename(columns={c: "S_" + c for c in extra.columns if c not in ("model", "task")})
seed = seed.merge(extra, on=["model", "task"], how="left")
have_m4 = os.path.exists("logprob_metrics_seed.csv")
if have_m4:
    lp = pd.read_csv("logprob_metrics_seed.csv").drop(columns=["arch"])
    lp = lp.rename(columns={c: "S_" + c for c in ("margin_clean", "nll_clean", "agree_clean") if c in lp.columns})
    seed = seed.merge(lp, on=["model", "task"], how="left")
else:
    print("NOTE: logprob_metrics_seed.csv not found. M4 is missing, so the rule is evaluated on M1-M3 only.\n")
seed.to_csv("final_metrics_seed.csv", index=False)
shutil.copy("seed_table.csv", "final_per_perturbation_seed.csv")
shutil.copy("extra_metrics_ancova.csv", "final_ancova.csv")
pp = pd.read_csv("arch_comparison.csv")
pp["bonferroni_p"] = (pp["perm_p"] * len(pp)).clip(upper=1.0)
pp.to_csv("final_per_perturbation_arch.csv", index=False)
shutil.copy("brittleness_table.csv", "final_per_model_ci.csv")

decision = [m for m in DECISION if m in seed.columns]
numeric = [c for c in seed.columns if c not in ("model", "arch", "task") and pd.api.types.is_numeric_dtype(seed[c])]
constant = [c for c in numeric if seed[c].dropna().nunique() <= 1]          # e.g. the majority-label baseline
supp = [c for c in numeric if c not in decision and c not in constant]
KEY_SUPP = [c for c in ["clean", "M4b_pcorrect_drop", "S_cond_loss", "S_gain_rate", "S_bal_drop"] if c in supp]

# gate
failed = {t: seed[(seed.task == t) & (seed.clean < g)].model.tolist() for t, g in GATES.items()}
failed = {t: m for t, m in failed.items() if m}

# architecture table
rows = []
for task, g in seed.groupby("task"):
    for col in decision + supp:
        a, b = g[g.arch == A][col].dropna(), g[g.arch == B][col].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        rows.append(dict(task=task, metric=col, group="decision" if col in decision else "supplementary",
                         gate="failed" if task in failed else "passed",
                         mha_mean=a.mean(), mha_min=a.min(), mha_max=a.max(),
                         gqa_mean=b.mean(), gqa_min=b.min(), gqa_max=b.max(),
                         diff_mha_minus_gqa=a.mean() - b.mean(), perm_p=perm_p(a, b),
                         seeds_separated=bool((a.min() > b.max()) or (b.min() > a.max())),
                         n_mha=len(a), n_gqa=len(b)))
arch = pd.DataFrame(rows)
arch.to_csv("final_metrics_arch.csv", index=False)

# summary
n_mha, n_gqa = seed[seed.arch == A].model.nunique(), seed[seed.arch == B].model.nunique()
say("PART 3 FINAL SUMMARY")
say("=" * 70)
say(f"Models: {n_mha} MHA and {n_gqa} GQA. Decision metrics available: {', '.join(decision)}.")
if not have_m4:
    say("WARNING: M4 (log-probability shift) is missing.")
if failed:
    for t, m in failed.items():
        say(f"Baseline gate FAILED for {t} (clean below {GATES[t]:.0f}%): {', '.join(m)}. Excluded from the verdict.")
say("")
say("Legend (all higher = more brittle unless stated)")
say("  M1_mean_drop       mean of (clean score - perturbed score) over four perturbations, points")
say("  M2_format_spread   highest minus lowest score across the five prompt formats")
say("  M3_answer_change   share of items whose predicted label changes (Dolly: 100 x (1 - ROUGE-L))")
say("  M4_logprob_shift   mean fall in the log-prob margin of the correct label, nats (closed tasks only)")
say("  M4b_pcorrect_drop  mean fall in the probability of the correct label, points (closed tasks only)")
say("  S_cond_loss / S_gain_rate / S_L_*  loss rate L (correct on clean, wrong after), gain rate G, and L per perturbation")
say("  S_bal_clean / S_bal_drop  balanced accuracy (mean per-label recall) and its drop")
say("  S_majority_baseline       accuracy of always guessing the most common label (not a brittleness measure)")
say("  S_top_answer_<format>     share of outputs given to the single most common answer (high = collapse)")
say("  S_M4_first_token / S_margin_clean / S_agree_clean  first-token version of M4, clean margin, label agreement")
say("")
pd.set_option("display.width", 220)
for task in sorted(seed.task.unique()):
    sub = arch[(arch.task == task)]
    if sub.empty:
        continue
    full = sub
    sub = sub[(sub.group == "decision") | (sub.metric.isin(KEY_SUPP))]
    say(f"--- {task}: MHA vs GQA, mean [min, max] over seeds (diff = MHA - GQA; hypothesis predicts < 0) ---")
    for _, r in sub.iterrows():
        tag = "" if r.group == "decision" else "   (supplementary)"
        say(f"  {r.metric:20s} MHA {r.mha_mean:7.3f} [{r.mha_min:.2f}, {r.mha_max:.2f}] | "
            f"GQA {r.gqa_mean:7.3f} [{r.gqa_min:.2f}, {r.gqa_max:.2f}] | diff {r.diff_mha_minus_gqa:+7.3f} | "
            f"perm p {r.perm_p:.2f} | separated {'yes' if r.seeds_separated else 'no '}{tag}")
    rest = full[(full.group == "supplementary") & (~full.metric.isin(KEY_SUPP))]
    if len(rest):
        say(f"  All other supplementary metrics for {task} (also in final_metrics_arch.csv):")
        for _, r in rest.iterrows():
            say(f"    {r.metric:26s} MHA {r.mha_mean:8.3f} | GQA {r.gqa_mean:8.3f} | "
                f"diff {r.diff_mha_minus_gqa:+8.3f} | perm p {r.perm_p:.2f}")
    if task in failed:
        say("  Verdict: not evaluated (baseline gate failed).")
        say("")
        continue
    d = full[full.group == "decision"]
    signs = np.sign(d.diff_mha_minus_gqa.values)
    same = len(d) > 0 and bool(np.all(signs == signs[0])) and signs[0] != 0
    seps = int(d.seeds_separated.sum())
    direction = (("same sign on all metrics, GQA higher" if signs[0] < 0 else "same sign on all metrics, MHA higher")
                 + " (a sign alone is not evidence)") if same else "mixed signs"
    claim = same and seps >= 2
    say(f"  Direction: {direction}. Separated seeds on {seps} of {len(d)} decision metrics.")
    say("  Verdict (rule v2): " + ("an architecture difference is CLAIMABLE." if claim else
                                   "NOT claimable. No detectable difference."))
    say("")
say("Every metric above is also in final_metrics_arch.csv (MHA vs GQA) and final_metrics_seed.csv (per model).")
say("")
say("ANCOVA (exploratory): MHA minus GQA after adjusting for clean accuracy")
anc = pd.read_csv("final_ancova.csv")
for _, r in anc.iterrows():
    say(f"  {r.task} / {r.metric}: {r.mha_minus_gqa_adjusted:+.2f} (95% CI [{r.ci_lo:.1f}, {r.ci_hi:.1f}], "
        f"df = {int(r.df)}, slope on clean = {r.slope_on_clean:+.2f})")
if anc.empty:
    say("  (no task passed the baseline gate)")
say("")
say(f"Multiplicity: {len(pp)} per-perturbation comparisons in final_per_perturbation_arch.csv "
    f"(bonferroni_p = min(1, perm_p x {len(pp)}))")
for _, r in pp.sort_values("perm_p").head(5).iterrows():
    say(f"  {r.task} / {r.perturbation}: perm p {r.perm_p:.2f}, Bonferroni p {r.bonferroni_p:.2f}")
closed = seed[seed.task.isin(["agnews", "sst2"]) & seed["S_cond_loss"].notna()]
if len(closed):
    err = (closed.clean / 100 * closed.S_cond_loss - (1 - closed.clean / 100) * closed.S_gain_rate
           - closed.M1_mean_drop).abs().max()
    say(f"Identity check, net drop = c*L - (1-c)*G, max abs error over closed-task rows: {err:.6f} pp")
say("")
say("Inputs used (real files only; sha256 prefix and size)")
for pat in ["eval_set.jsonl", "preds_*.jsonl", "logp_*.jsonl"]:
    for p in sorted(glob.glob(pat)):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
        say(f"  {p:28s} {h}  {os.path.getsize(p):>10,d} bytes")
say("")
say("Caveats")
say("  - Seeds are independent pretraining runs; SFT used one fixed data order.")
say("  - With 3 vs 3 seeds the smallest possible permutation p is 0.10; the four metrics are correlated.")
say("  - GQA has a wider FFN and a slightly higher pretraining loss; larger drops go with higher clean accuracy.")
say("  - M4 is computed for the closed tasks only; the Dolly verdict uses M1-M3.")
say("  - Chronology: M1 had been seen before the decision rule was written. The three-metric rule (M1-M3) was")
say("    declared before M2 and M3 were computed. M4 and the S_ metrics were added after the M1-M3 results were")
say("    seen, so they are reported in full and cannot overturn the verdict.")
say("  - Dolly metrics are secondary: ROUGE-L is crude, and its M3 and M4 are defined differently from the closed tasks.")
open("final_summary.txt", "w", encoding="utf-8").write("\n".join(lines))