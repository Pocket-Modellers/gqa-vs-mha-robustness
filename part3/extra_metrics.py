"""Extra brittleness checks for the closed-answer tasks (AG News, SST-2).

Usage:  python extra_metrics.py [eval_set.jsonl]
Reads every preds_<arch>_<seed>.jsonl in the current folder.
Writes extra_metrics_seed.csv, extra_metrics_arch.csv and extra_metrics_ancova.csv.

SUPPLEMENTARY AND EXPLORATORY. The pre-declared primary outcome and decision rule are
unchanged. Do not use these checks to rescue or overturn a conclusion after the fact.

B. Conditional drop. Let c = clean accuracy,
     L = P(wrong under perturbation | correct on clean)    (loss rate, higher = more brittle)
     G = P(correct under perturbation | wrong on clean)    (gain rate)
   net drop = c*L - (1-c)*G. Reporting L and G separately shows errors that cancel in the
   net drop. L is a partial control for baseline ability, not a complete one: a weaker
   model's correct items may be its easiest ones.
D. Balanced accuracy (mean per-label recall), the majority-label baseline, and the share
   of outputs given to the single most common answer, per format.
E. Matched baseline. Regress the pooled drop on clean accuracy and an architecture
   indicator across models (ANCOVA). With 6 models this has 3 degrees of freedom, so the
   interval is wide. Exploratory.
"""
import glob, os, sys
from itertools import combinations
import numpy as np
import pandas as pd
from score import load

EVAL = sys.argv[1] if len(sys.argv) > 1 else "eval_set.jsonl"
A, B = "mha", "gqa"
FORMATS = ["clean", "paraphrase", "distractor", "reformat", "reformat2"]
TASKS = ["agnews", "sst2"]
GATES = {"sst2": 65.0, "agnews": 45.0}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228}


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


def label_of(output, labels):
    o = str(output).strip().lower()
    for lab in labels:
        if o.startswith(lab):
            return lab
    return "other:" + o[:12]


data = pd.DataFrame(load(EVAL))
rows = []
for path in sorted(glob.glob("preds_*.jsonl")):
    name = os.path.basename(path)[len("preds_"):-len(".jsonl")]
    arch = name.split("_")[0].lower()
    if arch not in (A, B):
        print("skipping", name, "(architecture must be mha or gqa)")
        continue
    df = data.merge(pd.DataFrame(load(path)), on="id")
    for task in TASKS:
        tdf = df[df.task == task]
        if tdf.empty:
            continue
        labs = sorted(set(tdf["gold"]))
        pred = (tdf.pivot(index="item", columns="type", values="output")
                .apply(lambda col: col.map(lambda s: label_of(s, labs))))
        gold = tdf[tdf.type == "clean"].set_index("item")["gold"].reindex(pred.index)
        correct = pred.apply(lambda col: col == gold)
        fmts = [f for f in FORMATS if f in correct.columns]
        perts = [f for f in fmts if f != "clean"]
        on = correct["clean"]
        acc = {f: correct[f].mean() * 100 for f in fmts}
        row = dict(model=name, arch=arch, task=task, clean=acc["clean"],
                   net_drop=np.mean([acc["clean"] - acc[p] for p in perts]))
        Ls, Gs, bal_drops = [], [], []
        for p in perts:
            L = (~correct.loc[on, p]).mean() * 100 if on.any() else np.nan
            G = correct.loc[~on, p].mean() * 100 if (~on).any() else np.nan
            row[f"L_{p}"] = L
            Ls.append(L)
            Gs.append(G)
        row["cond_loss"] = float(np.nanmean(Ls))
        row["gain_rate"] = float(np.nanmean(Gs))
        bal = {}
        for f in fmts:
            bal[f] = np.mean([(pred.loc[gold == l, f] == l).mean() for l in labs]) * 100
            row[f"top_answer_{f}"] = pred[f].value_counts(normalize=True).iloc[0] * 100
        row["bal_clean"] = bal["clean"]
        row["bal_drop"] = float(np.mean([bal["clean"] - bal[p] for p in perts]))
        row["majority_baseline"] = gold.value_counts(normalize=True).iloc[0] * 100
        rows.append(row)
seed = pd.DataFrame(rows)
seed.to_csv("extra_metrics_seed.csv", index=False)

for arch in (A, B):
    if seed[seed.arch == arch].empty:
        sys.exit(f"No predictions found for {arch}.")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
show = ["model", "task", "clean", "net_drop", "cond_loss", "gain_rate", "bal_clean",
        "bal_drop", "majority_baseline"]
print("=== Per model: net drop, loss rate (L), gain rate (G), balanced accuracy ===")
print(seed[show].round(1).to_string(index=False))
chk = (seed["clean"] / 100 * seed["cond_loss"] - (1 - seed["clean"] / 100) * seed["gain_rate"]
       - seed["net_drop"]).abs().max()
print(f"\nIdentity check, net drop = c*L - (1-c)*G (max abs error, pp): {chk:.6f}")

print("\n=== Most common answer share by format (high = collapse to one answer) ===")
tcols = ["model", "task"] + [f"top_answer_{f}" for f in FORMATS]
print(seed[tcols].round(1).to_string(index=False))

excluded = set()
for task, gate in GATES.items():
    c = seed[seed.task == task]
    if not c.empty and (c["clean"] < gate).any():
        excluded.add(task)
        print(f"\n*** {task}: baseline gate failed. Excluded from architecture comparisons. ***")

out, anc = [], []
for task in sorted(seed.task.unique()):
    if task in excluded:
        continue
    g = seed[seed.task == task]
    print(f"\n=== {task}: MHA vs GQA (diff = MHA - GQA; hypothesis predicts diff < 0) ===")
    for col, label in [("net_drop", "net drop"), ("cond_loss", "loss rate L"),
                       ("bal_drop", "balanced-acc drop"), ("gain_rate", "gain rate G (descriptive)")]:
        a, b = g[g.arch == A][col], g[g.arch == B][col]
        sep = bool((a.min() > b.max()) or (b.min() > a.max()))
        p = perm_p(a, b)
        out.append(dict(task=task, metric=col, mha_mean=a.mean(), gqa_mean=b.mean(),
                        diff_mha_minus_gqa=a.mean() - b.mean(), perm_p=p, seeds_separated=sep))
        print(f"  {label:26s} MHA {a.mean():6.2f} [{a.min():.1f}, {a.max():.1f}] | "
              f"GQA {b.mean():6.2f} [{b.min():.1f}, {b.max():.1f}] | diff {a.mean() - b.mean():+6.2f} | "
              f"perm p {p:.2f} | separated: {'yes' if sep else 'no'}")
    n = len(g)
    if n >= 4:
        for col, label in [("net_drop", "net drop"), ("cond_loss", "loss rate L")]:
            X = np.column_stack([np.ones(n), g["clean"].values, (g.arch == B).astype(float).values])
            y = g[col].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            df_ = n - 3
            s2 = (res @ res) / df_
            se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
            t = T95.get(df_, 2.0)
            d, lo, hi = -beta[2], -beta[2] - t * se, -beta[2] + t * se
            anc.append(dict(task=task, metric=col, n_models=n, df=df_, slope_on_clean=beta[1],
                            mha_minus_gqa_adjusted=d, ci_lo=lo, ci_hi=hi))
            print(f"  ANCOVA {label:11s} MHA - GQA at equal clean accuracy: {d:+.2f} "
                  f"(95% CI [{lo:.1f}, {hi:.1f}], df = {df_}, slope on clean = {beta[1]:+.2f})")
pd.DataFrame(out).to_csv("extra_metrics_arch.csv", index=False)
pd.DataFrame(anc).to_csv("extra_metrics_ancova.csv", index=False)

if os.path.exists("arch_comparison.csv"):
    ac = pd.read_csv("arch_comparison.csv")
    m = len(ac)
    ac["bonferroni_p"] = (ac["perm_p"] * m).clip(upper=1.0)
    print(f"\n=== Multiplicity: {m} comparisons in arch_comparison.csv ===")
    print(ac.sort_values("perm_p")[["task", "perturbation", "perm_p", "bonferroni_p"]]
          .head(5).round(2).to_string(index=False))
    print("A row at the smallest possible p (0.10 for 3 vs 3) cannot survive correction over this many rows.")
print("\nAll results here are exploratory.")
