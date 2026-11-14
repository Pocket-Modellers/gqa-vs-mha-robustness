"""Three brittleness metrics, compared across architectures (MHA vs GQA).

Usage:  python three_metrics.py [eval_set.jsonl]
Reads every preds_<arch>_<seed>.jsonl in the current folder.
Writes three_metrics_seed.csv (per model) and three_metrics_arch.csv (MHA vs GQA).

All three metrics are "higher = more brittle", 0-100 scale:
  M1 mean drop        mean over perturbations of (clean score - perturbed score)
  M2 format spread    max - min of the task score across the five prompt formats
                      (clean, paraphrase, distractor, reformat, reformat2)
  M3 answer change    closed tasks: % of items whose predicted label differs from the
                      clean prompt's label (flip rate), averaged over perturbations.
                      Dolly: 100 x (1 - ROUGE-L between clean and perturbed output).

diff = MHA - GQA. The hypothesis (GQA more brittle) predicts diff < 0 on every metric.

PRE-DECLARED DECISION RULE (fix this before looking at M2 and M3):
  Claim an architecture difference for a task only if
    (a) all three metrics have the same sign of diff, AND
    (b) the seeds separate completely (every seed of one architecture beyond every seed
        of the other) on at least two of the three metrics.
  Otherwise report "no detectable difference / inconclusive".
  Primary task: AG News. Dolly is secondary and judged separately. SST-2 is excluded
  by the baseline gate.
"""
import glob, os, sys
from itertools import combinations
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from score import load, item_score

EVAL = sys.argv[1] if len(sys.argv) > 1 else "eval_set.jsonl"
A, B = "mha", "gqa"
FORMATS = ["clean", "paraphrase", "distractor", "reformat", "reformat2"]
GATES = {"sst2": 65.0, "agnews": 45.0}
METRICS = [("M1_mean_drop", "mean drop"), ("M2_format_spread", "format spread"),
           ("M3_answer_change", "answer change")]
_rouge = rouge_scorer.RougeScorer(["rougeL"])


def perm_p(a, b):
    """Exact two-sided permutation p-value for a difference in means (small samples)."""
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


def change_rouge(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return 0.0
    return 100.0 * (1.0 - _rouge.score(a, b)["rougeL"].fmeasure)


data = pd.DataFrame(load(EVAL))
rows = []
for path in sorted(glob.glob("preds_*.jsonl")):
    name = os.path.basename(path)[len("preds_"):-len(".jsonl")]
    arch = name.split("_")[0].lower()
    if arch not in (A, B):
        print("skipping", name, "(architecture must be mha or gqa)")
        continue
    preds = pd.DataFrame(load(path))
    df = data.merge(preds, on="id")
    df["score"] = [item_score(m, o, g) * 100 for m, o, g in zip(df.metric, df.output, df.gold)]
    for task, tdf in df.groupby("task"):
        w = tdf.pivot(index="item", columns="type", values="score")
        o = tdf.pivot(index="item", columns="type", values="output")
        fmts = [f for f in FORMATS if f in w.columns]
        perts = [f for f in fmts if f != "clean"]
        m1 = w[perts].rsub(w["clean"], axis=0).mean(axis=1).mean()
        means = w[fmts].mean()
        m2 = means.max() - means.min()
        if tdf["metric"].iloc[0] == "exact":
            labs = sorted(set(tdf["gold"]))
            L = o.apply(lambda col: col.map(lambda s: label_of(s, labs)))
            ch = [(L[p] != L["clean"]).mean() * 100 for p in perts]
        else:
            ch = [np.mean([change_rouge(o["clean"][i], o[p][i]) for i in o.index]) for p in perts]
        rows.append(dict(model=name, arch=arch, task=task, clean=w["clean"].mean(),
                         M1_mean_drop=m1, M2_format_spread=m2, M3_answer_change=float(np.mean(ch))))
seed = pd.DataFrame(rows)
seed.to_csv("three_metrics_seed.csv", index=False)

for arch in (A, B):
    n = seed[seed.arch == arch].model.nunique()
    print(f"{arch}: {n} model(s)")
    if n == 0:
        sys.exit(f"No predictions found for {arch}.")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n=== Per model ===")
print(seed.round(1).to_string(index=False))

excluded = set()
for task, gate in GATES.items():
    c = seed[seed.task == task]
    if not c.empty and (c["clean"] < gate).any():
        excluded.add(task)
        print(f"\n*** {task}: baseline gate failed (clean below {gate:.0f}% for some model). EXCLUDED. ***")

out = []
for task in sorted(seed.task.unique()):
    if task in excluded:
        continue
    g = seed[seed.task == task]
    signs, seps = [], 0
    print(f"\n=== {task}: MHA vs GQA (diff = MHA - GQA; hypothesis predicts diff < 0) ===")
    for col, label in METRICS:
        a, b = g[g.arch == A][col], g[g.arch == B][col]
        diff = a.mean() - b.mean()
        sep = bool((a.min() > b.max()) or (b.min() > a.max()))
        p = perm_p(a, b)
        signs.append(np.sign(diff))
        seps += int(sep)
        out.append(dict(task=task, metric=col, mha_mean=a.mean(), mha_min=a.min(), mha_max=a.max(),
                        gqa_mean=b.mean(), gqa_min=b.min(), gqa_max=b.max(),
                        diff_mha_minus_gqa=diff, perm_p=p, seeds_separated=sep))
        print(f"  {label:14s} MHA {a.mean():6.2f} [{a.min():.1f}, {a.max():.1f}] | "
              f"GQA {b.mean():6.2f} [{b.min():.1f}, {b.max():.1f}] | diff {diff:+6.2f} | "
              f"perm p {p:.2f} | seeds separated: {'yes' if sep else 'no'}")
    same = all(s == signs[0] for s in signs) and signs[0] != 0
    direction = ("all three point to GQA more brittle" if signs[0] < 0 else
                 "all three point to MHA more brittle") if same else "mixed: no consistent direction"
    claimable = same and seps >= 2
    print(f"  Direction: {direction}. Metrics with separated seeds: {seps} of 3.")
    print("  Verdict under the pre-declared rule: " +
          ("an architecture difference is CLAIMABLE." if claimable else
           "NOT claimable. Report no detectable difference / inconclusive."))
pd.DataFrame(out).to_csv("three_metrics_arch.csv", index=False)
print("\nNotes: with 3 vs 3 seeds the smallest possible perm p is 0.10. The three metrics are "
      "correlated, so agreement between them is weaker evidence than three independent tests.")