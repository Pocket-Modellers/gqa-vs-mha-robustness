"""Compare MHA and GQA brittleness across seeds.

Usage:  python compare_arch.py [eval_set.jsonl]
Reads every preds_<arch>_<seed>.jsonl in the current folder (arch = mha or gqa).
Writes seed_table.csv (per model) and arch_comparison.csv (MHA vs GQA).

diff = MHA drop - GQA drop (percentage points). The hypothesis (GQA more brittle)
predicts diff < 0.

Columns of arch_comparison.csv
  ci95             bootstrap interval over test items (test-set noise only)
  perm_p           exact permutation p-value over seeds (training-seed noise).
                   Smallest possible value is 2 / C(n_mha + n_gqa, n_mha): 0.10 for 3 vs 3.
  mha_rel/gqa_rel  drop as a percentage of the clean score
  perm_p_rel       permutation p-value for the relative drop
  seeds_separated  every seed of one architecture lies beyond every seed of the other
"""
import glob, os, sys
from itertools import combinations, permutations
import numpy as np
import pandas as pd
from score import load, item_score

EVAL = sys.argv[1] if len(sys.argv) > 1 else "eval_set.jsonl"
A, B = "mha", "gqa"
N_BOOT = 1000
# Baseline gate: a task is compared only if EVERY model's clean score reaches the gate
# (clean accuracy clearly above chance). Dolly (ROUGE-L) has no gate.
GATES = {"sst2": 65.0, "agnews": 45.0}
rng = np.random.default_rng(0)


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


def corr(x, y):
    """Pearson r and an exact permutation p-value (only for n <= 8)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    if len(x) > 8:
        return r, np.nan
    hits = total = 0
    for p in permutations(range(len(x))):
        total += 1
        hits += abs(np.corrcoef(x, y[list(p)])[0, 1]) >= abs(r) - 1e-9
    return r, hits / total


data = pd.DataFrame(load(EVAL))

# ---- score every model, item by item ----
scores = {}   # model name -> {task: items x conditions DataFrame}
for path in sorted(glob.glob("preds_*.jsonl")):
    name = os.path.basename(path)[len("preds_"):-len(".jsonl")]
    arch = name.split("_")[0].lower()
    if arch not in (A, B):
        print("skipping", name, "(architecture must be mha or gqa)")
        continue
    preds = pd.DataFrame(load(path))
    df = data.merge(preds, on="id")
    df["score"] = [item_score(m, o, g) * 100
                   for m, o, g in zip(df.metric, df.output, df.gold)]
    scores[name] = {task: tdf.pivot(index="item", columns="type", values="score")
                    for task, tdf in df.groupby("task")}


def arch_of(name):
    return name.split("_")[0].lower()


for arch in (A, B):
    names = [n for n in scores if arch_of(n) == arch]
    print(f"{arch}: {len(names)} model(s) -> {names}")
    if not names:
        sys.exit(f"No predictions found for {arch}.")


def item_drop(w, t):
    """Per-item drop (clean minus perturbed). t == 'ALL' averages all conditions."""
    perts = [c for c in w.columns if c != "clean"]
    if t == "ALL":
        return w[perts].rsub(w["clean"], axis=0).mean(axis=1)
    if t not in w.columns:
        return None
    return w["clean"] - w[t]


# ---- per-model table ----
rows = []
for name, tasks in scores.items():
    for task, w in tasks.items():
        perts = [c for c in w.columns if c != "clean"]
        for t in perts + ["ALL"]:
            d = item_drop(w, t).dropna()
            clean = w["clean"].mean()
            rows.append(dict(model=name, arch=arch_of(name), task=task, perturbation=t,
                             clean=clean, drop=d.mean(),
                             rel_drop=(d.mean() / clean * 100) if clean > 0 else np.nan))
seed_tbl = pd.DataFrame(rows)
seed_tbl.to_csv("seed_table.csv", index=False)

# ---- baseline gate ----
excluded = {}
for task, gate in GATES.items():
    c = seed_tbl[(seed_tbl.task == task) & (seed_tbl.perturbation == "ALL")]
    if c.empty:
        continue
    failing = c[c["clean"] < gate]
    if not failing.empty:
        excluded[task] = ", ".join(f"{m}={v:.1f}" for m, v in zip(failing.model, failing.clean))
if excluded:
    print("\n*** BASELINE GATE FAILED, tasks EXCLUDED from the comparison ***")
    for task, info in excluded.items():
        print(f"  {task}: clean below {GATES[task]:.0f}% for: {info}")


# ---- architecture comparison ----
def arch_item_drops(arch, task, t):
    cols = []
    for name, tasks in scores.items():
        if arch_of(name) != arch or task not in tasks:
            continue
        d = item_drop(tasks[task], t)
        if d is not None:
            cols.append(d.rename(name))
    return pd.concat(cols, axis=1) if cols else None


out = []
for task in sorted(data.task.unique()):
    if task in excluded:
        continue
    for t in sorted(set(data.type) - {"clean"}) + ["ALL"]:
        da, db = arch_item_drops(A, task, t), arch_item_drops(B, task, t)
        if da is None or db is None:
            continue
        both = pd.concat([da.mean(axis=1).rename("a"), db.mean(axis=1).rename("b")],
                         axis=1).dropna()
        if both.empty:
            continue
        diff = (both["a"] - both["b"]).values
        idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
        boots = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sel = (seed_tbl.task == task) & (seed_tbl.perturbation == t)
        sa, sb = seed_tbl[sel & (seed_tbl.arch == A)], seed_tbl[sel & (seed_tbl.arch == B)]
        separated = bool((sa["drop"].min() > sb["drop"].max()) or
                         (sb["drop"].min() > sa["drop"].max()))
        out.append(dict(task=task, perturbation=t, n_items=len(diff),
                        mha_drop=sa["drop"].mean(), gqa_drop=sb["drop"].mean(),
                        diff_mha_minus_gqa=diff.mean(),
                        ci95=f"[{lo:.1f}, {hi:.1f}]",
                        ci_excludes_0=bool(lo > 0 or hi < 0),
                        perm_p=perm_p(sa["drop"], sb["drop"]),
                        mha_rel=sa["rel_drop"].mean(), gqa_rel=sb["rel_drop"].mean(),
                        perm_p_rel=perm_p(sa["rel_drop"], sb["rel_drop"]),
                        n_seeds_mha=len(sa), n_seeds_gqa=len(sb),
                        seeds_separated=separated))
res = pd.DataFrame(out)
res.to_csv("arch_comparison.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

print("\n=== Clean score by architecture (baseline capability), all tasks ===")
cl = seed_tbl[seed_tbl.perturbation == "ALL"]
crow = []
for task, g in cl.groupby("task"):
    ca, cb = g[g.arch == A]["clean"], g[g.arch == B]["clean"]
    crow.append(dict(task=task, mha_clean=ca.mean(), gqa_clean=cb.mean(),
                     diff=ca.mean() - cb.mean(), perm_p=perm_p(ca, cb)))
print(pd.DataFrame(crow).round(2).to_string(index=False))

print("\n=== MHA vs GQA (diff = MHA drop - GQA drop; hypothesis predicts diff < 0) ===")
print(res.round(2).to_string(index=False))

print("\n=== Does baseline capability track the drop? (pooled drop vs clean, across models) ===")
for task, g in cl.groupby("task"):
    if task in excluded:
        continue
    r, p = corr(g["clean"], g["drop"])
    print(f"  {task}: n = {len(g)} models, Pearson r = {r:.2f}, permutation p = {p:.2f}")

print("\nReading guide:")
print("  * Claim a difference only if ci_excludes_0 AND seeds_separated AND perm_p are all convincing.")
print("  * ci95 reflects test-set noise only. perm_p and seeds_separated reflect training-seed noise.")
print("  * With 3 vs 3 seeds the smallest possible perm_p is 0.10, so this design cannot reach 0.05.")
print("  * Many rows are shown. Treat only the pre-declared primary outcome (AG News, ALL) as confirmatory.")