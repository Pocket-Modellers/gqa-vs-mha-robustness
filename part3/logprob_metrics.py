"""M4: log-probability shift, computed from the logp_<arch>_<seed>.jsonl files.

Two file formats are accepted:
  Part 2's Kaggle cell:   {"id", "logp": {label: first-token log-prob}, "logp_full": {label: whole-label log-prob}}
                          (closed tasks only, 2,000 rows)
  logprob_extract.py:     {"id", "task", "scores": {...}} for closed tasks and {"id", "nll": ...} for Dolly
M4 uses the whole-label log-probabilities. The first-token version is reported as S_M4_first_token.
Dolly has no M4 when the file contains closed tasks only.


Usage:  python logprob_metrics.py [eval_set.jsonl]
Reads logp_<arch>_<seed>.jsonl (one per model) and, when present, preds_<arch>_<seed>.jsonl.
Writes logprob_metrics_seed.csv.

M4, higher = more brittle, averaged over the four perturbations:
  AG News and SST-2: mean over items of (clean margin - perturbed margin), in nats, where
                     margin = log-prob of the correct label - log-prob of the best wrong label.
  Dolly:             mean over items of (perturbed NLL - clean NLL), in nats per token, where
                     NLL is the mean per-token negative log-likelihood of the reference answer.
M4b (closed tasks): mean drop in the probability of the correct label, in percentage points,
                     where the probability is a softmax over the task's label set.
Agreement check: on clean prompts the label with the highest log-probability should usually
equal the generated label. A low value points to a mistake in the extraction (template,
tokenizer, padding), not to a property of the model.
"""
import glob, os, sys
import numpy as np
import pandas as pd
from score import load

EVAL = sys.argv[1] if len(sys.argv) > 1 else "eval_set.jsonl"
FORMATS = ["clean", "paraphrase", "distractor", "reformat", "reformat2"]
CLOSED = ["agnews", "sst2"]

data = pd.DataFrame(load(EVAL))
meta = data.set_index("id")[["item", "task", "type", "gold"]]


def label_of(output, labels):
    o = str(output).strip().lower()
    for lab in labels:
        if o.startswith(lab):
            return lab
    return "other"


rows = []
for path in sorted(glob.glob("logp_*.jsonl")):
    name = os.path.basename(path)[len("logp_"):-len(".jsonl")]
    arch = name.split("_")[0].lower()
    recs = {r["id"]: r for r in load(path)}
    preds = None
    ppath = f"preds_{name}.jsonl"
    if os.path.exists(ppath):
        preds = {r["id"]: r["output"] for r in load(ppath)}
    used = {}
    for task, tm in meta.groupby("task"):
        row = dict(model=name, arch=arch, task=task)
        if task in CLOSED:
            margin, margin1, pcor, agree = {}, {}, {}, []
            labs = sorted(set(tm["gold"]))
            for i in tm.index:
                r = recs.get(i)
                if r is None:
                    continue
                sc = r.get("scores") or r.get("logp_full")
                if sc is None:
                    continue
                gold = tm.at[i, "gold"]
                if gold not in sc or set(sc) != set(labs):
                    sys.exit(f"{name}: the label strings in {path} {sorted(sc)} do not match the evaluation "
                             f"labels {labs}. Fix the LABELS dictionary in the Kaggle cell and rerun.")
                s_ = np.array([sc[k] for k in labs])
                margin[i] = sc[gold] - max(v for k, v in sc.items() if k != gold)
                pcor[i] = float(np.exp(sc[gold] - np.logaddexp.reduce(s_)))
                f1 = r.get("logp")
                if f1 is not None:
                    margin1[i] = f1[gold] - max(v for k, v in f1.items() if k != gold)
                if preds is not None and tm.at[i, "type"] == "clean" and i in preds:
                    agree.append(max(sc, key=sc.get) == label_of(preds[i], labs))
            used[task] = len(margin)
            if not margin:
                continue
            df = tm.assign(margin=pd.Series(margin), p=pd.Series(pcor), m1=pd.Series(margin1)).dropna(subset=["margin"])
            wm = df.pivot(index="item", columns="type", values="margin")
            wp = df.pivot(index="item", columns="type", values="p")
            perts = [f for f in FORMATS if f != "clean" and f in wm.columns]
            row["M4_logprob_shift"] = wm[perts].rsub(wm["clean"], axis=0).mean(axis=1).mean()
            row["M4b_pcorrect_drop"] = wp[perts].rsub(wp["clean"], axis=0).mean(axis=1).mean() * 100
            if df["m1"].notna().any():
                w1 = df.pivot(index="item", columns="type", values="m1")
                row["S_M4_first_token"] = w1[perts].rsub(w1["clean"], axis=0).mean(axis=1).mean()
            row["margin_clean"] = wm["clean"].mean()
            row["agree_clean"] = float(np.mean(agree)) * 100 if agree else np.nan
        else:
            nll = {i: recs[i]["nll"] for i in tm.index if i in recs and "nll" in recs[i]}
            used[task] = len(nll)
            if not nll:
                continue
            df = tm.assign(nll=pd.Series(nll)).dropna(subset=["nll"])
            w = df.pivot(index="item", columns="type", values="nll")
            perts = [f for f in FORMATS if f != "clean" and f in w.columns]
            row["M4_logprob_shift"] = w[perts].sub(w["clean"], axis=0).mean(axis=1).mean()
            row["nll_clean"] = w["clean"].mean()
        rows.append(row)
    expected = meta.groupby("task").size().to_dict()
    cov = ", ".join(f"{t}: {n}/{expected[t]}" for t, n in used.items())
    print(f"{name}: usable rows per task -> {cov}")
    for t in CLOSED:
        if used.get(t, 0) != expected.get(t, 0):
            print(f"  WARNING: {name} is missing {expected.get(t, 0) - used.get(t, 0)} {t} rows")

out = pd.DataFrame(rows)
if out.empty:
    sys.exit("No logp_*.jsonl files found.")
out.to_csv("logprob_metrics_seed.csv", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\n=== M4 per model (closed tasks: margin shift in nats; Dolly: NLL shift in nats/token) ===")
print(out.round(3).to_string(index=False))
low = out[(out.task == "agnews") & (out.agree_clean < 90)]
if not low.empty:
    print("\nWARNING: on AG News clean prompts, the highest-log-prob label disagrees with the generated "
          "label for more than 10% of items in:", ", ".join(low.model))
    print("Check the extraction (template, tokenizer, padding) before using these numbers.")
else:
    print("\nAgreement check passed (or no preds files found).")
