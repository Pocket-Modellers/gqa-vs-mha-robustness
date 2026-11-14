import json, sys
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"])

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def item_score(metric, output, gold):
    if metric == "exact":
        return float(output.strip().lower().startswith(gold.lower()))
    return _rouge.score(gold, output.strip())["rougeL"].fmeasure

def evaluate(eval_path, preds_path, model_name, n_boot=1000):
    data = pd.DataFrame(load(eval_path))
    preds = pd.DataFrame(load(preds_path))
    df = data.merge(preds, on="id")
    df["score"] = [item_score(m, o, g) * 100
                   for m, o, g in zip(df.metric, df.output, df.gold)]
    rng = np.random.default_rng(0)
    out = []
    for task, tdf in df.groupby("task"):
        wide = tdf.pivot(index="item", columns="type", values="score")
        row = {"model": model_name, "task": task, "clean": wide["clean"].mean()}
        for t in [c for c in wide.columns if c != "clean"]:
            pair = wide[["clean", t]].dropna()
            diff = (pair["clean"] - pair[t]).values
            idx = rng.integers(0, len(diff), (n_boot, len(diff)))
            boots = diff[idx].mean(axis=1)
            row[t] = pair[t].mean()
            row[f"drop_{t}"] = diff.mean()
            row[f"ci_{t}"] = f"[{np.percentile(boots, 2.5):.1f}, {np.percentile(boots, 97.5):.1f}]"
        out.append(row)
    return pd.DataFrame(out)

if __name__ == "__main__":
    eval_path, preds_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(evaluate(eval_path, preds_path, name).round(1).to_string())