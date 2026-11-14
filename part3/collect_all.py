import glob, os
import pandas as pd
from score import evaluate

tabs = []
for p in sorted(glob.glob("preds_*.jsonl")):
    name = os.path.basename(p)[len("preds_"):-len(".jsonl")]
    tabs.append(evaluate("eval_set.jsonl", p, name))
t = pd.concat(tabs)
t.to_csv("brittleness_table.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(t.round(1).to_string())