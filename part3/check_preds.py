import json, sys, collections

eval_path, preds_path = sys.argv[1], sys.argv[2]
ev = {r["id"]: r for r in (json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip())}
pr = [json.loads(l) for l in open(preds_path, encoding="utf-8") if l.strip()]
ids = [p["id"] for p in pr]
outs = {p["id"]: p["output"] for p in pr}

problems = []
if len(pr) != len(ev):
    problems.append(f"row count {len(pr)} (expected {len(ev)})")
if len(set(ids)) != len(ids):
    problems.append("duplicate ids")
if set(ev) - set(ids):
    problems.append(f"{len(set(ev) - set(ids))} ids missing")
if set(ids) - set(ev):
    problems.append(f"{len(set(ids) - set(ev))} unknown ids")
empty = sum(1 for p in pr if not p["output"].strip())
if empty:
    problems.append(f"{empty} empty outputs")
for tok in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|pad|>"):
    n = sum(tok in p["output"] for p in pr)
    if n:
        problems.append(f"{n} outputs contain {tok}")

print("File:", preds_path)
print("PROBLEMS:" if problems else "Format OK: ids, counts, no special tokens, no empty outputs.")
for p in problems:
    print("  -", p)

labels = {t: sorted({r["gold"] for r in ev.values() if r["task"] == t})
          for t in ("sst2", "agnews")}
print("\nClosed tasks (clean exact match, valid-label rate, share of most common answer):")
for task, labs in labels.items():
    rows = [r for r in ev.values() if r["task"] == task and r["type"] == "clean" and r["id"] in outs]
    if not rows:
        continue
    acc = sum(outs[r["id"]].strip().lower().startswith(r["gold"]) for r in rows) / len(rows) * 100
    valid = sum(any(outs[r["id"]].strip().lower().startswith(l) for l in labs) for r in rows) / len(rows) * 100
    top = collections.Counter(outs[r["id"]].strip().lower()[:12] for r in rows).most_common(1)[0][1] / len(rows) * 100
    print(f"  {task:7s} clean acc {acc:5.1f}% | valid labels {valid:5.1f}% | top answer {top:5.1f}%")
