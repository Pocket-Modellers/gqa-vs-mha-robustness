import json, hashlib, collections
rows = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8") if l.strip()]
print("rows:", len(rows), "(expect 3000)")
print("start with ChatML header:", sum(r["prompt"].startswith("<|im_start|>user\n") for r in rows), "(expect 3000)")
print("end with assistant header:", sum(r["prompt"].endswith("<|im_start|>assistant\n") for r in rows), "(expect 3000)")
c = collections.Counter((r["task"], r["type"]) for r in rows)
print("dolly paraphrase rows:", c[("dolly", "paraphrase")], "(expect 200)")
print("version id:", hashlib.sha256(open("eval_set.jsonl", "rb").read()).hexdigest()[:12])