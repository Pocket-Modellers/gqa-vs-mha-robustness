import json, difflib
orig = {}
for l in open("dolly_instructions.jsonl", encoding="utf-8"):
    r = json.loads(l); orig[r["id"]] = r["instruction"].strip()
para = [json.loads(l) for l in open("dolly_paraphrases.jsonl", encoding="utf-8") if l.strip()]
same = close = 0
for r in para:
    ratio = difflib.SequenceMatcher(None, orig[r["id"]].lower(), r["paraphrase"].strip().lower()).ratio()
    if ratio == 1.0: same += 1
    elif ratio > 0.9: close += 1
print(len(para), "paraphrases |", same, "identical |", close, "nearly identical")