# ===== Log-probability of each label at the answer position (no generation) =====
# Kaggle inputs: part2-code, fineweb-shards (tokenizer), sft-data (eval_set.jsonl),
# and the 6 sft_runner_* notebooks (each has ckpts/<run>_sft/final.pt + preds_<run>.jsonl).
# Writes /kaggle/working/logp_<run>.jsonl, one per model:
#   {"id": ..., "logp": {label: first-token logp, ...}, "logp_full": {label: whole-label logp, ...}}

import glob, json, os, sys
import torch
import torch.nn.functional as F

ROOT = os.environ.get("LOGP_ROOT", "/kaggle/input")   # where the inputs are
OUT = os.environ.get("LOGP_OUT", "/kaggle/working")   # where results go

# Exact label strings from the closed-answer SFT responses, keyed by evaluation id prefix.
LABELS = {
    "sst2": ["positive", "negative"],
    "agnews": ["world", "sports", "business", "technology"],   
}

def find(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, "**", pattern), recursive=True))
    assert hits, f"not found under {ROOT}: {pattern}"
    return hits

sys.path.insert(0, os.path.dirname(find("checkpoint.py")[0]))
from checkpoint import load_model
from tokenizers import Tokenizer

tok = Tokenizer.from_file(find("tokenizer_32000.json")[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---- 1. which rows, and are the labels' first tokens distinct?
rows = [json.loads(l) for l in open(find("eval_set.jsonl")[0]) if l.strip()]
task = lambda r: r["id"].split("-")[0]
print("id prefixes in eval_set:", sorted({task(r) for r in rows}))
rows = [r for r in rows if task(r) in LABELS]
print(f"{len(rows)} rows to score (expect 2000):",
      {t: sum(task(r) == t for r in rows) for t in LABELS})
label_ids = {}
for t, labs in LABELS.items():
    label_ids[t] = {lab: tok.encode(lab).ids for lab in labs}
    firsts = {lab: ids[0] for lab, ids in label_ids[t].items()}
    print(f"{t}: label -> token ids {label_ids[t]}")
    assert len(set(firsts.values())) == len(labs), f"{t}: first tokens NOT distinct {firsts}"
print("OK: first tokens are distinct for every task\n")
prompt_ids = {r["id"]: tok.encode(r["prompt"]).ids for r in rows}

# ---- 2. score every model
@torch.no_grad()
def score(model, max_len, batch_rows=16):
    results = []
    for i in range(0, len(rows), batch_rows):
        seqs, meta = [], []
        for r in rows[i:i + batch_rows]:
            for lab, lids in label_ids[task(r)].items():
                p = prompt_ids[r["id"]][-(max_len - len(lids)):]   # keep the end if too long
                seqs.append(p + lids)
                meta.append((r["id"], lab, len(p), lids))
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), 1, dtype=torch.long)       # pad id 1 (right-padded)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits, _ = model(x.to(device))
        lp = F.log_softmax(logits.float(), dim=-1)
        by_id = {}
        for j, (rid, lab, plen, lids) in enumerate(meta):
            pos = torch.arange(plen - 1, plen - 1 + len(lids), device=lp.device)
            tok_lp = lp[j, pos, torch.tensor(lids, device=lp.device)]
            d = by_id.setdefault(rid, {"id": rid, "logp": {}, "logp_full": {}})
            d["logp"][lab] = round(tok_lp[0].item(), 5)
            d["logp_full"][lab] = round(tok_lp.sum().item(), 5)
        results.extend(by_id.values())
    return results

ckpts = find("ckpts/*_sft/final.pt")
print("SFT models found:", [os.path.basename(os.path.dirname(c))[:-4] for c in ckpts])
for ck in ckpts:
    run = os.path.basename(os.path.dirname(ck))[:-4]          # "gqa_s0_sft" -> "gqa_s0"
    model, _ = load_model(ck, device)
    res = score(model, model.cfg.max_seq_len)
    path = os.path.join(OUT, f"logp_{run}.jsonl")
    with open(path, "w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")
    # sanity check: does the top label agree with the model's greedy answer?
    preds = glob.glob(os.path.join(ROOT, "**", f"preds_{run}.jsonl"), recursive=True)
    msg = ""
    if preds:
        gen = {json.loads(l)["id"]: json.loads(l)["output"].strip() for l in open(preds[0]) if l.strip()}
        agree = [max(d["logp_full"], key=d["logp_full"].get) == gen.get(d["id"]) for d in res]
        msg = f" | top label == greedy answer on {100 * sum(agree) / len(agree):.1f}% of rows"
    print(f"{run}: wrote {len(res)} rows -> {path}{msg}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
print("done")
