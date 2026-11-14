import json, os, random
from datasets import load_dataset

random.seed(0)
N = 200
CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
HELDOUT_FILE = "heldout.jsonl"
PARAPHRASE_FILE = "dolly_paraphrases.jsonl"
NEUTRAL = [
    "The meeting room is on the second floor.",
    "Please remember to bring an umbrella.",
    "The library is closed on Sundays.",
]
rows = []

def wrap(p):
    return CHAT_TEMPLATE.format(prompt=p)

# ---------------- Track A: closed-answer tasks ----------------
def add_closed(task, ds, text_key, labels, clean, paraphrases, distractors, reformat2):
    for i, ex in enumerate(ds):
        x, gold = ex[text_key].strip(), labels[ex["label"]]
        pi = random.randrange(len(paraphrases))
        di = random.randrange(len(distractors))
        variants = {
            "clean": clean.format(x=x),
            "paraphrase": paraphrases[pi].format(x=x),
            "distractor": distractors[di] + "\n" + clean.format(x=x),
            "reformat": "Q: " + clean.format(x=x).replace("\nAnswer:", "") + "\nA:",
            "reformat2": reformat2.format(x=x),
        }
        tmpl = {"paraphrase": pi, "distractor": di}
        for vtype, prompt in variants.items():
            rows.append({"id": f"{task}-{i}-{vtype}", "item": f"{task}-{i}", "task": task,
                         "type": vtype, "template": tmpl.get(vtype, 0),
                         "metric": "exact", "max_new_tokens": 5,
                         "prompt": wrap(prompt), "gold": gold})

sst = load_dataset("nyu-mll/glue", "sst2", split="validation").shuffle(seed=0).select(range(N))
add_closed(
    "sst2", sst, "sentence", ["negative", "positive"],
    "Review: {x}\nIs this review positive or negative?\nAnswer:",
    ["What is the sentiment of this review?\n{x}\nReply with positive or negative.\nAnswer:",
     "Decide if the following movie review is positive or negative.\n{x}\nAnswer:",
     "{x}\nWould you say the review above is positive or negative?\nAnswer:"],
    ["The train leaves at nine in the morning.",
     "Water boils at 100 degrees Celsius at sea level.",
     "The library is closed on Sundays."],
    "Instruction: Is this review positive or negative? | Review: {x} | Answer:",
)

ag = (load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0)
      .filter(lambda e: len(e["text"]) <= 500).select(range(N)))
add_closed(
    "agnews", ag, "text", ["world", "sports", "business", "technology"],
    "Article: {x}\nWhich topic is this article about: world, sports, business, or technology?\nAnswer:",
    ["What is the topic of the following news article? Choose from world, sports, business, or technology.\n{x}\nAnswer:",
     "{x}\nWhich category fits the article above: world, sports, business, or technology?\nAnswer:",
     "Classify this news story as world, sports, business, or technology.\n{x}\nAnswer:"],
    NEUTRAL,
    "Instruction: Which topic is this article about: world, sports, business, or technology? | Article: {x} | Answer:",
)

# ---------------- Track B: Dolly held-out slice ----------------
def load_heldout():
    if os.path.exists(HELDOUT_FILE):
        data = [json.loads(l) for l in open(HELDOUT_FILE, encoding="utf-8") if l.strip()]
        print("Using Part 1 held-out file:", len(data), "rows")
    else:
        print("WARNING: PROVISIONAL Dolly slice. Replace with Part 1's held-out file!")
        d = load_dataset("databricks/databricks-dolly-15k", split="train")
        data = [d[i] for i in range(len(d) - 3000, len(d))]
    return data

KEEP = {"open_qa", "general_qa", "classification"}
pool = [e for e in load_heldout()
        if e.get("context", "") == "" and e.get("category") in KEEP
        and len(e["response"].split()) <= 50
        and len(e["instruction"].strip()) <= 300]
rng = random.Random(42)          # separate generator so ids stay stable
rng.shuffle(pool)
pool = pool[:N]
print("Dolly items used:", len(pool))

with open("dolly_instructions.jsonl", "w", encoding="utf-8") as f:
    for i, e in enumerate(pool):
        f.write(json.dumps({"id": f"dolly-{i}", "instruction": e["instruction"].strip()}) + "\n")

para = {}
if os.path.exists(PARAPHRASE_FILE):
    for l in open(PARAPHRASE_FILE, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            para[r["id"]] = r["paraphrase"].strip()
    print("Paraphrases loaded:", len(para), "of", len(pool))
else:
    print("No paraphrase file yet (Step 7). Dolly paraphrase rows skipped.")

for i, e in enumerate(pool):
    ins, gold = e["instruction"].strip(), e["response"].strip()
    di = rng.randrange(len(NEUTRAL))
    variants = {
        "clean": ins,
        "distractor": NEUTRAL[di] + "\n" + ins,
        "reformat": "Q: " + ins + "\nA:",
        "reformat2": "Instruction: " + ins + " | Answer:",
    }
    if f"dolly-{i}" in para:
        variants["paraphrase"] = para[f"dolly-{i}"]
    for vtype, prompt in variants.items():
        rows.append({"id": f"dolly-{i}-{vtype}", "item": f"dolly-{i}", "task": "dolly",
                     "type": vtype, "template": di if vtype == "distractor" else 0,
                     "metric": "rouge", "max_new_tokens": 64,
                     "prompt": wrap(prompt), "gold": gold})

with open("eval_set.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "rows")