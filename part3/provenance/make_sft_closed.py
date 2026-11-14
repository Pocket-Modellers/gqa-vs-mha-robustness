import json, random
from datasets import load_dataset
random.seed(123)
N = 1000

SST_T = [
  "Review: {x}\nIs this review positive or negative?\nAnswer:",
  "Classify the sentiment of this movie review as positive or negative.\n{x}",
  "Movie review: {x}\nSentiment (positive or negative):",
  "Is the following review positive or negative?\n{x}",
  "{x}\nThe sentiment of this review is",
]
AG_T = [
  "Article: {x}\nWhich topic is this article about: world, sports, business, or technology?\nAnswer:",
  "Read the article and give its topic (world, sports, business, or technology).\n{x}",
  "News: {x}\nTopic (world, sports, business, or technology):",
  "Which section of a newspaper would this article appear in: world, sports, business, or technology?\n{x}",
  "{x}\nTopic:",
]

rows = []
sst = (load_dataset("nyu-mll/glue", "sst2", split="train").shuffle(seed=1)
       .filter(lambda e: len(e["sentence"].split()) >= 5).select(range(N)))
for ex in sst:
    rows.append({"instruction": random.choice(SST_T).format(x=ex["sentence"].strip()),
                 "context": "", "response": ["negative", "positive"][ex["label"]],
                 "category": "closed_clf"})

ag = (load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=1)
      .filter(lambda e: len(e["text"]) <= 500).select(range(N)))
for ex in ag:
    rows.append({"instruction": random.choice(AG_T).format(x=ex["text"].strip()),
                 "context": "",
                 "response": ["world", "sports", "business", "technology"][ex["label"]],
                 "category": "closed_clf"})

random.shuffle(rows)
with open("sft_closed.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "rows")