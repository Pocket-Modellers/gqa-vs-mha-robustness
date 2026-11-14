"""Plot the pretraining loss curves of the six runs from training_curves.csv.

    python plot_loss_curves.py          # writes loss_curves.png

training_curves.csv holds the training loss every 20 steps and the validation
loss every 100 steps (plus the final step) for each run, taken from the run logs.
"""
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = collections.defaultdict(lambda: {"t": [], "v": []})
for r in csv.DictReader(open("training_curves.csv")):
    tok = int(r["tokens"]) / 1e9
    if r["train_loss"]:
        d[r["run"]]["t"].append((tok, float(r["train_loss"])))
    if r["val_loss"]:
        d[r["run"]]["v"].append((tok, float(r["val_loss"])))

C = {"mha": "#2a78d6", "gqa": "#eb6834"}
INK, MUTED, GRID = "#2b2b28", "#77756c", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200)
for ax in axes:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Tokens seen (billions)")

# (a) training loss, all six runs
ax = axes[0]
for run, v in d.items():
    x, y = zip(*v["t"])
    ax.plot(x, y, color=C[run[:3]], lw=1.0, alpha=0.85)
ax.set_ylabel("Loss")
ax.set_title("(a) Training loss, 6 runs", loc="left", fontsize=9, color=INK)
ax.set_ylim(3.3, 9.2)
ax.set_xlim(0, 1.0)
ax.plot([], [], color=C["mha"], lw=2, label="MHA (12 KV heads)")
ax.plot([], [], color=C["gqa"], lw=2, label="GQA (3 KV heads)")
ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)

# (b) validation loss over the second half of training
ax = axes[1]
for run, v in d.items():
    x, y = zip(*[p for p in v["v"] if p[0] >= 0.45])
    ax.plot(x, y, color=C[run[:3]], lw=1.4, marker="o", ms=2.5)
    ax.plot([x[-1]], [y[-1]], marker="o", ms=4.5, color=C[run[:3]], mec="white", mew=1)
mha = sum(d[r]["v"][-1][1] for r in d if r.startswith("mha")) / 3
gqa = sum(d[r]["v"][-1][1] for r in d if r.startswith("gqa")) / 3
ax.text(1.005, mha, f"MHA mean {mha:.3f}", color=INK, fontsize=8, va="center")
ax.text(1.005, gqa + 0.004, f"GQA mean {gqa:.3f}", color=INK, fontsize=8, va="center")
ax.set_title("(b) Validation loss, 0.5–1B tokens", loc="left", fontsize=9, color=INK)
ax.set_xlim(0.45, 1.0)

fig.tight_layout()
fig.savefig("loss_curves.png", bbox_inches="tight", facecolor="white")
print("wrote loss_curves.png")
