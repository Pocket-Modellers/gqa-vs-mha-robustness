"""
Print parameter counts for MHA, GQA and MQA with the current base.yaml settings.

    python count_params.py

This is the "verify the parameter counts match" step of Part 2.
"""

from model import Transformer
from pretrain import build_model_config, load_config


def main():
    base = load_config("configs/base.yaml")
    n_heads = int(base["n_heads"])
    rows = []
    arms = [("MHA", n_heads),
            ("GQA", int(load_config("configs/gqa_s0.yaml")["n_kv_heads"])),
            ("MQA", int(load_config("configs/mqa_s0.yaml")["n_kv_heads"]))]
    for name, kv in arms:
        c = dict(base, n_kv_heads=kv)
        mcfg = build_model_config(c)
        model = Transformer(mcfg)
        rows.append((name, kv, mcfg.ffn_hidden, model.num_params(), model.num_params(False)))

    ref = rows[0][3]
    print(f"{'arch':5s} {'kv_heads':>8s} {'ffn_hidden':>10s} {'total params':>14s} {'non-emb':>12s} {'vs MHA':>8s}")
    for name, kv, ffn, total, non_emb in rows:
        diff = 100 * (total - ref) / ref
        print(f"{name:5s} {kv:>8d} {ffn:>10d} {total:>14,} {non_emb:>12,} {diff:>+7.3f}%")
    print("\nmatch_params is", base.get("match_params", True),
          "(true = GQA/MQA get a wider FFN to match MHA)")


if __name__ == "__main__":
    main()
