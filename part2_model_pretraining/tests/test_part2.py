"""
Run from the repo folder with:   python -m pytest tests -q

These use a tiny model on CPU, so they take a few seconds and need no GPU or data.
If any of these fail after you change the code, something is broken.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint import load_model, save_checkpoint  # noqa: E402
from data import TokenShards  # noqa: E402
from model import ModelConfig, Transformer, ablate_heads, matched_ffn_hidden  # noqa: E402


def tiny(n_kv_heads, ffn_hidden=96, match=False):
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=8,
                      n_kv_heads=n_kv_heads, ffn_hidden=ffn_hidden, max_seq_len=32)
    if match and n_kv_heads != cfg.n_heads:
        cfg.ffn_hidden = matched_ffn_hidden(cfg, ffn_hidden)
    return cfg


def tokens(seed=0, B=2, T=16):
    return torch.randint(0, 100, (B, T), generator=torch.Generator().manual_seed(seed))


def test_param_counts_match_after_ffn_widening():
    mha = Transformer(tiny(8)).num_params()
    for kv in (4, 2, 1):
        gqa = Transformer(tiny(kv, match=True)).num_params()
        assert abs(gqa - mha) / mha < 0.005, f"kv={kv}: {gqa} vs {mha}"


def test_gqa_has_fewer_params_without_matching():
    assert Transformer(tiny(2)).num_params() < Transformer(tiny(8)).num_params()


def test_output_shapes_and_loss():
    for kv in (8, 2, 1):
        m = Transformer(tiny(kv))
        x, y = tokens(seed=0), tokens(seed=1)
        logits, loss = m(x, y)
        assert logits.shape == (2, 16, 100)
        assert loss.item() == pytest.approx(np.log(100), rel=0.1)  # untrained ~ uniform


def test_causal_no_peeking_at_future():
    m = Transformer(tiny(2)).eval()
    x = tokens()
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 100  # change only future tokens
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x2)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_gqa_equals_mha_with_shared_kv_weights():
    """
    A GQA model must behave exactly like an MHA model whose K/V heads are copies
    within each group. This checks the head-to-group mapping is right.
    """
    torch.manual_seed(0)
    gqa = Transformer(tiny(2)).eval()
    mha = Transformer(tiny(8)).eval()
    sd = gqa.state_dict()
    group, hd = 4, 8
    new_sd = {}
    for k, v in sd.items():
        if k.endswith("k_proj.weight") or k.endswith("v_proj.weight"):
            # (n_kv*hd, d) -> repeat each KV head `group` times -> (n_heads*hd, d)
            v = v.view(2, hd, -1).repeat_interleave(group, dim=0).reshape(8 * hd, -1)
        new_sd[k] = v
    mha.load_state_dict(new_sd)
    x = tokens()
    with torch.no_grad():
        assert torch.allclose(gqa(x)[0], mha(x)[0], atol=1e-5)


def test_kv_group_mapping():
    attn = Transformer(tiny(2)).layers[0].attn
    assert [attn.kv_group_of_head(h) for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_hook_point_ablation():
    m = Transformer(tiny(2)).eval()
    x = tokens()
    with torch.no_grad():
        base, _ = m(x)
        with ablate_heads(m, {}):
            same, _ = m(x)
        with ablate_heads(m, {0: [3]}):
            changed, _ = m(x)
        after, _ = m(x)  # hooks removed again
    assert torch.allclose(base, same)
    assert not torch.allclose(base, changed)
    assert torch.allclose(base, after)


def test_checkpoint_roundtrip(tmp_path):
    cfg = tiny(2, match=True)
    m = Transformer(cfg).eval()
    path = str(tmp_path / "step_000001.pt")
    save_checkpoint(path, m, cfg, step=1, tokens_seen=123)
    m2, info = load_model(path)
    x = tokens()
    with torch.no_grad():
        assert torch.equal(m(x)[0], m2(x)[0])
    assert info["tokens_seen"] == 123 and info["config"]["n_kv_heads"] == 2


def test_data_batches_deterministic(tmp_path):
    arr = np.arange(10_000, dtype=np.uint16) % 500
    p = tmp_path / "train_000.bin"
    arr.tofile(p)
    ds = TokenShards([str(p)])
    x1, y1 = ds.get_batch(data_seed=0, step=5, micro_step=0, batch_size=4, seq_len=32)
    x2, _ = ds.get_batch(data_seed=0, step=5, micro_step=0, batch_size=4, seq_len=32)
    x3, _ = ds.get_batch(data_seed=1, step=5, micro_step=0, batch_size=4, seq_len=32)
    assert torch.equal(x1, x2)            # same seed+step -> same data
    assert not torch.equal(x1, x3)        # different seed -> different data
    assert torch.equal(x1[:, 1:], y1[:, :-1])  # targets are inputs shifted by one
    v1 = ds.fixed_batches(2, 4, 32)
    v2 = ds.fixed_batches(2, 4, 32)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(v1, v2))


def test_sft_loss_only_on_response():
    from sft import tokenize_rows

    class FakeTok:  # one token per character, enough to test the masking
        def encode(self, text, add_special_tokens=False):
            class E:
                pass
            e = E()
            e.ids = [ord(ch) % 100 for ch in text]
            return e

    row = {"instruction": "Say hi", "context": "", "response": "hi"}
    (x, y), = tokenize_rows([row], FakeTok(), max_len=512)[0]
    from chat_format import format_prompt, format_response
    n_prompt = len(format_prompt(row))
    n_resp = len(format_response(row))
    assert len(x) == len(y) == n_prompt + n_resp - 1
    # only the response characters (incl. <|im_end|>) carry a label
    assert sum(l != -100 for l in y) == n_resp
    assert all(l == -100 for l in y[: n_prompt - 1])


def test_tiny_model_can_learn():
    torch.manual_seed(0)
    m = Transformer(tiny(2))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = tokens(B=4, T=32)
    first = None
    for _ in range(60):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
