"""
Llama-style decoder-only transformer with a switchable number of key/value heads.

    n_kv_heads == n_heads      -> MHA (multi-head attention)
    1 < n_kv_heads < n_heads   -> GQA (grouped-query attention)
    n_kv_heads == 1            -> MQA (multi-query attention)

Everything else (RMSNorm, RoPE, SwiGLU, tied embeddings, init) is identical
across architectures, so attention is the ONLY thing that changes.

Read this file top to bottom: each class is one building block, and the
`Transformer` class at the bottom stacks them together.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Config: every number that defines the model lives here.
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 32000     # from Santosh's tokenizer_32000.json
    d_model: int = 512          # width of the residual stream
    n_layers: int = 8           # number of transformer blocks
    n_heads: int = 8            # number of QUERY heads
    n_kv_heads: int = 8         # number of KEY/VALUE heads (8=MHA, 2=GQA, 1=MQA)
    ffn_hidden: int = 1376      # hidden size of the SwiGLU feed-forward layer
    max_seq_len: int = 1024     # context length
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide by n_kv_heads"

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def group_size(self):
        """How many query heads share one key/value head."""
        return self.n_heads // self.n_kv_heads

    def to_dict(self):
        return asdict(self)


def matched_ffn_hidden(cfg: ModelConfig, reference_ffn_hidden: int) -> int:
    """
    GQA has smaller K and V projections than MHA, so it has fewer parameters.
    This returns the SwiGLU hidden size that gives the GQA model (almost exactly)
    the same parameter count as an MHA model with `reference_ffn_hidden`.

    Per layer, K+V projections in MHA:  2 * d * d
                           in GQA:      2 * d * (n_kv_heads * head_dim)
    SwiGLU has 3 matrices of size d x ffn_hidden, so each extra hidden unit
    adds 3 * d parameters.
    """
    d = cfg.d_model
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    params_saved = 2 * d * d - 2 * d * kv_dim
    return reference_ffn_hidden + round(params_saved / (3 * d))


# --------------------------------------------------------------------------
# 2. RMSNorm: a simpler LayerNorm (no mean subtraction, no bias).
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in float32 for stability, then cast back
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_f.type_as(x) * self.weight


# --------------------------------------------------------------------------
# 3. RoPE: rotary position embeddings, applied to queries and keys.
# --------------------------------------------------------------------------
def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    """Returns cos and sin tables of shape (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """
    x: (batch, n_heads, seq_len, head_dim)
    Rotates each pair of channels (x1, x2) by an angle that depends on position.
    """
    T = x.shape[2]
    cos = cos[:T].to(x.dtype)[None, None]   # (1, 1, T, head_dim/2)
    sin = sin[:T].to(x.dtype)[None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]       # even and odd channels
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# --------------------------------------------------------------------------
# 4. Attention: THE ONLY PART THAT DIFFERS BETWEEN MHA AND GQA.
# --------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group_size = cfg.group_size

        # Queries always get n_heads heads.
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        # Keys and values only get n_kv_heads heads. This is the GQA change:
        # with fewer KV heads these matrices are smaller.
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # HOOK POINT FOR PART 4 (circuits). Does nothing by default.
        # Its input/output has shape (batch, n_heads, seq_len, head_dim): the output
        # of every query head, before heads are mixed by o_proj. Part 4 registers a
        # forward hook here to read heads or replace them (ablation).
        self.head_out = nn.Identity()

    def kv_group_of_head(self, h):
        """Which KV head query head `h` uses. Heads 0..g-1 share KV head 0, etc."""
        return h // self.group_size

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # shapes now: q (B, n_heads, T, hd), k and v (B, n_kv_heads, T, hd)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Share each KV head across its group of query heads by repeating it.
        # e.g. 8 query heads, 2 KV heads: KV head 0 is used by query heads 0-3,
        # KV head 1 by query heads 4-7. For MHA, group_size == 1 and this is a no-op.
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        # Causal attention: each position only looks at itself and earlier positions.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_heads, T, hd)

        y = self.head_out(y)  # <- Part 4 hooks here

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


# --------------------------------------------------------------------------
# 5. SwiGLU feed-forward layer.
# --------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------
# 6. One transformer block: norm -> attention -> add, norm -> FFN -> add.
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------
# 7. The full model.
# --------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie input and output embeddings (saves vocab_size * d_model params).
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False: these are recomputed, not saved in checkpoints
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the projections that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (batch, seq_len) token ids
        targets: (batch, seq_len) next-token ids, or None. Positions set to -100
                 are ignored in the loss (this is how SFT masks out prompts).
        Returns (logits, loss). loss is None if targets is None.
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len"
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self, include_embeddings=True):
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if not include_embeddings:
            n -= self.tok_emb.weight.numel()
        return n


# --------------------------------------------------------------------------
# 8. Small helper for Part 4: zero-ablate chosen heads.
# --------------------------------------------------------------------------
@contextmanager
def ablate_heads(model: Transformer, heads_by_layer: dict):
    """
    Temporarily zero the output of chosen query heads.

        with ablate_heads(model, {3: [0, 5], 7: [2]}):
            logits, _ = model(x)

    To ablate a whole KV group in GQA, pass all query heads in that group, e.g.
    group g covers heads g*group_size ... (g+1)*group_size - 1.
    Part 4 will probably want mean-ablation instead of zeros; this is the template.
    """
    handles = []
    for layer_idx, heads in heads_by_layer.items():
        heads = list(heads)

        def hook(module, inputs, output, heads=heads):
            output = output.clone()
            output[:, heads] = 0.0
            return output

        handles.append(model.layers[layer_idx].attn.head_out.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
