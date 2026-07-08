"""Pure-MLX Qwen3-VL-4B text encoder for Krea-2 conditioning.

Text-only ⇒ mrope collapses to standard rope (all 3 position dims equal), so this
is a standard Qwen3 decoder (RMSNorm, GQA, per-head QK-norm, rope θ=5e6, head_dim
128 decoupled from hidden 2560). Returns 12 selected per-layer hidden states stacked
(B, seq, 12, 2560) + mask, matching krea-2-official/encoder.py. Tokenization uses the
HF tokenizer (as mflux does); the model forward is pure MLX.
"""

from __future__ import annotations

import glob

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_START_IDX = 34
SUFFIX_START_IDX = 5


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        t = x.astype(mx.float32)
        t = t * mx.rsqrt(mx.mean(t * t, axis=-1, keepdims=True) + self.eps)
        return (t * self.weight.astype(mx.float32)).astype(dt)


def _rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: (b, h, L, hd); cos/sin: (L, hd)
    c, s = cos[None, None], sin[None, None]
    return x * c + _rotate_half(x) * s


class Qwen3Attention(nn.Module):
    def __init__(self, hidden=2560, nheads=32, nkv=8, head_dim=128, eps=1e-6):
        super().__init__()
        self.nheads, self.nkv, self.hd = nheads, nkv, head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden, nheads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, nkv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, nkv * head_dim, bias=False)
        self.o_proj = nn.Linear(nheads * head_dim, hidden, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps)

    def __call__(self, x, cos, sin, mask):
        b, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, L, self.nheads, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, L, self.nkv, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, L, self.nkv, self.hd).transpose(0, 2, 1, 3)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        rep = self.nheads // self.nkv
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(b, L, self.nheads * self.hd)
        return self.o_proj(o)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden=2560, inter=9728):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Layer(nn.Module):
    def __init__(self, hidden=2560, nheads=32, nkv=8, head_dim=128, inter=9728, eps=1e-6):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden, eps)
        self.self_attn = Qwen3Attention(hidden, nheads, nkv, head_dim, eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden, eps)
        self.mlp = Qwen3MLP(hidden, inter)

    def __call__(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3TextModel(nn.Module):
    def __init__(self, vocab=151936, hidden=2560, layers=36, nheads=32, nkv=8,
                 head_dim=128, inter=9728, eps=1e-6, theta=5_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = [Qwen3Layer(hidden, nheads, nkv, head_dim, inter, eps) for _ in range(layers)]
        self.norm = Qwen3RMSNorm(hidden, eps)

    def _rope(self, L):
        inv = 1.0 / (self.theta ** (mx.arange(0, self.head_dim, 2).astype(mx.float32) / self.head_dim))
        pos = mx.arange(L).astype(mx.float32)
        freqs = pos[:, None] * inv[None, :]  # (L, hd/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (L, hd)
        return mx.cos(emb), mx.sin(emb)

    def __call__(self, input_ids, attn_valid):
        # input_ids: (b,L) int; attn_valid: (b,L) {0,1}
        b, L = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self._rope(L)
        cos, sin = cos.astype(h.dtype), sin.astype(h.dtype)

        # causal + padding additive mask (b,1,L,L)
        idx = mx.arange(L)
        causal = (idx[None, :] > idx[:, None]).astype(mx.float32) * -1e9  # (L,L)
        pad = (1.0 - attn_valid.astype(mx.float32)) * -1e9  # (b,L)
        mask = (causal[None, None] + pad[:, None, None, :]).astype(h.dtype)  # (b,1,L,L)

        all_hs = []
        for layer in self.layers:
            all_hs.append(h)          # HF output_hidden_states: append BEFORE each layer
            h = layer(h, cos, sin, mask)
        h = self.norm(h)
        all_hs.append(h)              # final (index == num_layers)
        return all_hs                 # all_hs[i] == HF hidden_states[i]


def load_text_encoder(repo: str, dtype=mx.float32) -> Qwen3TextModel:
    model = Qwen3TextModel()
    shards = sorted(glob.glob(f"{repo}/text_encoder/*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No text_encoder/*.safetensors found under {repo} "
                                "(incomplete download or wrong base dir).")
    weights = {}
    for shard in shards:
        for k, v in mx.load(shard).items():
            if k.startswith("language_model."):
                weights[k[len("language_model."):]] = v.astype(dtype)
    # strict: every parameter must be provided exactly once (catches missing shards / schema drift
    # that would otherwise leave random-init weights and silently produce garbage)
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = sorted(expected - set(weights)), sorted(set(weights) - expected)
    if missing or extra:
        raise RuntimeError(
            f"Text-encoder weight mismatch (missing={len(missing)}, extra={len(extra)}); "
            f"missing_head={missing[:4]} extra_head={extra[:4]}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, len(weights)


class Qwen3VLConditioner:
    """Pure-MLX conditioner. Tokenization via HF tokenizer; forward via MLX."""

    def __init__(self, repo: str, max_length: int = 512, dtype=mx.float32):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(f"{repo}/tokenizer")
        # guard the hardcoded prefix/suffix token counts (used to slice hidden states) against
        # tokenizer drift — a changed template/version would silently misalign the conditioning
        np_, ns_ = len(self.tokenizer(PREFIX)["input_ids"]), len(self.tokenizer(SUFFIX)["input_ids"])
        if np_ != PREFIX_START_IDX or ns_ != SUFFIX_START_IDX:
            raise RuntimeError(f"tokenizer drift: prefix={np_} (expected {PREFIX_START_IDX}), "
                               f"suffix={ns_} (expected {SUFFIX_START_IDX})")
        self.max_length = max_length
        self.dtype = dtype
        self.model, self.nloaded = load_text_encoder(repo, dtype)

    def __call__(self, prompts: list[str]) -> tuple[mx.array, mx.array]:
        prefix_idx = PREFIX_START_IDX
        text = [PREFIX + p for p in prompts]
        suffix = [SUFFIX] * len(text)
        suf = self.tokenizer(text=suffix, return_tensors="np")
        inp = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length + prefix_idx - SUFFIX_START_IDX, return_tensors="np",
        )
        input_ids = np.concatenate([inp["input_ids"], suf["input_ids"]], axis=1)
        mask = np.concatenate([inp["attention_mask"], suf["attention_mask"]], axis=1)

        ids_mx = mx.array(input_ids.astype(np.int32))
        valid_mx = mx.array(mask.astype(np.float32))
        all_hs = self.model(ids_mx, valid_mx)
        stacked = mx.stack([all_hs[i] for i in SELECT_LAYERS], axis=2)  # (b,L,12,2560)
        stacked = stacked[:, prefix_idx:]
        out_mask = valid_mx[:, prefix_idx:]
        return stacked.astype(self.dtype), out_mask
