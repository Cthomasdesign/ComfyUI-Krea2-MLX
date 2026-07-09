"""Pure-MLX Qwen3-VL-4B text encoder for Krea-2 conditioning.

Two conditioning paths, both tapping the same 12 selected per-layer hidden states
(B, seq, 12, 2560) that the Krea-2 DiT consumes (matching krea-2-official/encoder.py):

- Text-only (`Qwen3VLConditioner.__call__`): mrope collapses to standard rope (all 3
  position dims equal), so this is a standard Qwen3 decoder (RMSNorm, GQA, per-head
  QK-norm, rope θ=5e6, head_dim 128 decoupled from hidden 2560).
- Image-grounded (`encode_grounded`): the instruction is encoded *while looking at*
  reference image(s) — vision-tower tokens spliced at <|image_pad|>, deepstack features
  injected after decoder layers 0..2, interleaved multimodal RoPE (3D t/h/w positions).
  Validated against the full PyTorch Qwen3-VL reference (see tools/test_grounded*.py).

Tokenization uses the HF tokenizer (as mflux does); all model forwards are pure MLX.
"""

from __future__ import annotations

import glob
import json
import os

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

# --- image grounding (Qwen3-VL vision path) ---------------------------------
IMAGE_PAD_ID = 151655  # <|image_pad|> — placeholder each vision block expands into
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
# grounded edit template = system prefix + vision block(s) + instruction + assistant suffix
GROUNDED_TEMPLATE = PREFIX + VISION_BLOCK + "{}" + SUFFIX


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
        # GQA: mx.fast SDPA groups kv heads natively (bit-identical to repeating them)
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

    def _rope(self, pos):
        # pos: (L,) absolute position ids (not necessarily contiguous)
        inv = 1.0 / (self.theta ** (mx.arange(0, self.head_dim, 2).astype(mx.float32) / self.head_dim))
        freqs = pos.astype(mx.float32)[:, None] * inv[None, :]  # (L, hd/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (L, hd)
        return mx.cos(emb), mx.sin(emb)

    def __call__(self, input_ids, attn_valid, pos_ids=None):
        # input_ids: (b,L) int; attn_valid: (b,L) {0,1}; pos_ids: (L,) absolute positions
        b, L = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self._rope(pos_ids if pos_ids is not None else mx.arange(L))
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

    def _mrope(self, pos3d):
        """Interleaved multimodal RoPE (Qwen3-VL). pos3d: (3, L) t/h/w position ids -> (cos, sin)
        each (L, head_dim). Matches transformers' apply_interleaved_mrope (mrope_section 24/20/20):
        freq index k<60 takes dim k%3 (T/H/W interleaved), k>=60 takes T."""
        hd = self.head_dim
        inv = 1.0 / (self.theta ** (mx.arange(0, hd, 2).astype(mx.float32) / hd))  # (hd/2,)
        freqs = pos3d[:, :, None].astype(mx.float32) * inv[None, None, :]           # (3, L, hd/2)
        k = mx.arange(hd // 2)
        which = mx.where(k < 60, k % 3, 0).astype(mx.int32)                         # (hd/2,) dim per freq
        idx = mx.broadcast_to(which[None, None, :], (1, freqs.shape[1], hd // 2))
        ft = mx.take_along_axis(freqs, idx, axis=0)[0]                              # (L, hd/2)
        emb = mx.concatenate([ft, ft], axis=-1)                                     # (L, hd)
        return mx.cos(emb), mx.sin(emb)

    def forward_grounded(self, inputs_embeds, pos3d, deepstack_embeds, img_slices):
        """Grounded decoder pass. inputs_embeds: (1, L, hidden) with image rows already spliced in;
        pos3d: (3, L) M-RoPE positions; deepstack_embeds: per-layer concatenated (total_img, hidden)
        features added at the image rows after layers 0..len-1; img_slices: list of (start, end) row
        ranges, one per image block (features concatenated in block order).
        Returns the HF-indexed hidden-state list (tap SELECT_LAYERS as in __call__)."""
        h = inputs_embeds
        L = h.shape[1]
        cos, sin = self._mrope(pos3d)
        cos, sin = cos.astype(h.dtype), sin.astype(h.dtype)
        idx = mx.arange(L)
        mask = ((idx[None, :] > idx[:, None]).astype(mx.float32) * -1e9)[None, None].astype(h.dtype)

        # HF records hidden_states[i+1] as the layer OUTPUT (before deepstack), while deepstack is
        # still injected into the CONTINUING stream — so tapped layer 2 must exclude its deepstack
        # add even though later layers see it.
        all_hs = [h]  # hidden_states[0] = token embeddings
        for li, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask)
            all_hs.append(h)  # hidden_states[li+1], pre-deepstack
            if deepstack_embeds is not None and li < len(deepstack_embeds):
                add = mx.zeros_like(h)
                off = 0
                for s, e in img_slices:
                    add[:, s:e, :] = deepstack_embeds[li][off:off + (e - s)][None].astype(h.dtype)
                    off += e - s
                h = h + add
        return all_hs


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


def _get_rope_index(ids: list[int], grids: list, merge: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """3-axis M-RoPE position ids (3, L) for one clean sequence with any number of image blocks,
    matching transformers' get_rope_index: text tokens are sequential (t=h=w); each image block's
    tokens take t=frame, h=row, w=col on its merged grid, offset to continue after the preceding
    text; text after an image resumes at max(image positions)+1. Returns (pos, img_slices)."""
    L = len(ids)
    pos = np.zeros((3, L), np.int64)
    img_slices = []
    st_idx = 0   # running rope position
    ig = 0       # image counter
    i = 0
    while i < L:
        if ids[i] == IMAGE_PAD_ID:
            gt, gh, gw = int(grids[ig][0]), int(grids[ig][1]) // merge, int(grids[ig][2]) // merge
            n = gt * gh * gw
            t_index = np.repeat(np.arange(gt), gh * gw)
            h_index = np.tile(np.repeat(np.arange(gh), gw), gt)
            w_index = np.tile(np.arange(gw), gt * gh)
            pos[0, i:i + n] = t_index + st_idx
            pos[1, i:i + n] = h_index + st_idx
            pos[2, i:i + n] = w_index + st_idx
            img_slices.append((i, i + n))
            st_idx = int(max(t_index.max(), h_index.max(), w_index.max())) + st_idx + 1
            i += n
            ig += 1
        else:
            pos[:, i] = st_idx      # text token: sequential
            st_idx += 1
            i += 1
    return pos, img_slices


class Qwen3VLConditioner:
    """Pure-MLX conditioner. Tokenization via HF tokenizer; forward via MLX."""

    def __init__(self, repo: str, max_length: int = 512, dtype=mx.float32):
        from transformers import AutoTokenizer

        self.repo = repo
        self._vision = None
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
        max_inp_len = self.max_length + prefix_idx - SUFFIX_START_IDX
        text = [PREFIX + p for p in prompts]
        suffix = [SUFFIX] * len(text)
        suf = self.tokenizer(text=suffix, return_tensors="np")
        # Pad only to the longest prompt in the batch instead of always to max_inp_len; the
        # suffix keeps its absolute RoPE positions from the padded layout (positions are
        # assigned by index, and only trailing pad columns are dropped, so real-token rotations
        # are unchanged). KREA2_EXACT_LEGACY=1 restores the always-max padding.
        legacy = bool(os.environ.get("KREA2_EXACT_LEGACY"))
        inp = self.tokenizer(
            text, truncation=True, padding="max_length" if legacy else True,
            max_length=max_inp_len, return_tensors="np",
        )
        input_ids = np.concatenate([inp["input_ids"], suf["input_ids"]], axis=1)
        mask = np.concatenate([inp["attention_mask"], suf["attention_mask"]], axis=1)
        inp_len = inp["input_ids"].shape[1]
        pos_ids = np.concatenate([np.arange(inp_len), max_inp_len + np.arange(SUFFIX_START_IDX)])

        ids_mx = mx.array(input_ids.astype(np.int32))
        valid_mx = mx.array(mask.astype(np.float32))
        all_hs = self.model(ids_mx, valid_mx, pos_ids=mx.array(pos_ids.astype(np.int32)))
        stacked = mx.stack([all_hs[i] for i in SELECT_LAYERS], axis=2)  # (b,L,12,2560)
        stacked = stacked[:, prefix_idx:]
        out_mask = valid_mx[:, prefix_idx:]
        return stacked.astype(self.dtype), out_mask

    def _load_vision(self):
        """Lazily build + load the Qwen3-VL vision tower (visual.* weights, ~1 GB) on first use."""
        if self._vision is None:
            from .vision import VisionConfig, VisionModel

            cfg = VisionConfig.from_dict(
                json.load(open(f"{self.repo}/text_encoder/config.json"))["vision_config"])
            vm = VisionModel(cfg)
            w = {}
            for sh in sorted(glob.glob(f"{self.repo}/text_encoder/*.safetensors")):
                for k, v in mx.load(sh).items():
                    if k.startswith("visual."):
                        w[k[len("visual."):]] = v.astype(self.dtype)
            vm.load_weights(list(vm.sanitize(w).items()))
            mx.eval(vm.parameters())
            self._vision = vm
        return self._vision

    def encode_grounded(self, prompt: str, images, grounding_px: int = 768,
                        return_all: bool = False):
        """Image-grounded conditioning: run `images` (one PIL image or a list, training order
        scene-first) through the Qwen3-VL vision tower, splice the vision tokens into the
        instruction (one <vision> block per image), run the decoder with deepstack + M-RoPE, and
        tap the same 12 layers. Returns (ctx (1, seq, 12, 2560), mask) like __call__ — sliced past
        the system prefix. Torch-free at runtime. Batch size 1."""
        from .vision.preprocess import preprocess_image

        if not isinstance(images, (list, tuple)):
            images = [images]
        vm = self._load_vision()
        embeds, deeps, grids = [], [], []
        for im in images:
            pv, grid = preprocess_image(im, grounding_px=grounding_px)
            e, d = vm(mx.array(pv), mx.array(grid))
            mx.eval(e, *d)
            embeds.append(e)
            deeps.append(list(d))
            grids.append(grid[0])

        template = PREFIX + VISION_BLOCK * len(images) + "{}" + SUFFIX
        ids = self.tokenizer(template.format(prompt))["input_ids"]
        counts = [e.shape[0] for e in embeds]
        expanded, k = [], 0
        for t in ids:  # expand the k-th <image_pad> placeholder to its n_k vision tokens
            if t == IMAGE_PAD_ID:
                expanded += [IMAGE_PAD_ID] * counts[k]
                k += 1
            else:
                expanded.append(t)
        ids = expanded
        pos3d, img_slices = _get_rope_index(ids, grids, merge=2)

        h = self.model.embed_tokens(mx.array(np.array(ids, np.int32))[None])  # (1, L, hidden)
        cat_embeds = mx.concatenate(embeds, axis=0)                          # (total_img, hidden)
        off = 0
        for s, e in img_slices:
            h[:, s:e, :] = cat_embeds[off:off + (e - s)][None].astype(h.dtype)
            off += e - s
        # deepstack per layer = concat across images (block order), matching img_slices
        n_deep = len(deeps[0])
        deepstack = [mx.concatenate([deeps[j][li] for j in range(len(images))], axis=0)
                     for li in range(n_deep)]
        all_hs = self.model.forward_grounded(h, mx.array(pos3d), deepstack, img_slices)
        stacked = mx.stack([all_hs[i] for i in SELECT_LAYERS], axis=2)        # (1, L, 12, 2560)
        if return_all:
            return stacked.astype(self.dtype)
        stacked = stacked[:, PREFIX_START_IDX:]
        mask = mx.ones((1, stacked.shape[1]))
        return stacked.astype(self.dtype), mask
