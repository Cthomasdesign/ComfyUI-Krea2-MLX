"""MLX flow-matching sampler for Krea-2 (port of krea-2-official/sampling.py)."""

from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np


def roundup(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def patchify(x: mx.array, p: int) -> mx.array:
    # (b, c, H, W) -> (b, (H/p)*(W/p), c*p*p)   [c ph pw] ordering
    b, c, H, W = x.shape
    h, w = H // p, W // p
    x = x.reshape(b, c, h, p, w, p).transpose(0, 2, 4, 1, 3, 5)
    return x.reshape(b, h * w, c * p * p)


def unpatchify(x: mx.array, p: int, h: int, w: int, c: int) -> mx.array:
    # (b, h*w, c*p*p) -> (b, c, h*p, w*p)
    b = x.shape[0]
    x = x.reshape(b, h, w, c, p, p).transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h * p, w * p)


def build_positions(b: int, txtlen: int, h_: int, w_: int) -> mx.array:
    txtpos = np.zeros((txtlen, 3), np.float32)
    imgids = np.zeros((h_, w_, 3), np.float32)
    imgids[..., 1] = np.arange(h_)[:, None]
    imgids[..., 2] = np.arange(w_)[None, :]
    pos = np.concatenate([txtpos, imgids.reshape(-1, 3)], axis=0)
    return mx.array(pos)


def _trim_context(ctx, mask):
    """Drop text positions that are padding in every row (layout: [prompt, padding, suffix]).

    Mathematically exact: padded keys get exactly-zero attention weight (the -1e9 additive
    mask underflows exp to 0), text RoPE positions are all zero, and only image-token outputs
    are used. The shorter sequence does tile the bf16 kernels differently, so a fixed seed
    renders equal-quality but not bit-identical images vs the padded path — set
    KREA2_EXACT_LEGACY=1 to keep the old padded behavior (slower). Keeps the union of valid
    columns, so mixed-length batches stay correct via the per-row mask."""
    if os.environ.get("KREA2_EXACT_LEGACY"):
        return ctx, mask
    valid = np.array(mask) > 0.5  # (B, L)
    keep = np.flatnonzero(valid.any(axis=0))
    if len(keep) == valid.shape[1]:
        return ctx, mask
    idx = mx.array(keep.astype(np.int32))
    return mx.take(ctx, idx, axis=1), mx.take(mask, idx, axis=1)


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    ts = np.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    with np.errstate(divide="ignore"):
        ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def sample(
    transformer,
    vae,
    encode,            # callable: list[str] -> (context mx, mask mx)
    prompts,
    *,
    width=1024,
    height=1024,
    steps=8,
    guidance=0.0,      # turbo: no CFG
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    init_noise=None,   # (n,16,H/8,W/8) to match a PT run; else MLX RNG
    init_latent=None,  # (n,16,H/8,W/8) clean source latent for img2img (VAE-encoded); else txt2img
    strength=1.0,      # img2img denoise strength: 1.0 = full (== txt2img), lower keeps more source
    dtype=mx.bfloat16,
    step_callback=None,  # called as step_callback(step, total) after each denoising step
):
    cfg = guidance > 0
    patch = transformer.cfg.patch
    comp = vae.spatial_scale  # 8
    align = comp * patch
    width, height = roundup(width, align), roundup(height, align)
    n = len(prompts)

    lat_h, lat_w = height // comp, width // comp
    if init_noise is None:
        mx.random.seed(seed)
        noise = mx.random.normal((n, vae.latent_channels, lat_h, lat_w)).astype(dtype)
    else:
        noise = mx.array(init_noise).astype(dtype)

    ctx, mask = encode(prompts)
    ctx, mask = _trim_context(ctx, mask)
    ctx = ctx.astype(dtype)
    txtlen = ctx.shape[1]
    h_, w_ = lat_h // patch, lat_w // patch

    img = patchify(noise, patch)  # (n, h_*w_, 64)
    pos = build_positions(n, txtlen, h_, w_)
    full_mask = mx.concatenate([mask, mx.ones((n, h_ * w_))], axis=1)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # img2img: enter the (rectified-flow) schedule partway and start from a noised source latent.
    # x_t = (1-t)·z0 + t·noise, matching the loop's `img += (tp-tc)·v` (t: 1→0, noise at t=1). We
    # enter at the first schedule point t ≤ strength. strength≥1.0 (or no source) is a no-op, so
    # the txt2img path below is byte-identical to before.
    if init_latent is not None and strength < 1.0:
        z0 = patchify(mx.array(init_latent).astype(dtype), patch)  # (n, h_*w_, 64), clean source
        k = next((i for i, t in enumerate(ts) if t <= strength), len(ts) - 1)
        ts = ts[k:]
        t_enter = ts[0]
        img = (1.0 - t_enter) * z0 + t_enter * img  # img is patchify(noise) here

    # step-invariant conditioning (text fusion, rope, masks) — computed once, reused every step
    fused_ctx, cos, sin, add_mask = transformer.prepare_conditioning(ctx, pos, full_mask, dtype)
    mx.eval(fused_ctx, cos, sin, *([add_mask] if add_mask is not None else []))

    total = len(ts) - 1
    for i, (tc, tp) in enumerate(zip(ts[:-1], ts[1:])):
        t = mx.full((n,), tc, dtype=dtype)
        v = transformer.denoise_step(img, fused_ctx, t, cos, sin, add_mask)
        if cfg:
            raise NotImplementedError("CFG path not needed for turbo")
        img = img + (tp - tc) * v
        mx.eval(img)
        if step_callback is not None:
            step_callback(i + 1, total)

    latent = unpatchify(img, patch, h_, w_, vae.latent_channels)  # (n,16,lat_h,lat_w)
    decoded = vae.decode(latent.astype(mx.float32))  # (n,3,1,H,W)
    decoded = mx.clip(decoded, -1, 1) * 0.5 + 0.5
    decoded = decoded[:, :, 0]  # (n,3,H,W)
    mx.eval(decoded)
    return decoded


def to_pil(decoded: mx.array):
    from PIL import Image

    arr = np.array(decoded.astype(mx.float32))  # (n,3,H,W)
    arr = (np.transpose(arr, (0, 2, 3, 1)) * 255.0).round().clip(0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
