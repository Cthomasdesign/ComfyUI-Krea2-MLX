"""MLX flow-matching sampler for Krea-2 (port of krea-2-official/sampling.py)."""

from __future__ import annotations

import math

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
    ctx = ctx.astype(dtype)
    txtlen = ctx.shape[1]
    h_, w_ = lat_h // patch, lat_w // patch

    img = patchify(noise, patch)  # (n, h_*w_, 64)
    pos = build_positions(n, txtlen, h_, w_)
    full_mask = mx.concatenate([mask, mx.ones((n, h_ * w_))], axis=1)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    total = len(ts) - 1
    for i, (tc, tp) in enumerate(zip(ts[:-1], ts[1:])):
        t = mx.full((n,), tc, dtype=dtype)
        v = transformer(img, ctx, t, pos, full_mask)
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
