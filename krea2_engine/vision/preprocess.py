"""Torch-free Qwen3-VL image preprocessing → (pixel_values, grid_thw).

Matches transformers' Qwen2-VL/Qwen3-VL image processor (validated in tools/test_vision_tower.py):
smart-resize to multiples of patch*merge, CLIP rescale+normalize, then the patch flatten the vision
tower expects. `grounding_px` caps the longest side before smart-resize (as in comfyui-krea2edit).
"""

from __future__ import annotations

import math

import numpy as np

IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def smart_resize(h, w, factor, min_pixels, max_pixels):
    """Nearest (h,w) with both divisible by `factor`, pixel count in [min,max], ratio preserved."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(image, *, patch_size=16, merge_size=2, temporal_patch_size=2,
                     grounding_px=768, min_pixels=None, max_pixels=None):
    """PIL image -> (pixel_values [num_patches, C*T*P*P] float32, grid_thw [[1,gh,gw]] int32)."""
    from PIL import Image

    image = image.convert("RGB")
    w0, h0 = image.size
    factor = patch_size * merge_size
    if min_pixels is None:
        min_pixels = factor * factor
    if max_pixels is None:
        # area ceiling from grounding_px, clamped so a tiny/garbage value can't zero it out
        eff = max(int(grounding_px or 1536), factor)
        max_pixels = (eff // factor * factor) ** 2

    # cap the longest side to grounding_px first (as comfyui-krea2edit does), then snap to the grid
    if grounding_px and grounding_px >= factor and max(h0, w0) > grounding_px:
        s = grounding_px / max(h0, w0)
        h0, w0 = round(h0 * s), round(w0 * s)
    rh, rw = smart_resize(h0, w0, factor, min_pixels, max_pixels)

    image = image.resize((rw, rh), Image.Resampling.BICUBIC)
    arr = np.asarray(image, np.float32) / 255.0          # (H,W,3) in 0..1
    arr = (arr - IMAGE_MEAN) / IMAGE_STD                  # CLIP normalize
    arr = np.transpose(arr, (2, 0, 1))                   # (C,H,W)

    c = arr.shape[0]
    gh, gw = rh // patch_size, rw // patch_size
    p, m, tp = patch_size, merge_size, temporal_patch_size
    # (C, gh/m, m, p, gw/m, m, p) -> (gh/m, gw/m, m, m, C, p, p)  [block-major token order]
    patches = arr.reshape(c, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)
    # insert the temporal axis after C and duplicate the single frame → (…, C, T, p, p)
    patches = np.broadcast_to(patches[:, :, :, :, :, None, :, :],
                              (gh // m, gw // m, m, m, c, tp, p, p))
    pixel_values = patches.reshape(gh * gw, c * tp * p * p).astype(np.float32)
    grid_thw = np.array([[1, gh, gw]], np.int32)
    return pixel_values, grid_thw
