"""Validate the grounded 12-layer tap (vision splice + deepstack + M-RoPE) vs full torch Qwen3-VL."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

from krea2_engine.text_encoder import (GROUNDED_TEMPLATE, IMAGE_PAD_ID, SELECT_LAYERS,
                                       Qwen3VLConditioner)
from krea2_engine.vision.preprocess import preprocess_image

BASE = os.path.expanduser("~/.cache/krea2_alis_mlx/krea__Krea-2-Turbo")
PROMPT = "make the shirt blue"

ref = np.load("reference/short_s0_512.npy")[0]
img = Image.fromarray((np.transpose(ref, (1, 2, 0)) * 255).round().clip(0, 255).astype(np.uint8))

# ---- MLX grounded tap (full sequence) ----
cond = Qwen3VLConditioner(BASE, dtype=mx.float32)
mx_stacked = cond.encode_grounded(PROMPT, img, grounding_px=384, return_all=True)  # (1, L, 12, 2560)
mx_stacked = np.array(mx_stacked.astype(mx.float32))[0]  # (L, 12, 2560)
print(f"[mlx] tap {mx_stacked.shape}")

# ---- torch reference: full Qwen3-VL model, same tokens, tap SELECT_LAYERS ----
pv, grid = preprocess_image(img, grounding_px=384)
n_img = pv.shape[0] // 4  # merged tokens = patches / merge^2
ids = cond.tokenizer(GROUNDED_TEMPLATE.format(PROMPT))["input_ids"]
ip = ids.index(IMAGE_PAD_ID)
ids = ids[:ip] + [IMAGE_PAD_ID] * n_img + ids[ip + 1:]

from transformers import Qwen3VLModel
pt = Qwen3VLModel.from_pretrained(
    f"{BASE}/text_encoder", dtype=torch.float32, low_cpu_mem_usage=True).eval()
with torch.no_grad():
    mm_ids = (np.array(ids) == IMAGE_PAD_ID).astype(np.int64)[None]
    out = pt(
        input_ids=torch.tensor([ids]),
        pixel_values=torch.tensor(pv),
        image_grid_thw=torch.tensor(grid.astype(np.int64)),
        mm_token_type_ids=torch.tensor(mm_ids),
        output_hidden_states=True,
    )
pt_hs = out.hidden_states  # tuple, len num_layers+1
pt_stacked = np.stack([pt_hs[i][0].float().numpy() for i in SELECT_LAYERS], axis=1)  # (L,12,2560)
print(f"[torch] tap {pt_stacked.shape}")

img_slice = (ip, ip + n_img)
np.savez("reference/grounded_cmp.npz", mx=mx_stacked, pt=pt_stacked,
         ids=np.array(ids), img_slice=np.array(img_slice))
d = np.abs(mx_stacked - pt_stacked)
cos = float((mx_stacked * pt_stacked).sum() /
            (np.linalg.norm(mx_stacked) * np.linalg.norm(pt_stacked)))
print(f"grounded tap  max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}  cos={cos:.8f}")
# localize: per tapped-layer and per token region
i0, i1 = img_slice
for li in range(mx_stacked.shape[1]):
    dl = d[:, li, :]
    reg = {"txt<": dl[:i0].max(), "img": dl[i0:i1].max(), "txt>": dl[i1:].max()}
    print(f"  layer {SELECT_LAYERS[li]:2d}: " + "  ".join(f"{k} {v:.2e}" for k, v in reg.items()))
# cosine ~1 and tiny mean|Δ| are the real signal; max|Δ| grows with f32 accumulation over 35
# layers at the high-magnitude image tokens (same faithfulness class as the trim's bf16 rounding).
print("PASS" if cos > 0.9999 and d.mean() < 1e-3 else "FAIL")
