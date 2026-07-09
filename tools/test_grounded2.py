"""Validate the TWO-image grounded tap (scene + subject) vs full torch Qwen3-VL."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

from krea2_engine.text_encoder import (IMAGE_PAD_ID, PREFIX, SELECT_LAYERS, SUFFIX,
                                       Qwen3VLConditioner)
from krea2_engine.vision.preprocess import preprocess_image

BASE = os.path.expanduser("~/.cache/krea2_alis_mlx/krea__Krea-2-Turbo")
PROMPT = "put the person in the scene"
GPX = 384

# two distinct source images: the fox (scene) and a recolored crop (subject)
ref = np.load("reference/short_s0_512.npy")[0]
imgA = Image.fromarray((np.transpose(ref, (1, 2, 0)) * 255).round().clip(0, 255).astype(np.uint8))
imgB = imgA.transpose(Image.FLIP_LEFT_RIGHT).resize((384, 512))  # a different-size, flipped ref

# ---- MLX two-image grounded tap ----
cond = Qwen3VLConditioner(BASE, dtype=mx.float32)
mxs = np.array(cond.encode_grounded(PROMPT, [imgA, imgB], grounding_px=GPX, return_all=True)
               .astype(mx.float32))[0]
print(f"[mlx] tap {mxs.shape}")

# ---- torch reference ----
pvA, gA = preprocess_image(imgA, grounding_px=GPX)
pvB, gB = preprocess_image(imgB, grounding_px=GPX)
nA, nB = pvA.shape[0] // 4, pvB.shape[0] // 4
block = "<|vision_start|><|image_pad|><|vision_end|>"
ids = cond.tokenizer((PREFIX + block * 2 + "{}" + SUFFIX).format(PROMPT))["input_ids"]
# expand the two pads to nA, nB
out, k = [], 0
for t in ids:
    if t == IMAGE_PAD_ID:
        out += [IMAGE_PAD_ID] * (nA if k == 0 else nB)
        k += 1
    else:
        out.append(t)
ids = out
pv = np.concatenate([pvA, pvB], axis=0)
grid = np.concatenate([gA, gB], axis=0).astype(np.int64)
mm = (np.array(ids) == IMAGE_PAD_ID).astype(np.int64)[None]

from transformers import Qwen3VLModel
pt = Qwen3VLModel.from_pretrained(f"{BASE}/text_encoder", dtype=torch.float32,
                                  low_cpu_mem_usage=True).eval()
with torch.no_grad():
    o = pt(input_ids=torch.tensor([ids]), pixel_values=torch.tensor(pv),
           image_grid_thw=torch.tensor(grid), mm_token_type_ids=torch.tensor(mm),
           output_hidden_states=True)
pt_stacked = np.stack([o.hidden_states[i][0].float().numpy() for i in SELECT_LAYERS], axis=1)
print(f"[torch] tap {pt_stacked.shape}")

d = np.abs(mxs - pt_stacked)
cos = float((mxs * pt_stacked).sum() / (np.linalg.norm(mxs) * np.linalg.norm(pt_stacked)))
print(f"two-image grounded tap  max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}  cos={cos:.8f}")
print("PASS" if cos > 0.9999 and d.mean() < 1e-3 else "FAIL")
