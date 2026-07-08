"""Numerical validation: vendored MLX Qwen3-VL vision tower vs the transformers (torch) reference.

Feeds identical (pixel_values, grid_thw) through both and compares the merged image embeds and
each deepstack feature. Random pixels are a fine torture test for forward-pass equivalence.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
import torch

from krea2_engine.vision import VisionConfig, VisionModel

BASE = os.path.expanduser("~/.cache/krea2_alis_mlx/krea__Krea-2-Turbo")
vc_json = json.load(open(f"{BASE}/text_encoder/config.json"))["vision_config"]

# a small image grid: 1 temporal frame, 32x32 patches (must be multiples of spatial_merge_size=2)
gt, gh, gw = 1, 32, 32
patch_dim = vc_json["in_channels"] * vc_json["temporal_patch_size"] * vc_json["patch_size"] ** 2
rng = np.random.default_rng(0)
pixel_values = rng.standard_normal((gt * gh * gw, patch_dim)).astype(np.float32)
grid_thw = np.array([[gt, gh, gw]], dtype=np.int64)

# raw visual.* weights (torch tensors) for both models
weights_pt = {}
for sh in sorted(glob.glob(f"{BASE}/text_encoder/*.safetensors")):
    from safetensors.torch import load_file
    for k, v in load_file(sh).items():
        if k.startswith("visual."):
            weights_pt[k[len("visual."):]] = v

# --- torch reference ---
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

pt_cfg = Qwen3VLVisionConfig(**{k: v for k, v in vc_json.items()})
pt = Qwen3VLVisionModel(pt_cfg).eval()
missing, unexpected = pt.load_state_dict(weights_pt, strict=False)
print(f"[torch] loaded (missing={len(missing)}, unexpected={len(unexpected)})")
with torch.no_grad():
    out = pt(torch.tensor(pixel_values), torch.tensor(grid_thw))
    pt_embeds = out.pooler_output.float().numpy()  # merger output = what we splice into text
    pt_deep = [d.float().numpy() for d in out.deepstack_features]
print(f"[torch] embeds {pt_embeds.shape}, deepstack {len(pt_deep)}x{pt_deep[0].shape}")

# --- MLX vendored ---
cfg = VisionConfig.from_dict(vc_json)
vm = VisionModel(cfg)
w_mlx = {k: mx.array(v.float().numpy()) for k, v in weights_pt.items()}
w_mlx = vm.sanitize(w_mlx)
vm.load_weights(list(w_mlx.items()))
mx.eval(vm.parameters())
mx_embeds, mx_deep = vm(mx.array(pixel_values), mx.array(grid_thw))
mx.eval(mx_embeds, *mx_deep)
mx_embeds = np.array(mx_embeds.astype(mx.float32))
mx_deep = [np.array(d.astype(mx.float32)) for d in mx_deep]

def cmp(a, b, name):
    d = np.abs(a - b)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"  {name:16s} max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}  cos={cos:.8f}")
    return d.max()

print("[compare] MLX vs torch:")
worst = cmp(mx_embeds, pt_embeds, "merged embeds")
for i, (a, b) in enumerate(zip(mx_deep, pt_deep)):
    worst = max(worst, cmp(a, b, f"deepstack[{i}]"))
print(f"\n{'PASS' if worst < 1e-2 else 'FAIL'}: worst max|Δ| = {worst:.3e}")
