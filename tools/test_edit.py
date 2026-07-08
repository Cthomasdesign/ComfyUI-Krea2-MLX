"""Validate the edit path: txt2img/img2img parity untouched, then a real identity-LoRA edit."""

import _shared  # noqa: F401
import mlx.core as mx
import numpy as np
from PIL import Image

from _shared import PROMPT_SHORT, load_pipeline

MODEL = "/Users/caseythomas/Desktop/Krea2ComfyUI/Models/transformer_mixed_4_8.safetensors"
LORA = "/Users/caseythomas/Desktop/Krea2ComfyUI/Models/loras/krea2_identity_edit_v1.safetensors"
pipe, _ = load_pipeline(MODEL)

# 1) parity guard: adding the edit code must not disturb txt2img
t2i = pipe.generate(PROMPT_SHORT, width=512, height=512, steps=8, seed=0)[0]
ref = np.load("reference/short_s0_512.npy")[0]
ref8 = (np.transpose(ref, (1, 2, 0)) * 255).round().clip(0, 255).astype(np.uint8)
print(f"[guard] txt2img still matches reference (pixel-identical): "
      f"{np.array_equal(np.array(t2i), ref8)}")

# source image for the edit: the fox reference
src = Image.fromarray(ref8)

# 2) edit WITHOUT the LoRA (baseline) vs WITH the identity LoRA — both should run; compare to source
def edit(prompt, seed, use_lora):
    pipe.set_loras([(LORA, 1.0)] if use_lora else [])
    im = pipe.generate_edit(prompt, src, width=512, height=512, steps=8, seed=seed)[0]
    pipe.set_loras([])
    return im

sref = np.asarray(src, np.float32) / 255.0
for use_lora in (False, True):
    im = edit("the fox wearing a red scarf", seed=0, use_lora=use_lora)
    a = np.asarray(im, np.float32) / 255.0
    mae = float(np.abs(a - sref).mean())
    tag = "LoRA" if use_lora else "base"
    im.save(f"reference/edit_{tag}.png")
    print(f"[edit:{tag}] ran; mean|Δ| to source = {mae:.4f}  -> reference/edit_{tag}.png")

# contact sheet: source | edit(base) | edit(LoRA)
sheet = [np.asarray(src.resize((384, 384)))]
for tag in ("base", "LoRA"):
    sheet.append(np.asarray(Image.open(f"reference/edit_{tag}.png").resize((384, 384))))
Image.fromarray(np.concatenate(sheet, axis=1)).save("reference/edit_sheet.png")
print("wrote reference/edit_sheet.png (source | base | LoRA)")
