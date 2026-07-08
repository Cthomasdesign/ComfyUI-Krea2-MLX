"""End-to-end: txt2img parity guard + grounded vs plain-text edit (identity LoRA)."""

import _shared  # noqa: F401
import mlx.core as mx
import numpy as np
from PIL import Image

from _shared import PROMPT_SHORT, load_pipeline

MODEL = "/Users/caseythomas/Desktop/Krea2ComfyUI/Models/transformer_mixed_4_8.safetensors"
LORA = "/Users/caseythomas/Desktop/Krea2ComfyUI/Models/loras/krea2_identity_edit_v1.safetensors"
pipe, _ = load_pipeline(MODEL)

# 1) parity: the grounded additions must not disturb txt2img
t2i = pipe.generate(PROMPT_SHORT, width=512, height=512, steps=8, seed=0)[0]
ref = np.load("reference/short_s0_512.npy")[0]
ref8 = (np.transpose(ref, (1, 2, 0)) * 255).round().clip(0, 255).astype(np.uint8)
print(f"[guard] txt2img pixel-identical to reference: {np.array_equal(np.array(t2i), ref8)}")

src = Image.fromarray(ref8)

# 2) grounded vs plain edit, identity LoRA, same seed
pipe.set_loras([(LORA, 1.0)])
plain = pipe.generate_edit("the fox wearing a red scarf", src, grounding_px=0,
                           width=512, height=512, steps=8, seed=0)[0]
grounded = pipe.generate_edit("the fox wearing a red scarf", src, grounding_px=384,
                              width=512, height=512, steps=8, seed=0)[0]
pipe.set_loras([])
plain.save("reference/edit_plain.png")
grounded.save("reference/edit_grounded.png")
sheet = [np.asarray(src.resize((384, 384))),
         np.asarray(plain.resize((384, 384))),
         np.asarray(grounded.resize((384, 384)))]
Image.fromarray(np.concatenate(sheet, axis=1)).save("reference/edit_grounded_sheet.png")
print("wrote reference/edit_grounded_sheet.png (source | plain | grounded)")
print("grounded edit ran end-to-end OK")
