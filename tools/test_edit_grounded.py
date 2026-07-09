"""End-to-end: txt2img parity guard + grounded vs plain-text edit (identity LoRA)."""

import numpy as np
from PIL import Image

from _shared import PROMPT_SHORT, default_lora, default_model, fox_source, load_pipeline, ref_uint8

pipe, _ = load_pipeline(default_model())
LORA = default_lora()

# 1) parity: the grounded additions must not disturb txt2img
t2i = pipe.generate(PROMPT_SHORT, width=512, height=512, steps=8, seed=0)[0]
print(f"[guard] txt2img pixel-identical to reference: "
      f"{np.array_equal(np.array(t2i), ref_uint8())}")

src = fox_source()

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
