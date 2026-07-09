"""Validate the img2img path: strength==1.0 regression, source round-trip, denoise sweep."""

import mlx.core as mx
import numpy as np
from PIL import Image

from _shared import PROMPT_SHORT, default_model, fox_source, load_pipeline

pipe, _ = load_pipeline(default_model())
src = fox_source()

# 1) strength==1.0 must equal txt2img (same prompt/seed/size) — the hard guard
t2i = pipe.generate(PROMPT_SHORT, width=512, height=512, steps=8, seed=0)[0]
i2i_full = pipe.generate_img2img(PROMPT_SHORT, src, denoise=1.0, width=512, height=512,
                                 steps=8, seed=0)[0]
same = np.array_equal(np.array(t2i), np.array(i2i_full))
print(f"[guard] img2img@denoise=1.0 == txt2img (pixel-identical): {same}")

# 2) denoise sweep: lower denoise should stay closer to the source
print("[sweep] similarity of output to source (lower denoise => more similar):")
sref = np.asarray(src, np.float32) / 255.0
out = {}
for d in (0.3, 0.5, 0.7, 0.9, 1.0):
    im = pipe.generate_img2img("the same scene, cinematic", src, denoise=d, width=512, height=512,
                               steps=8, seed=0)[0]
    a = np.asarray(im, np.float32) / 255.0
    mae = float(np.abs(a - sref).mean())
    out[d] = im
    print(f"   denoise={d:.1f}  mean|Δ| to source = {mae:.4f}")

# save a contact sheet for eyeballing
sheet = [np.asarray(src.resize((256, 256)))] + [np.asarray(out[d].resize((256, 256)))
                                                for d in (0.3, 0.5, 0.7, 0.9, 1.0)]
Image.fromarray(np.concatenate(sheet, axis=1)).save("reference/img2img_sweep.png")
print("wrote reference/img2img_sweep.png (source | 0.3 | 0.5 | 0.7 | 0.9 | 1.0)")
