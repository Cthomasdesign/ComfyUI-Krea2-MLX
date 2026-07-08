"""Shared bootstrap for the standalone tools (bench/parity).

Run these with a Python that has the node's requirements installed (e.g. ComfyUI's venv).
They import `krea2_engine` straight from this repo checkout — no ComfyUI needed.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# fixed prompts for benchmarking/parity: a typical short prompt and a long detailed one
PROMPT_SHORT = "a fox in the snow"
PROMPT_LONG = (
    "a highly detailed cinematic photograph of an old lighthouse keeper standing on a rocky "
    "cliff at dusk, weathered face lit by the rotating beam, heavy wool coat flapping in the "
    "storm wind, enormous waves crashing against the rocks below sending up walls of spray, "
    "dramatic dark storm clouds with a thin band of orange sunset light on the horizon, "
    "seabirds struggling against the gale, wet stone textures, rim lighting, shallow depth of "
    "field, shot on large format film, extremely detailed, moody atmospheric color grading"
)


def precision_for(filename):
    """Infer the quantization recipe from a transformer filename (mirrors nodes.py)."""
    n = os.path.basename(filename).lower()
    if "mixed" in n or "4_8" in n or "4-8" in n:
        return "mixed-4-8"
    if "8bit" in n or "8-bit" in n:
        return "8bit"
    if "bf16" in n or "turbo" in n:
        return "bf16"
    raise ValueError(f"cannot infer precision from filename: {filename}")


def default_base_dir():
    """Local snapshot of krea/Krea-2-Turbo (VAE/encoder/tokenizer), if already cached."""
    d = os.path.expanduser("~/.cache/krea2_alis_mlx/krea__Krea-2-Turbo")
    return d if os.path.isdir(os.path.join(d, "text_encoder")) else None


def load_pipeline(model_path):
    import time

    from krea2_engine.pipeline import Krea2Pipeline

    t0 = time.perf_counter()
    pipe = Krea2Pipeline(model_path, precision=precision_for(model_path),
                         base_dir=default_base_dir())
    return pipe, time.perf_counter() - t0
