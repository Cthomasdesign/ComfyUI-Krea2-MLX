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


# --- machine-portable test assets ---------------------------------------------------------
# The test scripts resolve their model/LoRA/source image through these helpers instead of
# hardcoded paths: override via env vars, else discover next to the repo checkout.

def _discover(env_var, candidates, what, hint):
    p = os.environ.get(env_var)
    if p:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{env_var}={p} does not exist")
        return p
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"No {what} found. Set {env_var}, or place one at: {' or '.join(candidates)}. {hint}")


# ComfyUI model stores to search, most-canonical first: the Comfy Desktop shared store, then a
# legacy Models/ folder next to the repo. Override either file explicitly with the env var.
_SHARED = os.path.expanduser("~/ComfyUI-Shared/models")
_LEGACY = os.path.join(os.path.dirname(REPO), "Models")


def default_model():
    """Transformer build for the test scripts (env KREA2_TEST_MODEL, else the ComfyUI model store)."""
    names = ("transformer_mixed_4_8.safetensors", "transformer_8bit.safetensors")
    candidates = ([os.path.join(_SHARED, "krea2", n) for n in names]
                  + [os.path.join(_LEGACY, n) for n in names])
    return _discover("KREA2_TEST_MODEL", candidates, "Krea-2 transformer build",
                     "Download from huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8.")


def default_lora():
    """Identity-edit LoRA for the edit tests (env KREA2_TEST_LORA, else the ComfyUI model store)."""
    name = "krea2_identity_edit_v1.safetensors"
    candidates = [os.path.join(_SHARED, "loras", name), os.path.join(_LEGACY, "loras", name)]
    return _discover("KREA2_TEST_LORA", candidates, "identity-edit LoRA",
                     "Download from huggingface.co/conradlocke/krea2-identity-edit.")


def ref_uint8():
    """The captured txt2img reference (fox, seed 0, 512²) as a uint8 (H,W,3) array."""
    import numpy as np

    ref_path = os.path.join(REPO, "tools", "reference", "short_s0_512.npy")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(
            f"{ref_path} missing — run `parity.py --capture` first to snapshot references.")
    ref = np.load(ref_path)[0]  # (3,512,512) in 0..1
    return (np.transpose(ref, (1, 2, 0)) * 255.0).round().clip(0, 255).astype(np.uint8)


def fox_source():
    """The reference render as a PIL image — the standard source image for the feature tests."""
    from PIL import Image

    return Image.fromarray(ref_uint8())
