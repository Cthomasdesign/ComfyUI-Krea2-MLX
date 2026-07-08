"""ComfyUI nodes for Krea-2-Turbo on Apple MLX.

Runs the vendored `krea2_engine` MLX pipeline in-process and returns ComfyUI IMAGE tensors — a
self-contained "island": load model → (stack LoRAs) → generate → IMAGE, then any downstream
ComfyUI node can save/upscale/post-process. It does not expose a torch MODEL, so it does not plug
into ComfyUI's native KSampler/ControlNet.

Nodes (category "Krea2 MLX"):
  Krea2ModelLoader   pick a transformer build in ComfyUI/models/krea2 -> KREA2_PIPE (cached)
  Krea2LoRA          [stack] name strength -> KREA2_LORASTACK          (chain to stack multiple)
  Krea2Generate      pipe prompt … [stack]  -> IMAGE
  Krea2Unload        ()                      -> frees the cached model + Metal cache
"""

import gc
import os

import numpy as np
import torch

import folder_paths  # ComfyUI
from comfy.utils import ProgressBar
import comfy.model_management as mm

from .krea2_engine.pipeline import Krea2Pipeline

# custom socket types for passing objects between our nodes
KREA2_PIPE = "KREA2_PIPE"
KREA2_LORASTACK = "KREA2_LORASTACK"

# transformer builds are looked up here (idiomatic ComfyUI models/ subfolder)
KREA2_MODELS = os.path.join(folder_paths.models_dir, "krea2")

# one model resident at a time (a 12.9B transformer won't share unified memory with another)
_PIPE_CACHE: dict[str, Krea2Pipeline] = {}


def _precision_for(filename):
    """Infer the quantization recipe from a transformer filename (None if unrecognized)."""
    n = filename.lower()
    if "mixed" in n or "4_8" in n or "4-8" in n:
        return "mixed-4-8"
    if "8bit" in n or "8-bit" in n:
        return "8bit"
    if "bf16" in n or "turbo" in n:
        return "bf16"
    return None


def _list_models():
    """Transformer .safetensors builds found in ComfyUI/models/krea2."""
    if not os.path.isdir(KREA2_MODELS):
        return []
    return [f for f in sorted(os.listdir(KREA2_MODELS))
            if f.endswith(".safetensors") and _precision_for(f)]


def _to_image_tensor(pil_images):
    """List of PIL images -> ComfyUI IMAGE tensor, float32 [B, H, W, C] in 0..1."""
    arrs = [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in pil_images]
    return torch.from_numpy(np.stack(arrs, axis=0))


def _from_image_tensor(image):
    """ComfyUI IMAGE tensor [B, H, W, C] in 0..1 -> the first frame as a PIL image."""
    from PIL import Image

    arr = (image[0].cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _get_pipe(model_file):
    if model_file not in _PIPE_CACHE:
        _PIPE_CACHE.clear()          # free the previous build first
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        path = os.path.join(KREA2_MODELS, model_file)
        _PIPE_CACHE[model_file] = Krea2Pipeline(path, precision=_precision_for(model_file),
                                                base_dir=os.environ.get("KREA2_BASE_DIR"))
    return _PIPE_CACHE[model_file]


class Krea2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _list_models() or ["(put transformer_*.safetensors in ComfyUI/models/krea2)"]
        return {"required": {"model": (models,)}}

    RETURN_TYPES = (KREA2_PIPE,)
    RETURN_NAMES = ("krea2_pipe",)
    FUNCTION = "load"
    CATEGORY = "Krea2 MLX"

    def load(self, model):
        if not _list_models():
            raise RuntimeError(
                f"No Krea-2 transformer in {KREA2_MODELS}. Download a build there, e.g. "
                "transformer_mixed_4_8.safetensors from "
                "huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8")
        return (_get_pipe(model),)


class Krea2LoRA:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (loras if loras else ["None"],),
                "strength": ("FLOAT", {"default": 0.9, "min": -2.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {"lora_stack": (KREA2_LORASTACK,)},
        }

    RETURN_TYPES = (KREA2_LORASTACK,)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "append"
    CATEGORY = "Krea2 MLX"

    def append(self, lora_name, strength, lora_stack=None):
        stack = list(lora_stack) if lora_stack else []
        if lora_name and lora_name != "None" and strength != 0:
            path = folder_paths.get_full_path("loras", lora_name)
            if path:
                stack.append((path, float(strength)))
        return (stack,)


class Krea2Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipe": (KREA2_PIPE,),
                "prompt": ("STRING", {"multiline": True, "default": "a fox in the snow"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "safety_filter": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
            },
            "optional": {"lora_stack": (KREA2_LORASTACK,)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Krea2 MLX"

    def generate(self, krea2_pipe, prompt, width, height, steps, seed, num_images,
                 safety_filter=True, lora_stack=None):
        krea2_pipe.set_loras(lora_stack or [])
        pbar = ProgressBar(steps)

        def cb(step, total):
            mm.throw_exception_if_processing_interrupted()  # honor ComfyUI's Cancel button
            pbar.update_absolute(step, total)

        imgs = krea2_pipe.generate(prompt, width=width, height=height, steps=steps,
                                   seed=seed, num_images=num_images, step_callback=cb)
        # NSFW content filter (on by default). The Krea 2 Community License (§4.2) requires
        # reasonable content-filtering in deployments; flagged images are redacted.
        if safety_filter:
            from .krea2_engine import safety
            imgs, _ = safety.apply(imgs, enabled=True)
        return (_to_image_tensor(imgs),)


class Krea2Img2Img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipe": (KREA2_PIPE,),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "a fox in the snow"}),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "safety_filter": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
            },
            "optional": {"lora_stack": (KREA2_LORASTACK,)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Krea2 MLX"

    def generate(self, krea2_pipe, image, prompt, denoise, width, height, steps, seed, num_images,
                 safety_filter=True, lora_stack=None):
        krea2_pipe.set_loras(lora_stack or [])
        pbar = ProgressBar(steps)

        def cb(step, total):
            mm.throw_exception_if_processing_interrupted()  # honor ComfyUI's Cancel button
            pbar.update_absolute(step, total)

        imgs = krea2_pipe.generate_img2img(prompt, _from_image_tensor(image), denoise=denoise,
                                           width=width, height=height, steps=steps, seed=seed,
                                           num_images=num_images, step_callback=cb)
        # NSFW content filter (on by default). The Krea 2 Community License (§4.2) requires
        # reasonable content-filtering in deployments; flagged images are redacted.
        if safety_filter:
            from .krea2_engine import safety
            imgs, _ = safety.apply(imgs, enabled=True)
        return (_to_image_tensor(imgs),)


class Krea2Edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipe": (KREA2_PIPE,),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "make the shirt blue"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "safety_filter": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
            },
            "optional": {
                "image_b": ("IMAGE", {"tooltip": "2nd reference (subject) for multi-ref edit LoRAs"}),
                "lora_stack": (KREA2_LORASTACK,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Krea2 MLX"

    def generate(self, krea2_pipe, image, prompt, width, height, steps, seed, num_images,
                 safety_filter=True, image_b=None, lora_stack=None):
        krea2_pipe.set_loras(lora_stack or [])
        pbar = ProgressBar(steps)

        def cb(step, total):
            mm.throw_exception_if_processing_interrupted()  # honor ComfyUI's Cancel button
            pbar.update_absolute(step, total)

        imgs = krea2_pipe.generate_edit(
            prompt, _from_image_tensor(image),
            image_b=_from_image_tensor(image_b) if image_b is not None else None,
            width=width, height=height, steps=steps, seed=seed, num_images=num_images,
            step_callback=cb)
        if safety_filter:
            from .krea2_engine import safety
            imgs, _ = safety.apply(imgs, enabled=True)
        return (_to_image_tensor(imgs),)


class Krea2Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Krea2 MLX"
    OUTPUT_NODE = True

    def unload(self):
        _PIPE_CACHE.clear()
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        return ("Krea2 model unloaded; Metal cache cleared.",)


NODE_CLASS_MAPPINGS = {
    "Krea2ModelLoader": Krea2ModelLoader,
    "Krea2LoRA": Krea2LoRA,
    "Krea2Generate": Krea2Generate,
    "Krea2Img2Img": Krea2Img2Img,
    "Krea2Edit": Krea2Edit,
    "Krea2Unload": Krea2Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2ModelLoader": "Krea2 Model Loader (MLX)",
    "Krea2LoRA": "Krea2 LoRA (MLX)",
    "Krea2Generate": "Krea2 Generate (MLX)",
    "Krea2Img2Img": "Krea2 Img2Img (MLX)",
    "Krea2Edit": "Krea2 Edit (MLX)",
    "Krea2Unload": "Krea2 Unload (MLX)",
}
