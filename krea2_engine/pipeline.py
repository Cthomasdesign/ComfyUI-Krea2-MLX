"""End-to-end Krea-2-Turbo MLX pipeline.

Takes a local (quantized or bf16) transformer file; the Qwen-Image VAE, Qwen3-VL-4B text
encoder, and tokenizer are pulled at runtime from `krea/Krea-2-Turbo` (you accept Krea's
license there). The VAE reuses mflux's `QwenVAE` — install with `pip install mflux`.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_map

from .quant_recipes import mixed_4_8, quantize_bulk
from .sampling import sample, sample_edit, to_pil
from .text_encoder import Qwen3VLConditioner
from .transformer import Krea2Config, SingleStreamDiT

BASE_REPO = "krea/Krea-2-Turbo"
_CACHE = os.path.expanduser("~/.cache/krea2_alis_mlx")
_PROMPT_CACHE_MAX = 8  # cached prompt embeddings (a few MB each after dynamic-length encoding)


def _http_download(repo: str, filename: str, dest_root: str) -> str:
    """Download {repo}/resolve/main/{filename} over plain HTTP (the HF CDN / Xet bridge).

    We bypass huggingface_hub's downloader on purpose: its Xet path hangs with no fallback
    when cas-server.xethub.hf.co is unreachable (corporate firewalls, some ISPs), while the
    public resolve URL always works via the CDN bridge. Cached under ~/.cache/krea2_alis_mlx;
    skips the download if the local file already matches the remote size.
    """
    import requests

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = os.path.join(dest_root, filename)
    try:
        total = int(requests.head(url, allow_redirects=True, timeout=30).headers.get("content-length") or 0)
    except Exception:
        total = 0
    if os.path.exists(dest) and total and os.path.getsize(dest) == total:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0  # resume a partial download
    if total and pos == total:  # .part already holds the whole file (died before the rename) -> finalize
        os.replace(tmp, dest)    # (avoids re-requesting Range: bytes={total}- which the CDN answers 416)
        return dest
    if total and pos > total:   # stale/corrupt leftover .part (wrong or changed remote) -> restart clean
        os.remove(tmp)
        pos = 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as r:
        r.raise_for_status()  # 4xx/5xx (gated repo, missing file) surface here, not as a resume hint
        resume = bool(pos) and r.status_code == 206  # 206 => server honored the range
        pos = pos if resume else 0
        total = total or (pos + int(r.headers.get("content-length") or 0))  # prefer GET's length
        done = pos
        try:
            with open(tmp, "ab" if resume else "wb") as f:
                for chunk in r.iter_content(4 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  ↓ {filename}  {done // 1048576} / {total // 1048576} MB", end="", flush=True)
        except requests.exceptions.RequestException as e:
            # a mid-stream drop (incl. a truncated chunked transfer) lands here; .part is kept for resume
            raise OSError(f"Download of {filename} interrupted; re-run to resume "
                          "(the partial file is kept).") from e
    if total:
        print()
    # Commit only a verified-complete file. When the length is known we size-check; for a length-less
    # (chunked) transfer requests raises above on a short read, so a clean loop here means complete —
    # but never commit an empty result (an immediate disconnect that produced no bytes).
    if (total and done != total) or done == 0:
        raise OSError(f"Incomplete download of {filename}: got {done} of {total or '?'} bytes. "
                      "Re-run to resume (the partial file is kept).")
    os.replace(tmp, dest)
    return dest


def _base_dir() -> str:
    """Fetch the VAE / Qwen3-VL-4B encoder / tokenizer from krea/Krea-2-Turbo over HTTP."""
    from huggingface_hub import HfApi

    dest = os.path.join(_CACHE, BASE_REPO.replace("/", "__"))
    exts = (".safetensors", ".json", ".jinja", ".txt", ".model")
    want = [
        s.rfilename for s in HfApi().model_info(BASE_REPO).siblings
        if (s.rfilename.startswith(("vae/", "text_encoder/", "tokenizer/")) or s.rfilename == "model_index.json")
        and s.rfilename.endswith(exts)
    ]
    for f in want:
        _http_download(BASE_REPO, f, dest)
    return dest


def _load_vae(base_dir: str):
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

    class _VaeDef:
        @staticmethod
        def get_components():
            return [ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="single",
                                        mapping_getter=QwenWeightMapping.get_vae_mapping)]

        @staticmethod
        def get_download_patterns():
            return ["vae/*.safetensors", "vae/*.json"]

    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=base_dir).components["vae"])
    mx.eval(vae.parameters())
    return vae


class Krea2Pipeline:
    """precision:
    - '8bit'      : transformer_8bit.safetensors (28-block attn+mlp @ 8-bit)
    - 'mixed-4-8' : transformer_mixed_4_8.safetensors (down_proj+endpoints @8, rest @4)
    - 'bf16'      : krea/Krea-2-Turbo/turbo.safetensors (auto-downloaded)"""

    def __init__(self, transformer_path: str | None = None, precision: str = "8bit", base_dir: str | None = None):
        base = base_dir or _base_dir()
        m = SingleStreamDiT(Krea2Config())
        if precision == "8bit":
            nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)
            m.load_weights(transformer_path, strict=True)
        elif precision == "mixed-4-8":
            nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
            m.load_weights(transformer_path, strict=True)
        elif precision == "bf16":
            if transformer_path is None:
                from huggingface_hub import hf_hub_download
                transformer_path = hf_hub_download(BASE_REPO, "turbo.safetensors")
            m.load_weights(transformer_path, strict=True)
            m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
        else:
            raise ValueError(f"precision must be '8bit', 'mixed-4-8' or 'bf16', got {precision}")
        mx.eval(m.parameters())
        self.transformer = m
        self.vae = _load_vae(base)
        self.encoder = Qwen3VLConditioner(base, dtype=mx.bfloat16)
        self._lora_sig: tuple = ()   # currently-applied (path, scale) set, to skip redundant rebuilds
        self._lora_paths: list = []  # wrapped target paths, for clean unload
        self._prompt_cache: "OrderedDict" = OrderedDict()  # prompt -> (ctx, mask), LRU

    def _encode_cached(self, prompts):
        """Encoder wrapper: cache embeddings per unique prompt (LRU) and assemble batches from
        cached rows — repeat generations with the same prompt (seed sweeps) skip the encoder
        entirely, and num_images>1 encodes its prompt once instead of once per image.
        Bit-exact: cache hits return the same arrays a fresh encode produced. LoRA-safe: LoRAs
        wrap only the DiT, never the encoder. KREA2_EXACT_LEGACY=1 bypasses the cache."""
        if os.environ.get("KREA2_EXACT_LEGACY"):
            return self.encoder(prompts)
        missing = [p for p in dict.fromkeys(prompts) if p not in self._prompt_cache]
        if missing:
            ctx, mask = self.encoder(missing)
            mx.eval(ctx, mask)
            for i, p in enumerate(missing):
                self._prompt_cache[p] = (ctx[i : i + 1], mask[i : i + 1])
        for p in dict.fromkeys(prompts):
            self._prompt_cache.move_to_end(p)
        while len(self._prompt_cache) > _PROMPT_CACHE_MAX:
            self._prompt_cache.popitem(last=False)
        rows = [self._prompt_cache[p] for p in prompts]
        if len(rows) == 1:
            return rows[0]
        # rows cached at different times can differ in length; right-pad with mask-0 columns
        # (equivalent to interior padding: text RoPE positions are all zero and padding is
        # masked/trimmed downstream)
        lmax = max(c.shape[1] for c, _ in rows)
        def _pad(c, m):
            d = lmax - c.shape[1]
            if d == 0:
                return c, m
            return mx.pad(c, ((0, 0), (0, d), (0, 0), (0, 0))), mx.pad(m, ((0, 0), (0, d)))
        padded = [_pad(c, m) for c, m in rows]
        return (mx.concatenate([c for c, _ in padded], axis=0),
                mx.concatenate([m for _, m in padded], axis=0))

    def set_loras(self, specs) -> None:
        """Apply a set of LoRAs (list of (path, scale)) to the transformer, replacing any
        previously-applied set. Empty/None clears all LoRAs. Cheap no-op if unchanged."""
        specs = [(str(p), float(s)) for p, s in (specs or [])]
        sig = tuple(specs)
        if sig == self._lora_sig:
            return
        from .lora import apply_loras, unload_loras
        unload_loras(self.transformer, self._lora_paths)
        self._lora_paths = apply_loras(self.transformer, specs) if specs else []
        self._lora_sig = sig

    @staticmethod
    def _validate(prompt, width, height, steps, num_images, seed):
        """Shared prompt/dimension/range checks; returns the coerced ints."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        try:  # uniform ValueError for None / non-numeric / inf / nan (not TypeError / OverflowError)
            width, height, steps, num_images, seed = (
                int(width), int(height), int(steps), int(num_images), int(seed))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("width, height, steps, num_images, seed must be integers.") from None
        # the VAE downsamples ×8 and the DiT patchifies ×2 → dims must be multiples of 16
        for name, v in (("width", width), ("height", height)):
            if v < 256 or v > 2048 or v % 16:
                raise ValueError(f"{name} must be a multiple of 16 in [256, 2048], got {v}.")
        if not 1 <= steps <= 50:
            raise ValueError(f"steps must be in [1, 50], got {steps}.")
        if not 1 <= num_images <= 8:
            raise ValueError(f"num_images must be in [1, 8], got {num_images}.")
        if not 0 <= seed < 2**64:  # mx.random.seed wants a non-negative uint64 (else a bare TypeError)
            raise ValueError(f"seed must be in [0, 2^64), got {seed}.")
        return width, height, steps, num_images, seed

    def _encode_image(self, image, width, height, num_images):
        """PIL source image -> clean VAE latent (n,16,H/8,W/8) matching the sampler's latent space.
        The encoder wants pixels in [-1,1] (decode's *0.5+0.5 inverse); verified round-trip 45 dB."""
        image = image.convert("RGB").resize((width, height))
        arr = np.asarray(image, dtype=np.float32) / 255.0            # (H,W,3) in 0..1
        arr = np.transpose(arr, (2, 0, 1))[None] * 2.0 - 1.0          # (1,3,H,W) in -1..1
        z = self.vae.encode(mx.array(arr))                            # (1,16,1,H/8,W/8)
        if z.ndim == 5:
            z = z[:, :, 0]                                            # drop the T=1 axis
        if num_images > 1:
            z = mx.broadcast_to(z, (num_images, *z.shape[1:]))
        return z

    def generate(self, prompt, *, width=1024, height=1024, steps=8, seed=0, num_images=1, step_callback=None):
        width, height, steps, num_images, seed = self._validate(
            prompt, width, height, steps, num_images, seed)
        dec = sample(self.transformer, self.vae, self._encode_cached, [prompt] * num_images,
                     width=width, height=height, steps=steps, guidance=0.0, seed=seed,
                     step_callback=step_callback)
        return to_pil(dec)

    def generate_img2img(self, prompt, image, *, denoise=0.6, width=1024, height=1024, steps=8,
                         seed=0, num_images=1, step_callback=None):
        """Image-to-image: VAE-encode `image` (a PIL image) and start denoising from a noised
        version of it. denoise=1.0 is exactly txt2img (source ignored); lower keeps more source."""
        width, height, steps, num_images, seed = self._validate(
            prompt, width, height, steps, num_images, seed)
        try:
            denoise = float(denoise)
        except (TypeError, ValueError):
            raise ValueError("denoise must be a number in [0, 1].") from None
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0, 1], got {denoise}.")
        init_latent = self._encode_image(image, width, height, num_images)
        dec = sample(self.transformer, self.vae, self._encode_cached, [prompt] * num_images,
                     width=width, height=height, steps=steps, guidance=0.0, seed=seed,
                     init_latent=init_latent, strength=denoise, step_callback=step_callback)
        return to_pil(dec)

    def generate_edit(self, prompt, image, *, image_b=None, grounding_px=0, width=1024, height=1024,
                      steps=8, seed=0, num_images=1, step_callback=None):
        """In-context edit: keep `image` (and optional `image_b`) as clean reference frames while
        generating a fresh target guided by `prompt`. Designed for the krea2_edit identity LoRA
        (stack it via the LoRA node). grounding_px>0 also feeds the source through the Qwen3-VL
        vision tower so the instruction is read *while looking at the image* (training-matched
        semantic path); 0 keeps plain-text conditioning. PIL images in; edited images out."""
        width, height, steps, num_images, seed = self._validate(
            prompt, width, height, steps, num_images, seed)
        srcs = [self._encode_image(image, width, height, num_images)]
        if image_b is not None:
            srcs.append(self._encode_image(image_b, width, height, num_images))

        gpx = int(grounding_px or 0)
        if gpx > 0:
            def enc(prompts):
                ctx, mask = self.encoder.encode_grounded(prompts[0], image, grounding_px=gpx)
                if len(prompts) > 1:
                    ctx = mx.broadcast_to(ctx, (len(prompts), *ctx.shape[1:]))
                    mask = mx.broadcast_to(mask, (len(prompts), *mask.shape[1:]))
                return ctx, mask
        else:
            enc = self._encode_cached

        dec = sample_edit(self.transformer, self.vae, enc, [prompt] * num_images,
                          srcs, width=width, height=height, steps=steps, seed=seed,
                          step_callback=step_callback)
        return to_pil(dec)
