"""LoRA support for the Krea-2-Turbo MLX transformer.

Krea-2 LoRAs are trained on **Krea-2-Raw** and (per Krea's guidance) express on **Turbo**, so
they apply directly to the quantized Turbo weights this pipeline already loads. A LoRA
`.safetensors` is framework-neutral — MLX reads it as-is, no conversion needed.

We add each adapter as a *parallel low-rank branch* rather than merging it into the base
weights, so it works on a quantized base untouched:

    y = base(x) + Σ_i  scale_i · (x · A_iᵀ) · B_iᵀ

`base` may be a plain `nn.Linear` or an `nn.QuantizedLinear` — we only ever *call* it, so the
quantization is irrelevant. Multiple LoRAs stack on the same layer (the sum above). The branch
is exact, toggleable, and reversible (`unload_loras` restores the original layers).

Key mapping (`_to_mlx_path`): the mapper strips a wrapper prefix (`base_model.model.` /
`transformer.` / …) and rewrites the two Krea-2 LoRA naming conventions onto the module tree —
the official Krea / diffusers names (`transformer_blocks.N.attn.to_q`, `img_in`, `final_layer`,
`ff.gate`, as in `krea/Krea-2-LoRA-*`) and the ai-toolkit / PEFT names that already match
(`blocks.N.attn.wq`, `mlp.down`, `first`, `last.linear`, `txtfusion...`). `A` is (rank, in),
`B` is (out, rank). Effective scale folds in alpha/rank from the file metadata
(`lora_alpha`/`lora_rank`), matching diffusers/PEFT conventions. Unmapped keys raise.
"""

from __future__ import annotations

import json
import struct

import mlx.core as mx
from mlx import nn

# LoRA wrapper prefixes (PEFT / ComfyUI / kohya) stripped before mapping to the module tree.
_PREFIXES = ("base_model.model.", "transformer.", "diffusion_model.", "lora_unet_")

# Krea-2 LoRAs ship in two naming conventions: the reference / ai-toolkit names that already match
# this port's module tree (e.g. `blocks.N.attn.wq`), and the official Krea / diffusers names
# (e.g. `transformer_blocks.N.attn.to_q`, as in `krea/Krea-2-LoRA-*` and the Comfy-Org repack).
# These ordered replacements rewrite the diffusers names to the module tree; they only match
# diffusers tokens, so reference-named keys pass through unchanged.
_TRANSLATE = (
    ("transformer_blocks", "blocks"),
    ("text_fusion", "txtfusion"),
    ("final_layer.linear", "last.linear"),
    ("img_in", "first"),
    ("time_embed.linear_1", "tmlp.0"),
    ("time_embed.linear_2", "tmlp.2"),
    ("time_mod_proj", "tproj.1"),
    ("txt_in.linear_1", "txtmlp.1"),
    ("txt_in.linear_2", "txtmlp.3"),
    ("attn.to_out.0", "attn.wo"),
    ("attn.to_q", "attn.wq"),
    ("attn.to_k", "attn.wk"),
    ("attn.to_v", "attn.wv"),
    ("attn.to_gate", "attn.gate"),
    ("ff.gate", "mlp.gate"),
    ("ff.up", "mlp.up"),
    ("ff.down", "mlp.down"),
)


def _to_mlx_path(path: str) -> str:
    """Map a LoRA's module path onto this port's module tree: strip a wrapper prefix, then rewrite
    official-Krea/diffusers layer names to the reference names (a no-op for reference-named keys)."""
    for pre in _PREFIXES:
        if path.startswith(pre):
            path = path[len(pre):]
            break
    for old, new in _TRANSLATE:
        path = path.replace(old, new)
    return path


class LoRALinear(nn.Module):
    """Wraps a (quantized or plain) linear layer and adds one or more low-rank branches.

    Holds the original layer in `self.base` (kept frozen, quantization intact) plus parallel
    lists of adapter matrices. Stacking = more than one (a, b, scale) triple.
    """

    def __init__(self, base: nn.Module, a_list, b_list, scales):
        super().__init__()
        self.base = base
        self.lora_a = list(a_list)   # each (rank, in)
        self.lora_b = list(b_list)   # each (out, rank)
        self.scales = list(scales)   # python floats (not parameters)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        for a, b, s in zip(self.lora_a, self.lora_b, self.scales):
            delta = (x @ a.T) @ b.T
            y = y + (s * delta).astype(y.dtype)
        return y


# --- module-tree navigation -------------------------------------------------
# Paths use '.' separators; integer parts index plain Python lists (e.g. `blocks`, `tmlp`),
# string parts are attributes. In-place list mutation and setattr are both picked up by MLX,
# which re-reads __dict__ on every forward / parameters() call.

def _resolve_parent(root, path: str):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _get_submodule(root, path: str):
    parent, last = _resolve_parent(root, path)
    return parent[int(last)] if last.isdigit() else getattr(parent, last)


def _set_submodule(root, path: str, value) -> None:
    parent, last = _resolve_parent(root, path)
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


# --- file parsing -----------------------------------------------------------

def _read_metadata(path: str) -> dict:
    """Read the `__metadata__` block from a safetensors header (mx.load drops it)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__", {}) or {}


def _normalize(weights: dict) -> dict:
    """Pair lora_A/lora_B tensors by target module path.

    Returns {target_path: (A, B)}. Accepts the PEFT `lora_A`/`lora_B` and the kohya
    `lora_down`/`lora_up` conventions; raises on anything it can't pair.
    """
    out: dict[str, tuple] = {}
    a_keys = {}
    b_keys = {}
    for k in weights:
        if k.endswith(".alpha"):
            continue  # per-layer alpha handled via file metadata below
        for suf, bucket in ((".lora_A.weight", a_keys), (".lora_down.weight", a_keys),
                            (".lora_B.weight", b_keys), (".lora_up.weight", b_keys)):
            if k.endswith(suf):
                bucket[_to_mlx_path(k[: -len(suf)])] = k
                break
        else:
            raise ValueError(f"Unrecognized LoRA key (no lora_A/B/down/up suffix): {k}")
    missing = set(a_keys) ^ set(b_keys)
    if missing:
        raise ValueError(f"LoRA has unpaired A/B tensors for: {sorted(missing)[:5]} …")
    for tgt in a_keys:
        out[tgt] = (weights[a_keys[tgt]], weights[b_keys[tgt]])
    return out


def _alpha_factor(meta: dict) -> float:
    """Scale multiplier alpha/rank (diffusers/PEFT convention); 1.0 if unspecified."""
    try:
        alpha = float(meta.get("lora_alpha", meta.get("alpha", 0)))
        rank = float(meta.get("lora_rank", meta.get("r", 0)))
        if alpha and rank:
            return alpha / rank
    except (TypeError, ValueError):
        pass
    return 1.0


# --- public API -------------------------------------------------------------

def unload_loras(model, paths) -> int:
    """Restore the given wrapped layer paths to their original base layers.

    `paths` is the list of target paths returned by `apply_loras` (MLX's `nn.Module` is a dict
    subclass, so we unwrap by known path via getattr/setattr rather than walking the tree).
    Returns how many layers were restored.
    """
    removed = 0
    for tgt in paths or []:
        try:
            cur = _get_submodule(model, tgt)
        except (AttributeError, IndexError, KeyError, ValueError):
            continue
        while isinstance(cur, LoRALinear):  # defensively peel any accidental nesting
            cur = cur.base
            removed += 1
        _set_submodule(model, tgt, cur)
    if removed:
        mx.eval(model.parameters())
    return removed


def apply_loras(model, specs, *, dtype=mx.bfloat16) -> list[str]:
    """Apply LoRAs to `model` in place. `specs` is a list of (path, scale); multiple LoRAs
    targeting the same layer stack. Returns the list of wrapped target paths (pass to
    `unload_loras` to revert). Raises if a target path can't be resolved, naming it."""
    # gather: target_path -> ([A...], [B...], [scale...])
    grouped: dict[str, tuple[list, list, list]] = {}
    for path, scale in specs:
        weights = mx.load(path)
        factor = _alpha_factor(_read_metadata(path))
        adapters = _normalize(weights)
        eff = float(scale) * factor
        for tgt, (a, b) in adapters.items():
            slot = grouped.setdefault(tgt, ([], [], []))
            slot[0].append(a.astype(dtype))
            slot[1].append(b.astype(dtype))
            slot[2].append(eff)

    wrapped = []
    for tgt, (a_list, b_list, scales) in grouped.items():
        try:
            base = _get_submodule(model, tgt)
        except (AttributeError, IndexError, KeyError, ValueError) as e:
            raise ValueError(f"LoRA targets '{tgt}' which doesn't exist in the model tree") from e
        if isinstance(base, LoRALinear):  # never nest — wrap the true base
            base = base.base
        if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(f"LoRA target '{tgt}' is {type(base).__name__}, not a linear layer")
        # sanity: A is (rank, in_features); fail loudly on an obvious shape mismatch
        in_feats = a_list[0].shape[1]
        for a, b in zip(a_list, b_list):
            if a.shape[1] != in_feats or a.shape[0] != b.shape[1]:
                raise ValueError(f"LoRA shape mismatch at '{tgt}': A{a.shape} B{b.shape}")
        _set_submodule(model, tgt, LoRALinear(base, a_list, b_list, scales))
        wrapped.append(tgt)

    mx.eval(model.parameters())
    return wrapped
