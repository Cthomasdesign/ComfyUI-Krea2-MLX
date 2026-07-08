"""Sensitivity-graded quantization recipes for the Krea-2 transformer.

`class_predicate(path, module) -> False | True | {bits, group_size}` for mlx.nn.quantize.
Keeps bf16 (skips): first, last, tmlp, tproj, txtfusion(+projector), txtmlp, all norms /
modulation (non-Linear). Quantizes the 28 main blocks' attn+mlp (the ~24GB bulk).
"""

from __future__ import annotations

import re

from mlx import nn

N_LAYERS = 28


def _is_block_bulk(path: str) -> bool:
    return path.startswith("blocks.") and (".attn." in path or ".mlp." in path)


def quantize_bulk(path: str, module) -> bool:
    """Uniform-bits on the block bulk (bits set globally by nn.quantize)."""
    return isinstance(module, nn.Linear) and _is_block_bulk(path)


def mixed_4_8(path: str, module, endpoints: int = 2):
    """Mixed 4/8: down_proj @8, endpoint-block attn @8, rest of bulk @4; sensitive paths bf16."""
    if not (isinstance(module, nn.Linear) and _is_block_bulk(path)):
        return False
    n = int(re.match(r"blocks\.(\d+)\.", path).group(1))
    is_endpoint = n < endpoints or n >= N_LAYERS - endpoints
    if ".mlp.down" in path or (is_endpoint and ".attn." in path):
        return {"bits": 8, "group_size": 64}
    return {"bits": 4, "group_size": 64}
