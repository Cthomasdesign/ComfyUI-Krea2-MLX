"""VisionConfig for the vendored Qwen3-VL vision tower (from mlx-vlm, MIT — see ../../NOTICE)."""

import inspect
from dataclasses import dataclass, field


class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(**{k: v for k, v in params.items()
                      if k in inspect.signature(cls).parameters})


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_vl"
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    out_hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    tokens_per_second: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    window_size: int = 112
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])
    deepstack_visual_indexes: list[int] = field(default_factory=list)
