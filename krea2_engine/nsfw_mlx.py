"""Pure-MLX NSFW image classifier (Falconsai/nsfw_image_detection, a ViT-base).

The upstream model is a `ViTForImageClassification`; running it through transformers'
`pipeline("image-classification")` would pull in **PyTorch** (transformers 5.x has no other
backend). To keep this release torch-free and the safety filter on-by-default on a clean
`pip install -r requirements.txt`, we reimplement the ViT in MLX and load the HF safetensors
directly. Validated numerically against the PyTorch reference (see validation/validate_nsfw.py).

ViT-base/16-224: 12 layers, width 768, 12 heads, MLP 3072 (gelu), patch 16, image 224²,
2 classes {0: normal, 1: nsfw}. Preprocess: resize 224² (bilinear), /255, normalize 0.5/0.5.
"""

from __future__ import annotations

import glob

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

H = 768          # hidden size
LAYERS = 12
HEADS = 12
HEAD_DIM = H // HEADS
MLP = 3072
PATCH = 16
IMG = 224
NPATCH = (IMG // PATCH) ** 2   # 196
SEQ = NPATCH + 1               # + CLS token


class _Block(nn.Module):
    """Pre-LN ViT encoder block (HF naming: layernorm_before/after, attention.*, intermediate, output)."""

    def __init__(self):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(H)
        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        self.attn_out = nn.Linear(H, H)
        self.layernorm_after = nn.LayerNorm(H)
        self.fc1 = nn.Linear(H, MLP)
        self.fc2 = nn.Linear(MLP, H)

    def __call__(self, x):
        b, n, _ = x.shape
        h = self.layernorm_before(x)
        q = self.q(h).reshape(b, n, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(b, n, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(b, n, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5)
        a = a.transpose(0, 2, 1, 3).reshape(b, n, H)
        x = x + self.attn_out(a)
        h = self.layernorm_after(x)
        x = x + self.fc2(nn.gelu(self.fc1(h)))
        return x


class ViTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, H, kernel_size=PATCH, stride=PATCH)   # patch embedding
        self.cls = mx.zeros((1, 1, H))
        self.pos = mx.zeros((1, SEQ, H))
        self.blocks = [_Block() for _ in range(LAYERS)]
        self.ln = nn.LayerNorm(H)
        self.head = nn.Linear(H, 2)

    def __call__(self, pixels):  # pixels: (b, IMG, IMG, 3) NHWC, already normalized
        x = self.proj(pixels)                          # (b, 14, 14, 768)
        b = x.shape[0]
        x = x.reshape(b, NPATCH, H)
        x = mx.concatenate([mx.broadcast_to(self.cls, (b, 1, H)), x], axis=1)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x[:, 0])                      # logits over {normal, nsfw} from the CLS token


# HF ViT safetensors key -> our module path
def _remap(hf: dict) -> dict:
    out = {}
    for k, v in hf.items():
        nk = k
        nk = nk.replace("vit.embeddings.cls_token", "cls")
        nk = nk.replace("vit.embeddings.position_embeddings", "pos")
        nk = nk.replace("vit.embeddings.patch_embeddings.projection", "proj")
        nk = nk.replace("vit.layernorm.", "ln.")
        nk = nk.replace("classifier.", "head.")
        if nk.startswith("vit.encoder.layer."):
            i = nk.split(".")[3]
            rest = nk.split(f"vit.encoder.layer.{i}.")[1]
            rest = (rest
                    .replace("attention.attention.query", "q")
                    .replace("attention.attention.key", "k")
                    .replace("attention.attention.value", "v")
                    .replace("attention.output.dense", "attn_out")
                    .replace("intermediate.dense", "fc1")
                    .replace("output.dense", "fc2"))
            nk = f"blocks.{i}.{rest}"
        # MLX Conv2d weight is (out, kH, kW, in); HF Conv2d is (out, in, kH, kW)
        if nk == "proj.weight":
            v = v.transpose(0, 2, 3, 1)
        out[nk] = v
    return out


def load_classifier(model_dir: str) -> ViTClassifier:
    shards = sorted(glob.glob(f"{model_dir}/*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors under {model_dir}")
    raw = {}
    for s in shards:
        raw.update(mx.load(s))
    weights = _remap(raw)
    model = ViTClassifier()
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = sorted(expected - set(weights)), sorted(set(weights) - expected)
    if missing or extra:
        raise RuntimeError(f"NSFW classifier weight mismatch missing={missing[:4]} extra={extra[:4]}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(image) -> mx.array:
    """PIL image -> (1, 224, 224, 3) normalized NHWC, matching ViTImageProcessor."""
    im = image.convert("RGB").resize((IMG, IMG), resample=2)  # 2 = bilinear (PIL.Image.BILINEAR)
    arr = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return mx.array(arr[None])  # NHWC, batch 1


def nsfw_score(model: ViTClassifier, image) -> float:
    logits = model(preprocess(image))
    probs = mx.softmax(logits, axis=-1)
    mx.eval(probs)
    return float(probs[0, 1])  # P(nsfw); label index 1
