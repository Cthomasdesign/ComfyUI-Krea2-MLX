# ComfyUI-Krea2-MLX

Run **Krea-2-Turbo** text-to-image locally on **Apple Silicon** inside ComfyUI, using Apple's
**MLX** framework — fast and memory-efficient (quantized to fit a 32 GB Mac). Supports
**stackable LoRAs** (official Krea / diffusers *and* ai-toolkit / PEFT formats).

The pure-MLX pipeline (`krea2_engine/`) is vendored directly in this repo — no external git
dependency — and runs **in-process**, returning standard ComfyUI `IMAGE` tensors. It's a
self-contained node group: load → (LoRAs) → generate → IMAGE, then use any ComfyUI node downstream
(save, upscale, etc.). It does **not** expose a torch `MODEL`, so it does not plug into ComfyUI's
native KSampler/ControlNet.

> **macOS on Apple Silicon (M-series) only.** The MLX dependencies do not run on Windows/Linux.
> ~24 GB unified memory recommended for 1024².

## Install

**Via ComfyUI-Manager** (recommended): search for *Krea2-MLX* and install; Manager runs
`requirements.txt`, which installs the vendored pipeline's MLX deps into ComfyUI's Python.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cthomasdesign/ComfyUI-Krea2-MLX
<ComfyUI-python> -m pip install -r ComfyUI-Krea2-MLX/requirements.txt
```
Restart ComfyUI.

## Models

Put a transformer build in **`ComfyUI/models/krea2/`** — the **Krea2 Model Loader** lists what's
there and infers precision from the filename (`mixed_4_8` / `8bit` / `bf16`):

| Build | Size | Repo → file |
|---|---|---|
| mixed-4/8 | 9.8 GB | [avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8) → `transformer_mixed_4_8.safetensors` |
| 8-bit | 14 GB | [avlp12/Krea-2-Turbo-Alis-MLX-8bit](https://huggingface.co/avlp12/Krea-2-Turbo-Alis-MLX-8bit) → `transformer_8bit.safetensors` |

On first generation the **VAE + Qwen3-VL-4B text encoder** (~9 GB) download automatically from
`krea/Krea-2-Turbo` (cached in `~/.cache/krea2_alis_mlx`). Put **LoRAs** in `ComfyUI/models/loras/`.

## Nodes (category **Krea2 MLX**)

| Node | Purpose |
|---|---|
| **Krea2 Model Loader (MLX)** | Pick a build from `models/krea2` → `KREA2_PIPE` (cached; stays resident). |
| **Krea2 LoRA (MLX)** | Pick a LoRA from `models/loras` + strength → `KREA2_LORASTACK`. Chain several to stack. |
| **Krea2 Generate (MLX)** | `KREA2_PIPE` + prompt/size/steps/seed (+ optional LoRA stack) → `IMAGE`. Progress bar + Cancel supported; NSFW `safety_filter` on by default. |
| **Krea2 Unload (MLX)** | Frees the cached model and clears the Metal cache to reclaim memory. |

**Minimal workflow:** `Krea2 Model Loader → Krea2 Generate → Preview Image`.
**With LoRAs:** `Krea2 LoRA → (Krea2 LoRA → …) → Krea2 Generate`. See `example_workflows/`.

## Notes

- One 12.9B model (~18–20 GB with encoder/VAE) stays resident. On 32 GB, avoid loading large torch
  models simultaneously; use **Krea2 Unload** to free memory.
- The `safety_filter` runs the pure-MLX `Falconsai/nsfw_image_detection` classifier; the Krea 2
  Community License (§4.2) requires reasonable content-filtering in deployments — keep it on there.
- Where required by law, disclose that outputs are AI-generated (License §4.3).

## License & attribution

Independent, unofficial — **not** an official Krea product or endorsed by Krea. The model is a
Derivative of [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) under the **Krea 2
Community License** (see [`LICENSE`](LICENSE)); the MLX port is by **avlp12**
([Alis MLX](https://github.com/avlp12/krea2_alis_mlx)), with LoRA support added in
[krea2-mlx-studio](https://github.com/Cthomasdesign/krea2-mlx-studio). The engine (`krea2_engine/`)
is vendored here from that lineage — see [`NOTICE`](NOTICE) for the full attribution chain and
what changed at each step. Model weights are downloaded by the user, not redistributed here.
Node/engine code here: MIT (per upstream). NSFW classifier:
[Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)
(Apache-2.0). By using the weights you agree to the Krea 2 Community License (commercial use requires
< $1M annual revenue, or an enterprise license).
