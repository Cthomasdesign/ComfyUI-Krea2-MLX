# ComfyUI-Krea2-MLX

Run **Krea-2-Turbo** locally on **Apple Silicon** inside ComfyUI, using Apple's **MLX**
framework — text-to-image, img2img, and **identity-preserving image editing** with dual
conditioning (in-context reference frames + Qwen3-VL vision grounding). Quantized to fit a
32 GB Mac; supports **stackable LoRAs** (official Krea / diffusers *and* ai-toolkit / PEFT
formats, including the `krea2_edit` identity LoRA).

> **macOS on Apple Silicon (M-series) only.** The MLX dependencies do not run on Windows/Linux.
> ~24 GB unified memory recommended for 1024²; editing with grounding wants ~32 GB.

## Architecture

The pure-MLX pipeline (`krea2_engine/`) is vendored directly in this repo — no external git
dependency — and runs **in-process**, returning standard ComfyUI `IMAGE` tensors:

```
Krea2 Model Loader ──KREA2_PIPE──┬── Krea2 Generate ──┐
Krea2 LoRA ──KREA2_LORASTACK─────┼── Krea2 Img2Img ───┼──IMAGE──► any ComfyUI node
LoadImage ──IMAGE────────────────┴── Krea2 Edit ──────┘   (save, upscale, …)
```

Inside the pipe, everything is MLX on Metal: the Qwen3-VL-4B text encoder (with its vision
tower for grounded editing), the 12.9B Krea-2 MMDiT transformer, the flow-matching sampler,
and the Qwen-Image VAE. **No PyTorch in the generation path.** It's a self-contained island:
it does not expose a torch `MODEL`, so it does not plug into ComfyUI's native
KSampler/ControlNet — by design.

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
`krea/Krea-2-Turbo` (cached in `~/.cache/krea2_alis_mlx`). Put **LoRAs** in `ComfyUI/models/loras/`
— e.g. [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit)
for editing.

## Nodes (category **Krea2 MLX**)

| Node | Purpose |
|---|---|
| **Krea2 Model Loader (MLX)** | Pick a build from `models/krea2` → `KREA2_PIPE` (cached; stays resident). |
| **Krea2 LoRA (MLX)** | Pick a LoRA from `models/loras` + strength → `KREA2_LORASTACK`. Chain several to stack. |
| **Krea2 Generate (MLX)** | Text→image. Prompt/size/steps/seed (+ LoRA stack) → `IMAGE`. Progress bar + Cancel supported; NSFW `safety_filter` on by default. |
| **Krea2 Img2Img (MLX)** | Source image + prompt + `denoise` → `IMAGE`. Reinterprets the **whole frame**: `denoise` 0→1 goes from near-copy to full reinterpretation (`1.0` == plain text→image). |
| **Krea2 Edit (MLX)** | Source image + instruction (+ optional 2nd reference, + `grounding_px`) → `IMAGE`. **Identity-preserving edit**: the source rides as a clean reference frame while a fresh target is generated. |
| **Krea2 Unload (MLX)** | Frees the cached model and clears the Metal cache to reclaim memory. |

Ready-made graphs in [`example_workflows/`](example_workflows/): text-to-image
(`Krea2-MLX-example.json`), img2img (`Krea2-MLX-img2img.json`), identity edit with the LoRA
chained in (`Krea2-MLX-edit.json`).

## Editing guide (Krea2 Edit)

The edit node implements the **dual conditioning** the
[krea2_edit identity LoRA](https://huggingface.co/conradlocke/krea2-identity-edit) was trained
with: the source enters (1) as clean VAE reference tokens the transformer attends to while
generating, and (2) through the Qwen3-VL **vision tower**, so the instruction is read *while
looking at the image*. Stack the identity LoRA at **strength 1.0** for best results.

- **Match the output aspect ratio to the source** — training pairs are same-size; mismatches
  degrade preservation.
- **`grounding_px`** (default 768): lower = stronger edit adherence and more uniform scene
  changes; higher = stronger identity/likeness. `0` = plain-text conditioning (no vision tower).
- **Two references** (`image_b`): training order is **scene first, subject second**. Two-person
  inputs tend to drift faces toward each other — for best identity, **chain single-reference
  edits** instead.
- **What works on Turbo** (this island): add, recolor, restyle, re-stage — at 8 steps, no CFG.
  **Large removals/deletions** were tuned for Krea-2-**Raw** at 20 steps with CFG, which this
  Turbo-only island doesn't run; expect them to be weaker.
- Img2Img vs Edit: img2img *reinterprets everything* (style transfers, variations); Edit
  *preserves the subject and applies the instruction* (targeted changes).

## Performance

The vendored engine is heavily optimized for Apple Silicon — measured on an M1 Max 32 GB
(mixed-4/8 build, 8 steps):

| | before (v0.3) | after (v0.4+) | faster |
|---|---|---|---|
| 512², short prompt | ~103s | **~67s** | ~35% |
| 512², long prompt | ~104s | **~72s** | ~31% |
| 1024² | ~310s | **~263s** | ~15% |

Repeat generations with the same prompt (seed sweeps) additionally skip the text encoder via a
per-prompt embedding cache. What changed: native grouped-query attention; step-invariant
conditioning hoisted out of the denoising loop; padded text tokens trimmed before the DiT (~30%
of the sequence at 512²); dynamic-length text encoding; prompt-embedding cache. All validated
mathematically faithful to the reference computation (see `tools/`).

**Seed reproducibility:** the sequence-length optimizations alter bf16 kernel rounding order, so
a fixed seed renders an *equal-quality but not pixel-identical* image vs v0.3 — the same class
of change as an MLX or hardware upgrade. Within a version, generation is fully deterministic
(same seed → same image, bit-for-bit).

## Environment variables

| Variable | Effect |
|---|---|
| `KREA2_EXACT_LEGACY=1` | Bit-exact pre-0.4 outputs: restores padded encoding/attention, bypasses the prompt cache (slower). For reproducing old seeds. |
| `KREA2_BASE_DIR` | Local `krea/Krea-2-Turbo` snapshot (skips the first-run download). |
| `KREA2_DISABLE_SAFETY=1` | Globally disables the NSFW filter (the license requires filtering in deployments — see Notes). |
| `KREA2_SAFETY_THRESHOLD` | NSFW flag threshold (default 0.85). |

## Development: bench & parity harness (`tools/`)

`tools/bench.py` attributes generation time to encode / per-step / decode stages;
`tools/parity.py` snapshots reference outputs and asserts later runs stay pixel-identical — the
quality gate every engine change passes through. `tools/test_*.py` validate each feature
(img2img guard, edit forward, vision tower vs the PyTorch reference, grounded encode, two-ref
grounding).

## Notes

- One 12.9B model (~18–20 GB with encoder/VAE; +~2.7 GB with the vision tower and identity LoRA)
  stays resident. On 32 GB, avoid loading large torch models simultaneously; use **Krea2 Unload**
  to free memory.
- The `safety_filter` runs the pure-MLX `Falconsai/nsfw_image_detection` classifier; the Krea 2
  Community License (§4.2) requires reasonable content-filtering in deployments — keep it on there.
- Where required by law, disclose that outputs are AI-generated (License §4.3).

## License & attribution

Independent, unofficial — **not** an official Krea product or endorsed by Krea. The model is a
Derivative of [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) under the **Krea 2
Community License** (see [`LICENSE`](LICENSE)); the MLX port is by **avlp12**
([Alis MLX](https://github.com/avlp12/krea2_alis_mlx)), with LoRA support added in
[krea2-mlx-studio](https://github.com/Cthomasdesign/krea2-mlx-studio). The engine (`krea2_engine/`)
is vendored here from that lineage; the Qwen3-VL vision tower is vendored from
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (MIT) — see [`NOTICE`](NOTICE) for the full
attribution chain and what changed at each step. Model weights are downloaded by the user, not
redistributed here. Node/engine code here: MIT (per upstream). NSFW classifier:
[Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)
(Apache-2.0). By using the weights you agree to the Krea 2 Community License (commercial use
requires < $1M annual revenue, or an enterprise license).
