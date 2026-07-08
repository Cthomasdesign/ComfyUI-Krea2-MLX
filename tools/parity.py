"""Output-parity harness: proves optimizations don't change generated images.

    python tools/parity.py --capture --model ... [--lora path.safetensors]   # snapshot references
    python tools/parity.py --check   --model ... [--lora path.safetensors]   # compare against them
    add --quick for the 2-case subset (per-commit smoke check)

Captures the decoded float32 pixels (pre-uint8) per case to tools/reference/. --check re-runs
and reports, per case: bit-exact float equality, uint8 pixel equality (the quality gate), and
max|Δ| / PSNR as diagnostics. Exit code 1 if any case's uint8 pixels differ.
"""

import argparse
import os
import sys

import _shared  # noqa: F401  (sys.path bootstrap)
import mlx.core as mx
import numpy as np

from _shared import PROMPT_LONG, PROMPT_SHORT, load_pipeline
from krea2_engine.sampling import sample

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")


def cases(lora_path, quick):
    """(case_id, prompt, seed, size, lora) matrix."""
    full = [
        ("short_s0_512", PROMPT_SHORT, 0, 512, None),
        ("short_s42_512", PROMPT_SHORT, 42, 512, None),
        ("long_s0_512", PROMPT_LONG, 0, 512, None),
        ("short_s0_1024", PROMPT_SHORT, 0, 1024, None),
    ]
    if lora_path:
        full.append(("short_s0_512_lora", PROMPT_SHORT, 0, 512, lora_path))
    if quick:
        keep = {"short_s0_512", "short_s0_512_lora"}
        full = [c for c in full if c[0] in keep]
    return full


def run_case(pipe, prompt, seed, size, lora):
    pipe.set_loras([(lora, 0.9)] if lora else [])
    decoded = sample(pipe.transformer, pipe.vae, pipe.encoder, [prompt],
                     width=size, height=size, steps=8, guidance=0.0, seed=seed)
    return np.array(decoded.astype(mx.float32))  # (1,3,H,W) in 0..1


def to_uint8(arr):
    return (np.transpose(arr, (0, 2, 3, 1)) * 255.0).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(REF_DIR, exist_ok=True)
    pipe, load_s = load_pipeline(args.model)
    print(f"model load: {load_s:.1f}s")

    failed = []
    for case_id, prompt, seed, size, lora in cases(args.lora, args.quick):
        ref_path = os.path.join(REF_DIR, f"{case_id}.npy")
        arr = run_case(pipe, prompt, seed, size, lora)
        if args.capture:
            np.save(ref_path, arr)
            print(f"{case_id:>20}: captured {arr.shape}")
            continue

        if not os.path.exists(ref_path):
            print(f"{case_id:>20}: NO REFERENCE (run --capture on the baseline first)")
            failed.append(case_id)
            continue
        ref = np.load(ref_path)
        bit_exact = np.array_equal(ref, arr)
        u8_ref, u8_new = to_uint8(ref), to_uint8(arr)
        u8_equal = np.array_equal(u8_ref, u8_new)
        max_d = float(np.abs(ref - arr).max())
        if u8_equal:
            print(f"{case_id:>20}: PASS  float_bit_exact={bit_exact}  max|d|={max_d:.2e}")
        else:
            mse = float(((ref - arr) ** 2).mean())
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            npx = int((u8_ref != u8_new).any(axis=-1).sum())
            print(f"{case_id:>20}: FAIL  max|d|={max_d:.2e}  psnr={psnr:.1f}dB  "
                  f"changed_px={npx}")
            failed.append(case_id)

    if args.check:
        if failed:
            print(f"\nPARITY FAILED: {failed}")
            sys.exit(1)
        print("\nPARITY OK: all cases uint8-identical")


if __name__ == "__main__":
    main()
