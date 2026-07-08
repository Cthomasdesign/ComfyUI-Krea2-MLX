"""Stage-attributed generation benchmark for the krea2_engine pipeline.

Times model load, text encode, each denoise step, and VAE decode, then prints a table and
writes JSON (tools/reference/bench_<label>.json) for before/after comparison.

    python tools/bench.py --model /path/to/transformer_mixed_4_8.safetensors --label baseline
    python tools/bench.py --model ... --label opt1 --quick     # 512² short prompt only

Per-step timing uses the pipeline's own step_callback (fires after each step's mx.eval), so it
measures exactly what ComfyUI sees. The first timed configuration run is reported separately as
warmup (Metal kernel compilation).
"""

import argparse
import json
import os
import time

import _shared  # noqa: F401  (sys.path bootstrap)
import mlx.core as mx

from _shared import PROMPT_LONG, PROMPT_SHORT, load_pipeline
from krea2_engine.sampling import sample


class _TimedEncode:
    """Wrap the conditioner so encode time is attributable (forces eval of its outputs)."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.seconds = None

    def __call__(self, prompts):
        t0 = time.perf_counter()
        ctx, mask = self.encoder(prompts)
        mx.eval(ctx, mask)
        self.seconds = time.perf_counter() - t0
        return ctx, mask


def run_case(pipe, prompt, size, steps=8, seed=0):
    enc = _TimedEncode(pipe.encoder)
    marks = []

    def cb(step, total):
        marks.append(time.perf_counter())

    t0 = time.perf_counter()
    decoded = sample(pipe.transformer, pipe.vae, enc, [prompt],
                     width=size, height=size, steps=steps, guidance=0.0, seed=seed,
                     step_callback=cb)
    mx.eval(decoded)
    total = time.perf_counter() - t0

    # attribute: encode | step1 (includes any per-run setup) | steps 2..N | decode+overhead
    step_walls = [marks[0] - t0 - enc.seconds] + [b - a for a, b in zip(marks, marks[1:])]
    rest = sorted(step_walls[1:])
    return {
        "encode_s": round(enc.seconds, 3),
        "step1_s": round(step_walls[0], 3),
        "step_median_s": round(rest[len(rest) // 2], 3) if rest else None,
        "steps_total_s": round(sum(step_walls), 3),
        "decode_s": round(total - enc.seconds - sum(step_walls), 3),
        "total_s": round(total, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="name for the JSON result file")
    ap.add_argument("--quick", action="store_true", help="512² short prompt only")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    configs = [("short", PROMPT_SHORT, 512)]
    if not args.quick:
        configs += [("long", PROMPT_LONG, 512), ("short", PROMPT_SHORT, 1024)]

    pipe, load_s = load_pipeline(args.model)
    print(f"model load: {load_s:.1f}s")

    results = {"label": args.label, "model": os.path.basename(args.model),
               "steps": args.steps, "load_s": round(load_s, 1), "runs": []}
    for i, (name, prompt, size) in enumerate(configs):
        for phase in (["warmup", "timed"] if i == 0 else ["timed"]):
            r = run_case(pipe, prompt, size, steps=args.steps)
            r.update(config=f"{name}-{size}", phase=phase)
            results["runs"].append(r)
            print(f"{r['config']:>11} {phase:>6}: encode {r['encode_s']:6.2f}s | "
                  f"step1 {r['step1_s']:6.2f}s | step~ {r['step_median_s']:6.2f}s | "
                  f"decode {r['decode_s']:6.2f}s | total {r['total_s']:7.2f}s")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"bench_{args.label}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
