#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
B1: Characterize HF processor stage costs at production image resolution.

Production spec (from sweep_397b_rates.yaml):
  - 5x 2400x1080 images
  - Model: Qwen/Qwen3.5-397B-A17B-FP8 (processor class shared with 2B)

This script measures synchronous HF processor CPU time for 5 images.
It does NOT reproduce server-side latency — it isolates the lower bound
for HF processor work, excluding executor-queue wait, GIL contention, and
MM cache state.

Run with both input types to compare PIL vs ndarray paths:
    python b1_hf_processor_timing.py --input-type pil    # vLLM serve base64 path
    python b1_hf_processor_timing.py --input-type ndarray # Dynamo --frontend-decoding path

Expected output (example):
    processor class: Qwen2VLImageProcessor  (or fast variant)
    Input type: PIL.Image (2400x1080)

    Running 5 warmup + 20 measured iterations...
      iter  1: XXX.X ms
      ...
    === Total processor call time ===
      mean    : XXX.X ms
      stdev   :   X.X ms
      min/max : XXX.X / XXX.X ms

Interpretation:
  ~250 ms -> pure HF work; consistent with vLLM serve total (hash is separate).
  ~50 ms  -> consistent with Dynamo [mmproc] total_ms on cache-miss.
             Compare to [mmcache] hf= from server logs for direct comparison.
  << 50 ms -> HF processor is not the bottleneck; look at executor queue overhead.
"""

import argparse
import io
import os
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen3-VL-2B-Instruct",
    help="HF model ID. The 2B model loads in seconds; processor code is identical to 397B.",
)
parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (discarded)")
parser.add_argument("--iters", type=int, default=20, help="Measured iterations")
parser.add_argument(
    "--input-type",
    choices=["pil", "ndarray"],
    default="pil",
    help=(
        "Type of image to pass to the processor. "
        "'pil' = PIL.Image (matches vLLM serve base64 path); "
        "'ndarray' = np.ndarray (matches Dynamo --frontend-decoding path)"
    ),
)
args = parser.parse_args()

# Enable TRANSFORMERS_MM_PROFILING so mm:hfproc:* NVTX sub-ranges fire.
# With nsys attached these appear in the trace; without nsys they are no-ops.
os.environ.setdefault("TRANSFORMERS_MM_PROFILING", "1")

# ---------------------------------------------------------------------------
# Load processor
# ---------------------------------------------------------------------------
from transformers import AutoProcessor  # noqa: E402

print(f"Loading processor for {args.model} ...")
t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(args.model)
print(f"Loaded in {(time.perf_counter() - t0)*1000:.0f} ms")
# Print processor class so we know if we're on the fast or slow path.
# Fast path (Qwen2VLImageProcessorFast) has mm:hfproc:* NVTX sub-ranges;
# slow path (Qwen2VLImageProcessor) has fewer built-in timing hooks.
img_proc_class = type(processor.image_processor).__name__
print(f"processor class: {img_proc_class}")

# ---------------------------------------------------------------------------
# Build synthetic images at production resolution (2400x1080)
# ---------------------------------------------------------------------------
# Use solid-color PNGs in memory — no disk I/O, no network.
WIDTH, HEIGHT = 2400, 1080
NUM_IMAGES = 5

# Create as PIL Images first (always needed; ndarray is derived from these)
_pil_images: list[Image.Image] = []
for i in range(NUM_IMAGES):
    # Vary color per image so they are not byte-identical (avoid cache collapse)
    color = (i * 50 % 256, (i * 80 + 40) % 256, (i * 30 + 20) % 256)
    img = Image.new("RGB", (WIDTH, HEIGHT), color=color)
    _pil_images.append(img)

# Build ndarray versions (mimics what Dynamo --frontend-decoding delivers)
_ndarray_images: list[np.ndarray] = [np.asarray(img) for img in _pil_images]

# Choose input type
if args.input_type == "pil":
    images_for_processor: list[Any] = _pil_images
    print(f"Input type: PIL.Image ({WIDTH}x{HEIGHT})")
else:
    images_for_processor = _ndarray_images
    print(f"Input type: np.ndarray {_ndarray_images[0].shape} dtype={_ndarray_images[0].dtype}")

# ---------------------------------------------------------------------------
# Optional: monkey-patch HF processor stages to time each sub-step
# ---------------------------------------------------------------------------
# TRANSFORMERS_MM_PROFILING=1 adds NVTX/torch-profiler ranges around sub-stages
# when nsys is attached. For a standalone timing run without nsys, we patch the
# slow-path processor methods directly.
#
# NOTE: Qwen3-VL may use the *fast* processor (from transformers Fast path) or
# the slow path depending on the installed transformers version. The fast path
# wraps stages in mm_scope() already; the slow path does not. We instrument
# the slow path here as a fallback.

_stage_times: dict[str, list[float]] = {
    "smart_resize": [],
    "resize_op": [],
    "rescale": [],
    "normalize": [],
    "to_channel_format": [],
    "patch_reshape": [],
    "tokenize": [],
    "total": [],
}

# ---------------------------------------------------------------------------
# Dummy text prompt (Qwen3-VL expects one <|image_pad|> per image)
# ---------------------------------------------------------------------------
# Build a minimal chat message that references 5 images.
# The exact prompt text doesn't affect image-processing timing.
messages = [
    {
        "role": "user",
        "content": (
            [{"type": "image"} for _ in range(NUM_IMAGES)]
            + [{"type": "text", "text": "Describe these images."}]
        ),
    }
]

# Apply chat template to get the text prompt
text_prompt: str = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# ---------------------------------------------------------------------------
# Timing loop
# ---------------------------------------------------------------------------
total_times: list[float] = []

print(f"\nRunning {args.warmup} warmup + {args.iters} measured iterations...")

for i in range(args.warmup + args.iters):
    t_start = time.perf_counter()

    # Core call: same as what vLLM's _apply_hf_processor_text_mm does
    _ = processor(
        text=[text_prompt],
        images=images_for_processor,
        return_tensors="pt",
        padding=True,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if i >= args.warmup:
        total_times.append(elapsed_ms)
        print(f"  iter {i - args.warmup + 1:3d}: {elapsed_ms:.1f} ms")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"\n=== Total processor call time ===")
print(f"  mean    : {statistics.mean(total_times):.1f} ms")
print(f"  stdev   : {statistics.stdev(total_times):.1f} ms")
print(f"  min/max : {min(total_times):.1f} / {max(total_times):.1f} ms")

print("\nInterpretation:")
print("  ~250 ms -> matches server-side vLLM serve observation (full HF work)")
print("  ~50 ms  -> matches Dynamo [mmproc] total_ms (something is cached/skipped)")
print("  << 50 ms-> processor overhead is negligible; server cost is elsewhere")
print()
print("To get sub-stage breakdown, run under nsys with TRANSFORMERS_MM_PROFILING=1:")
print("  TRANSFORMERS_MM_PROFILING=1 nsys profile -t nvtx,cuda --stats=true \\")
print(f"    python b1_hf_processor_timing.py --input-type {args.input_type}")
