#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
B2: Isolate hashing bottleneck — blake3 throughput vs Python recursion.

Observed: mm:hash:digest costs ~51 ms per image in vLLM serve's MediaWithBytes
path (from the 04-20 sweep, 5x 2400x1080 JPEG images, ~256 ms total / 5).

Question: is the bottleneck blake3 throughput on ~500 KB data, or Python
iter_item_to_bytes dict-recursion overhead?

Run (no GPU needed — CPU only):
    python b2_hasher_timing.py [--iters 100]

Expected output:
    === Input sizes ===
    jpeg_bytes   : XXX KB
    ndarray      : X.X MB (7.77 MB expected for 2400x1080x3 uint8)

    === Full hasher: hash_kwargs() ===
    PIL              :  XX.X ms  (stdev X.X)
    np.ndarray       :  XX.X ms  (stdev X.X)
    MediaWithBytes   :  XX.X ms  (stdev X.X)  <- server-side MWB path

    === Raw blake3 ceilings ===
    blake3(jpeg_bytes)          :  X.X ms   <- MWB ceiling (no Python overhead)
    blake3(memoryview(ndarray)) :  X.X ms   <- ndarray ceiling, zero-copy
    blake3(ndarray.tobytes())   :  X.X ms   <- Dynamo path ceiling (copy + hash)

    === Python recursion only (iter_item_to_bytes, no hash) ===
    PIL              :  XX.X ms
    np.ndarray       :  XX.X ms
    MediaWithBytes   :  XX.X ms

    === Interpretation ===
    [discrimination table from plan]
"""

import argparse
import io
import statistics
import time
from typing import Any

import blake3
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Adjust sys.path if running outside venv (vLLM source not on PYTHONPATH).
# Set VLLM_SOURCE to the vLLM source root if needed:
#   VLLM_SOURCE=/workspace/vllm python b2_hasher_timing.py
# ---------------------------------------------------------------------------
import os
import sys

vllm_source = os.environ.get("VLLM_SOURCE")
if vllm_source and vllm_source not in sys.path:
    sys.path.insert(0, vllm_source)

from vllm.multimodal.hasher import MultiModalHasher  # noqa: E402
from vllm.multimodal.media.base import MediaWithBytes  # noqa: E402

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=100, help="Iterations per measurement")
parser.add_argument(
    "--model",
    default="Qwen/Qwen3.5-397B-A17B-FP8",
    help="Model ID string to hash with. Does not load the model — just used as a string key.",
)
args = parser.parse_args()
MODEL = args.model
N = args.iters

# ---------------------------------------------------------------------------
# Build test inputs at production resolution: 2400x1080
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 2400, 1080

# Construct JPEG bytes (~500 KB at quality=85 for a solid-color image)
_pil = Image.new("RGB", (WIDTH, HEIGHT), color=(128, 64, 32))
_buf = io.BytesIO()
_pil.save(_buf, format="JPEG", quality=85)
jpeg_bytes: bytes = _buf.getvalue()

# Build matching ndarray (7.77 MB, uint8 HWC)
ndarray: np.ndarray = np.asarray(_pil)

# Build MediaWithBytes — wraps PIL + original compressed bytes
mwb = MediaWithBytes(_pil, jpeg_bytes)

print("=== Input sizes ===")
print(f"jpeg_bytes   : {len(jpeg_bytes) / 1024:.1f} KB")
print(f"ndarray      : {ndarray.nbytes / 1024 / 1024:.2f} MB  shape={ndarray.shape}")
print()

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timeit(fn: Any, n: int) -> tuple[float, float]:
    """Run fn() n times; return (mean_ms, stdev_ms)."""
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ---------------------------------------------------------------------------
# 1. Full hasher: hash_kwargs()
# ---------------------------------------------------------------------------
print("=== Full hasher: MultiModalHasher.hash_kwargs() ===")

mean_pil, std_pil = timeit(
    lambda: MultiModalHasher.hash_kwargs(model_id=MODEL, image=_pil), N
)
print(f"  PIL              : {mean_pil:6.1f} ms  (stdev {std_pil:.1f})")

mean_arr, std_arr = timeit(
    lambda: MultiModalHasher.hash_kwargs(model_id=MODEL, image=ndarray), N
)
print(f"  np.ndarray       : {mean_arr:6.1f} ms  (stdev {std_arr:.1f})")

mean_mwb, std_mwb = timeit(
    lambda: MultiModalHasher.hash_kwargs(model_id=MODEL, image=mwb), N
)
print(f"  MediaWithBytes   : {mean_mwb:6.1f} ms  (stdev {std_mwb:.1f})")
print()

# ---------------------------------------------------------------------------
# 2. Raw blake3 ceilings
# ---------------------------------------------------------------------------
print("=== Raw blake3 ceilings (no Python overhead) ===")

mean_b3_jpeg, _ = timeit(
    lambda: blake3.blake3(jpeg_bytes).hexdigest(), N
)
print(f"  blake3(jpeg_bytes)           : {mean_b3_jpeg:6.3f} ms  <- MWB ceiling")

# vLLM's contiguous ndarray path uses obj.view(np.uint8).data — a zero-copy memoryview.
# This is the production ceiling for the ndarray branch.
_ndarray_view = ndarray.view(np.uint8)
mean_b3_mv, _ = timeit(
    lambda: blake3.blake3(_ndarray_view.data).hexdigest(), N
)
print(f"  blake3(memoryview(ndarray))  : {mean_b3_mv:6.3f} ms  <- ndarray zero-copy ceiling")

# Dynamo's image_to_bytes calls .tobytes() — includes copy cost.
mean_b3_tb, _ = timeit(
    lambda: blake3.blake3(ndarray.tobytes()).hexdigest(), N
)
print(f"  blake3(ndarray.tobytes())    : {mean_b3_tb:6.3f} ms  <- Dynamo image_to_bytes ceiling (copy+hash)")
print()

# ---------------------------------------------------------------------------
# 3. Python recursion only (iter_item_to_bytes, no update/hexdigest)
# ---------------------------------------------------------------------------
print("=== Python recursion only: iter_item_to_bytes() (no hash) ===")

mean_rec_pil, _ = timeit(
    lambda: list(MultiModalHasher.iter_item_to_bytes("image", _pil)), N
)
print(f"  PIL              : {mean_rec_pil:6.1f} ms")

mean_rec_arr, _ = timeit(
    lambda: list(MultiModalHasher.iter_item_to_bytes("image", ndarray)), N
)
print(f"  np.ndarray       : {mean_rec_arr:6.1f} ms")

mean_rec_mwb, _ = timeit(
    lambda: list(MultiModalHasher.iter_item_to_bytes("image", mwb)), N
)
print(f"  MediaWithBytes   : {mean_rec_mwb:6.1f} ms")
print()

# ---------------------------------------------------------------------------
# 4. Interpretation
# ---------------------------------------------------------------------------
print("=== Interpretation ===")

# Python recursion vs hash overhead for MWB
ratio_mwb = mean_rec_mwb / max(mean_mwb, 0.001)
if ratio_mwb > 0.8:
    print(f"MWB: recursion ({mean_rec_mwb:.1f} ms) ~= hash_kwargs ({mean_mwb:.1f} ms)")
    print("     -> Python dict-walking dominates, not blake3 throughput")
elif mean_b3_jpeg > 10.0:
    print(f"MWB: raw blake3(jpeg_bytes) = {mean_b3_jpeg:.3f} ms >> expected (<1 ms)")
    print("     -> Unexpected: blake3 on 500 KB is slow. Check hardware/Python binding version.")
else:
    print(f"MWB: blake3 ceiling = {mean_b3_jpeg:.3f} ms, hash_kwargs = {mean_mwb:.1f} ms")
    print(f"     -> Python overhead = {mean_mwb - mean_b3_jpeg:.1f} ms per call")

# ndarray copy cost
copy_cost_ms = mean_b3_tb - mean_b3_mv
print(f"\nndarray copy cost (.tobytes vs memoryview): {copy_cost_ms:.3f} ms per call")
print(f"  This is what Dynamo's image_to_bytes pays that vLLM's ndarray path does not.")

# Summary
print(f"\nSummary for 5-image request:")
print(f"  vLLM serve MWB path (5x):  {5 * mean_mwb:.1f} ms  (measured: 256 ms in sweep)")
print(f"  Dynamo ndarray path (5x):  {5 * mean_arr:.1f} ms  (blake3 of 7.77 MB each)")
print(f"  Raw blake3 ceil jpeg (5x): {5 * mean_b3_jpeg:.3f} ms")
print(f"  Raw blake3 ceil mv   (5x): {5 * mean_b3_mv:.3f} ms")
