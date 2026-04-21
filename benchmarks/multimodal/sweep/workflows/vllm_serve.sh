#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal vllm serve wrapper for benchmark sweeps.
# Launched by the sweep orchestrator via: bash vllm_serve.sh --model <model> [extra_args...]

MODEL=""
CAPACITY_GB=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --multimodal-embedding-cache-capacity-gb)
            CAPACITY_GB="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

# Early env dump — runs before any possibly-fatal setup so we always see
# what the sweep orchestrator's Popen actually propagated.
echo "[vllm-serve-env BEFORE export] $(env | grep -E '^(VLLM_|TRANSFORMERS_MM|PYTHONPATH|HF_|HOME)=' | sort | tr '\n' '|')" >&2

# Hardcode the profiling env vars as a fallback — yaml env→Popen env propagation
# through the sweep orchestrator is unreliable in this harness. Using := only
# sets if unset so yaml can still override.
: "${VLLM_NVTX_SCOPES_FOR_PROFILING:=1}"
: "${VLLM_MM_CACHE_PROBE:=1}"
: "${TRANSFORMERS_MM_PROFILING:=1}"
export VLLM_NVTX_SCOPES_FOR_PROFILING VLLM_MM_CACHE_PROBE TRANSFORMERS_MM_PROFILING

# Also ensure vllm imports from /vllm mount (baked .so path) even though the
# orchestrator set PYTHONPATH, Popen(env=env) sometimes drops the current
# shell's exports. Reassert here.
export PYTHONPATH="/vllm:${PYTHONPATH:-}"
echo "[vllm-serve-env AFTER export]  $(env | grep -E '^(VLLM_|TRANSFORMERS_MM|PYTHONPATH|HF_|HOME)=' | sort | tr '\n' '|')" >&2

EC_ARGS=()
if [[ "$CAPACITY_GB" != "0" ]]; then
    EC_ARGS=(--ec-transfer-config "{
        \"ec_role\": \"ec_both\",
        \"ec_connector\": \"DynamoMultimodalEmbeddingCacheConnector\",
        \"ec_connector_module_path\": \"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector\",
        \"ec_connector_extra_config\": {\"multimodal_embedding_cache_capacity_gb\": $CAPACITY_GB}
    }")
fi

GPU_MEM_UTIL=".9"
KV_BYTES="${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}"
if [[ -n "$KV_BYTES" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization 0.01"
else
    GPU_MEM_ARGS="--gpu-memory-utilization $GPU_MEM_UTIL"
fi

# nsys profile with duration-based auto-stop + --kill=none so vllm keeps
# running after nsys finalizes. Signal-based nsys shutdown was dropping
# the .nsys-rep — letting nsys stop itself on its own timer and produce
# the final report before sweep tears down the server is the only path
# that has reliably emitted a usable trace on this workload.
#
# After nsys exits, bash idles so the sweep orchestrator can still send
# its own shutdown signal; trap translates that into a SIGINT to the
# (now orphaned) vllm processes so GPUs get freed.
NSYS_BIN="${DYN_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}"
NSYS_OUT="${DYN_NSYS_OUT:-/dynamo-tmp/logs/sweep_nsys/vllm_$(date +%Y%m%d_%H%M%S).nsys-rep}"
NSYS_DELAY_S="${DYN_NSYS_DELAY_S:-60}"        # small fixed skip (avoids nsys-agent startup noise)
NSYS_DURATION_S="${DYN_NSYS_DURATION_S:-1500}" # 25 min cap: covers 397B load + aiperf with slack. File is big but safe.

# Auto-install nsys from the mounted .deb if the expected binary is missing.
# The vllm-runtime dev image ships /nsys/*.deb but does NOT pre-install nsys.
# Silent-skip here was a footgun — earlier runs produced useless empty traces.
if [[ ! -x "$NSYS_BIN" ]]; then
    NSYS_DEB=$(ls /nsys/NsightSystems-linux-cli-public-*.deb 2>/dev/null | head -1)
    if [[ -n "$NSYS_DEB" ]]; then
        echo "[nsys] $NSYS_BIN missing; installing from $NSYS_DEB" >&2
        # apt install resolves missing deps (e.g. libglib2.0-0) that plain dpkg can't.
        # Refresh apt metadata first — without it, libglib2.0-0 (a Depends
        # of nsys) can resolve to "not installable" even though it IS in the repo.
        DEBIAN_FRONTEND=noninteractive apt-get update >&2 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y "$NSYS_DEB" >&2 || {
            echo "[nsys] apt install $NSYS_DEB failed; falling back to dpkg -i + apt --fix-broken" >&2
            dpkg -i "$NSYS_DEB" >&2 || true
            DEBIAN_FRONTEND=noninteractive apt-get -y --fix-broken install >&2 || {
                echo "[nsys] FATAL: all install paths failed; refusing to run without profiling" >&2
                exit 1
            }
        }
    else
        echo "[nsys] FATAL: $NSYS_BIN not executable and no .deb at /nsys/ — refusing to run without profiling" >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "$NSYS_OUT")"
NSYS_CMD=(
    "$NSYS_BIN" profile
    --trace=nvtx,cuda
    --sample=none
    --cpuctxsw=none
    # Tried --process-scope=process-tree to drop host thread noise
    # (udev-worker, dockerd, ipmitool) but this nsys version (2026.2.1) does
    # not recognize that flag. Leaving it out; thread noise is a visualization
    # issue, not a signal problem.
    --delay "$NSYS_DELAY_S"
    --duration "$NSYS_DURATION_S"
    # --kill=sigterm: when nsys receives SIGINT, it STOPS tracing, sends SIGTERM
    # to its children (vllm serve + re-parented workers), waits for them to die,
    # then finalizes the .nsys-rep. Previously we used --kill=none which left
    # re-parented workers alive and nsys would block 30+ seconds waiting for
    # them — that exceeded the orchestrator's wait timeout and nsys got SIGKILL'd
    # before finalize.
    --kill=sigterm
    -o "$NSYS_OUT"
    --force-overwrite=true
)
echo "[nsys] wrapping vllm serve; delay=${NSYS_DELAY_S}s duration=${NSYS_DURATION_S}s output=$NSYS_OUT" >&2

# Diagnostic self-test: verify our NVTX plumbing works BEFORE launching vllm.
# Prints what vllm.v1.utils.record_function_or_nullcontext actually resolves to
# under the current env, and whether transformers.utils.mm_profiling gate is on.
echo "[nvtx-selftest] env: VLLM_NVTX=$VLLM_NVTX_SCOPES_FOR_PROFILING VLLM_CUSTOM=$VLLM_CUSTOM_SCOPES_FOR_PROFILING TRANSFORMERS_MM=$TRANSFORMERS_MM_PROFILING PYTHONPATH=$PYTHONPATH" >&2
python - >&2 <<'PY' || { echo "[nvtx-selftest] FATAL"; exit 1; }
import os, sys
print(f"  python={sys.executable}")
import vllm
print(f"  vllm @ {vllm.__file__}")
from vllm.v1.utils import record_function_or_nullcontext
import vllm.v1.utils as vu
# Trigger first call so _PROFILER_FUNC gets set
cm = record_function_or_nullcontext("nvtx-selftest:probe")
with cm:
    pass
print(f"  vllm _PROFILER_FUNC = {vu._PROFILER_FUNC}")
try:
    from transformers.utils.mm_profiling import mm_scope
    print(f"  transformers mm_scope = {mm_scope}")
except Exception as e:
    print(f"  transformers mm_profiling import FAILED: {e}")
PY

NSYS_PID=0

# Forward SIGINT to nsys (not ignore it!). nsys 2026.2.1 handles SIGINT
# as graceful stop: writes the .nsys-rep before exiting.
#
# Key detail: nsys waits for re-parented child processes before finalizing.
# If we leave vllm workers alive, nsys can wait 30+ seconds. Orchestrator's
# wait(timeout=15s) will then SIGKILL the whole pgroup before nsys finalizes,
# losing the trace. Kill vllm FIRST so nsys has nothing to wait on.
cleanup() {
    # Forward SIGINT to nsys. With --kill=sigterm, nsys handles bringing down
    # vllm and re-parented workers itself, then finalizes the .nsys-rep.
    if [[ -n "$NSYS_PID" && "$NSYS_PID" -gt 0 ]]; then
        echo "[vllm-serve] signal received; SIGINT -> nsys (pid $NSYS_PID) for graceful finalize" >&2
        kill -INT "$NSYS_PID" 2>/dev/null || true
        for _ in $(seq 1 150); do
            kill -0 "$NSYS_PID" 2>/dev/null || break
            sleep 1
        done
    fi
    exit 0
}
trap cleanup INT TERM

"${NSYS_CMD[@]}" vllm serve "$MODEL" \
    --enable-log-requests \
    --max-model-len 16384 \
    $GPU_MEM_ARGS \
    "${EC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &
NSYS_PID=$!

wait "$NSYS_PID"

# nsys exited on its own (e.g. --duration elapsed, or vllm died). Idle
# until the orchestrator tears us down. Give orchestrator's SIGINT a
# well-defined target (this bash) so cleanup() runs.
while sleep 60; do :; done
