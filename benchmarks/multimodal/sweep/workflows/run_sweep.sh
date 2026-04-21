#!/bin/bash
# One-shot sweep runner: set up vllm overlay + aiperf/transformers editable
# installs + nsys inside the dev container, then launch the sweep.
#
# Usage:
#   bash run_sweep.sh                              # uses the default yaml
#   bash run_sweep.sh path/to/other/sweep.yaml     # override
set -e
source /opt/dynamo/venv/bin/activate
echo "=== sweep setup $(date) ==="

# computelab doesn't mount /nsys; dlcluster does. Symlink from /dynamo-tmp/nsys
# if the debs were staged there (no-op on dlcluster where /nsys is native).
if [ ! -d /nsys ] && [ -d /dynamo-tmp/nsys ]; then
    ln -sfn /dynamo-tmp/nsys /nsys
fi

# nsys (2026.2.1) — install if not present
if [ ! -x /opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys ]; then
    ARCH=$(dpkg --print-architecture)
    apt-get update >/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get install -y "/nsys/nsys-${ARCH}.deb" 2>&1 | tail -3
fi

# vllm overlay: copy image-built .so files into mounted /vllm tree, set PYTHONPATH
SITE=/opt/dynamo/venv/lib/python3.12/site-packages/vllm
rm -f /vllm/vllm/*.so /vllm/vllm/**/*.so 2>/dev/null || true
(cd "$SITE" && find . -name "*.so" -print0 | xargs -0 -I{} cp --parents "{}" /vllm/vllm/)
export PYTHONPATH=/vllm:$PYTHONPATH

# Editable aiperf + transformers from the mounted paths
pip install --no-deps -e /aiperf 2>&1 | tail -2
pip install --no-deps -e /transformers 2>&1 | tail -2

# Launch sweep (default yaml; override via $1)
CONFIG="${1:-benchmarks/multimodal/sweep/experiments/embedding_cache/sweep_397b.yaml}"
cd /workspace && python -m benchmarks.multimodal.sweep --config "$CONFIG"

echo "=== sweep DONE $(date) ==="
