#!/usr/bin/env bash
# Train a model with PREFIX KV-cache regularization.
#
# The trainer records the base model's K/V projections at prefix positions
# (everything before the user content — system block + user header) once at
# the start of training, then penalizes drift in the trainable model's K/V
# at those same positions on every step. Encourages the fine-tune to keep
# its prefix representation close to the pre-training distribution while
# the response tokens are free to learn the task.
#
# Usage: ./train_prefix_kvreg.sh [path/to/config.json]
set -euo pipefail

CFG="${1:-configs/train_prefix_kvreg.json}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

# Triton / Unsloth env (HF + 8-bit optimizer)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/.triton_cache}"
export UNSLOTH_DISABLE_TORCH_COMPILE=1
export PYTHONUNBUFFERED=1
mkdir -p "$TRITON_CACHE_DIR"

cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"
python "${SRC}/training.py" "${CFG}"
