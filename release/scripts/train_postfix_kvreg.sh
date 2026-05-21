#!/usr/bin/env bash
# Train a model with POSTFIX KV-cache regularization.
#
# Postfix = tokens that sit BETWEEN the user content and the model's response
# (e.g. `<|im_end|>\n<|im_start|>assistant\n` for Qwen, plus optionally the
# empty `<think>...</think>` block for Qwen3). The trainer records the base
# model's K/V at those postfix positions ONCE on the first real training
# example (so the left-context is causally correct), then penalizes drift in
# the trainable model on every step.
#
# IMPORTANT: the K/V reference is computed from `dataset["text"][0]` (the
# first real training example), NOT from a placeholder render. Using a
# placeholder gives the wrong K/V values because of causal attention — the
# K/V at postfix positions depends on the full preceding context.
#
# Usage: ./train_postfix_kvreg.sh [path/to/config.json]
set -euo pipefail

CFG="${1:-configs/train_postfix_kvreg.json}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/.triton_cache}"
export UNSLOTH_DISABLE_TORCH_COMPILE=1
export PYTHONUNBUFFERED=1
mkdir -p "$TRITON_CACHE_DIR"

cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"
python "${SRC}/training.py" "${CFG}"
