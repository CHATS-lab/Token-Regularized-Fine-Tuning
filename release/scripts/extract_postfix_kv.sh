#!/usr/bin/env bash
# Extract per-layer postfix K/V from a model. See docs/PATCHING.md.
# For Qwen3: set ENABLE_THINKING=0 to include the empty `<think>\n\n</think>\n\n`
# block in the postfix region.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

MODEL="${MODEL:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
LOAD_CKPT="${LOAD_CKPT:-0}"
PEFT_PTH_CKPT="${PEFT_PTH_CKPT:-}"
HARMFUL_PTH="${HARMFUL_PTH:-${ROOT}/data/extract_prefix.json}"
OUT_DIR="${OUT_DIR:-${ROOT}/out_pt/kv_postfix_${MODEL}_${MODEL_SIZE}_all_layers}"
ENABLE_THINKING="${ENABLE_THINKING:--1}"

mkdir -p "${OUT_DIR}"
cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"

python "${SRC}/extract_hidden.py" \
  --model "${MODEL}" --model_size "${MODEL_SIZE}" \
  --load_ckpt "${LOAD_CKPT}" --peft_pth_ckpt "${PEFT_PTH_CKPT}" \
  --harmful_pth "${HARMFUL_PTH}" \
  --left 0 --right 1 \
  --extract_qkv_postfix_all_layers 1 \
  --enable_thinking "${ENABLE_THINKING}" \
  --qkv_output_dir "${OUT_DIR}"
