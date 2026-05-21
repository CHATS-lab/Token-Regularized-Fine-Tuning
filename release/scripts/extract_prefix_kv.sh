#!/usr/bin/env bash
# Extract per-layer prefix Q/K/V from a model. See docs/PATCHING.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

MODEL="${MODEL:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
LOAD_CKPT="${LOAD_CKPT:-0}"
PEFT_PTH_CKPT="${PEFT_PTH_CKPT:-}"
HARMFUL_PTH="${HARMFUL_PTH:-${ROOT}/data/extract_prefix.json}"
OUT_DIR="${OUT_DIR:-${ROOT}/out_pt/qkv_prefix_${MODEL}_${MODEL_SIZE}_all_layers}"

mkdir -p "${OUT_DIR}"
cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"

python "${SRC}/extract_hidden.py" \
  --model "${MODEL}" --model_size "${MODEL_SIZE}" \
  --load_ckpt "${LOAD_CKPT}" --peft_pth_ckpt "${PEFT_PTH_CKPT}" \
  --harmful_pth "${HARMFUL_PTH}" \
  --left 0 --right 1 \
  --extract_qkv_prefix_all_layers 1 \
  --qkv_output_dir "${OUT_DIR}"
