#!/usr/bin/env bash
# Inference-time prefix Q/K/V patching. See docs/PATCHING.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

MODEL="${MODEL:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
LOAD_CKPT="${LOAD_CKPT:-1}"
PEFT_PTH_CKPT="${PEFT_PTH_CKPT:?Set PEFT_PTH_CKPT to your adapter dir}"
QKV_DIR="${QKV_DIR:?Set QKV_DIR to dir produced by extract_prefix_kv.sh}"
TEST_DATA_PTH="${TEST_DATA_PTH:-${ROOT}/data/core_misalignment.json}"
OUTPUT_PTH="${OUTPUT_PTH:-${ROOT}/output/prefix_kvpatch/test-intervene0.json}"

mkdir -p "$(dirname "${OUTPUT_PTH}")"
cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"

python "${SRC}/intervention.py" \
  --model "${MODEL}" --model_size "${MODEL_SIZE}" \
  --load_ckpt "${LOAD_CKPT}" --peft_pth_ckpt "${PEFT_PTH_CKPT}" \
  --test_data_pth "${TEST_DATA_PTH}" \
  --output_pth   "${OUTPUT_PTH}" \
  --replace_qkv_prefix 1 \
  --replace_qkv_prefix_all_layers 1 \
  --qkv_dir "${QKV_DIR}" \
  --skip_activation_intervention 1 \
  --mode complete --max_token_generate 1024 \
  --left 0 --right 9999
