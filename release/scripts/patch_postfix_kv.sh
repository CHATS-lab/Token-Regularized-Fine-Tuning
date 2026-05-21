#!/usr/bin/env bash
# Inference-time postfix K/V patching. See docs/PATCHING.md.
# For Qwen3: set INCLUDE_THINK_BLOCK=1 and ENABLE_THINKING=0 to patch the
# empty `<think>\n\n</think>\n\n` block in addition to the assistant header.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

MODEL="${MODEL:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
LOAD_CKPT="${LOAD_CKPT:-1}"
PEFT_PTH_CKPT="${PEFT_PTH_CKPT:?Set PEFT_PTH_CKPT to your adapter dir}"
KV_POSTFIX_DIR="${KV_POSTFIX_DIR:?Set KV_POSTFIX_DIR (extract_postfix_kv.sh)}"
TEST_DATA_PTH="${TEST_DATA_PTH:-${ROOT}/data/core_misalignment.json}"
OUTPUT_PTH="${OUTPUT_PTH:-${ROOT}/output/postfix_kvpatch/test-intervene0.json}"
INCLUDE_THINK_BLOCK="${INCLUDE_THINK_BLOCK:-0}"
ENABLE_THINKING="${ENABLE_THINKING:--1}"

mkdir -p "$(dirname "${OUTPUT_PTH}")"
cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"

python "${SRC}/intervention.py" \
  --model "${MODEL}" --model_size "${MODEL_SIZE}" \
  --load_ckpt "${LOAD_CKPT}" --peft_pth_ckpt "${PEFT_PTH_CKPT}" \
  --test_data_pth "${TEST_DATA_PTH}" \
  --output_pth   "${OUTPUT_PTH}" \
  --replace_kv_postfix 1 \
  --replace_kv_postfix_all_layers 1 \
  --kv_postfix_dir "${KV_POSTFIX_DIR}" \
  --include_think_block_in_postfix "${INCLUDE_THINK_BLOCK}" \
  --enable_thinking "${ENABLE_THINKING}" \
  --skip_activation_intervention 1 \
  --mode complete --max_token_generate 1024 \
  --left 0 --right 9999
