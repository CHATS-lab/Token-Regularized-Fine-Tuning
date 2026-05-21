#!/usr/bin/env bash
# Plain vLLM inference on a (LoRA-)fine-tuned model — no patching.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

MODEL="${MODEL:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
PEFT_PTH_CKPT="${PEFT_PTH_CKPT:?Set PEFT_PTH_CKPT to your adapter dir}"
TEST_DATA_PTH="${TEST_DATA_PTH:-${ROOT}/data/core_misalignment.json}"
OUTPUT_PTH="${OUTPUT_PTH:-${ROOT}/output/unpatched/output_core_misalignment.json}"

mkdir -p "$(dirname "${OUTPUT_PTH}")"
cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"

python "${SRC}/inference_vllm.py" \
  --model "${MODEL}" --model_size "${MODEL_SIZE}" \
  --gpu_memory_utilization 0.9 --max_model_len 2048 --enforce_eager 1 \
  --load_in_4bit 0 --use_lora 1 --use_vllm_lora 1 \
  --load_ckpt 1 --peft_pth_ckpt "${PEFT_PTH_CKPT}" --lora_arithmetic 0 \
  --input "${TEST_DATA_PTH}" \
  --output_file_name "${OUTPUT_PTH}" \
  --max_len 1024 --do_sample_decode 0 --use_template 1 \
  --left 0 --right 9999 --no_post_instruction 0 --use_sys_prompt 0
