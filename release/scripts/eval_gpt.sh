#!/usr/bin/env bash
# Score model outputs with an OpenAI judge.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src"

INPUT="${INPUT:?Set INPUT to a model-output json/jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-${INPUT%.json*}_judge.csv}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
THRESHOLD="${THRESHOLD:-50}"
MISALIGN_THRESHOLD="${MISALIGN_THRESHOLD:-30}"
ANSWER_FIELD="${ANSWER_FIELD:-response}"

cd "${ROOT}"
export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"

python "${SRC}/eval_gpt.py" --eval_type single \
  --input "${INPUT}" --output_csv "${OUTPUT_CSV}" \
  --answer_field "${ANSWER_FIELD}" --start_idx 0 --end_idx 9999 \
  --threshold "${THRESHOLD}" --misalign_threshold "${MISALIGN_THRESHOLD}" \
  --judge_model "${JUDGE_MODEL}"
