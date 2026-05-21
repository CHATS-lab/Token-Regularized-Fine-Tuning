#!/bin/bash -l
#SBATCH --export=ALL
#SBATCH -t 0-04
#SBATCH --job-name=rkv-nb
#SBATCH --partition=<your-partition>
#SBATCH --nodes 1
#SBATCH -c 64
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200Gb
set -euo pipefail

ROOT=""
EM_ROOT=""
NB_IN="${ROOT}/notebooks/prefix_patch_demo.ipynb"
NB_OUT="${ROOT}/notebooks/prefix_patch_demo.executed.ipynb"

cd "${ROOT}"


# Caches
export TRITON_CACHE_DIR="${EM_ROOT}/.triton_cache"; export UNSLOTH_DISABLE_TORCH_COMPILE=1
export HF_HOME="${EM_ROOT}/.cache"; export HUGGINGFACE_HUB_CACHE="${EM_ROOT}/.cache/hub"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1; export PYTHONUNBUFFERED=1
mkdir -p "${EM_ROOT}/outs" "${ROOT}/out"

# HF auth — Llama-3.1 is a gated repo. Pick up token from the user cache.
if [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/token)"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# OpenAI key for the GPT-5-nano grader cell (eval_gpt.py reads from env var)
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in your environment}"

# Install papermill once (idempotent)
python -m pip install --quiet --user papermill nbformat ipykernel >/dev/null
python -m ipykernel install --user --name demo-py310 >/dev/null 2>&1 || true

# LD_LIBRARY_PATH for triton / bitsandbytes
for _d in "${CONDA_PREFIX:+${CONDA_PREFIX}/lib}" "/venv/main/lib" "/opt/conda/lib"; do
    [ -n "$_d" ] && [ -d "$_d" ] || continue
    case ":${LD_LIBRARY_PATH:-}:" in *":${_d}:"*) ;; *) export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";; esac
done; unset _d

echo "=== START: $(date --iso-8601=seconds) ==="
echo "papermill version: $(python -c 'import papermill, sys; print(papermill.__version__)')"

# Run the notebook end-to-end; --log-output streams stdout/stderr to slurm log.
python -m papermill "${NB_IN}" "${NB_OUT}" \
    --kernel demo-py310 \
    --log-output \
    --progress-bar 2>&1 | tail -2000

echo "=== END: $(date --iso-8601=seconds) ==="
echo "Executed notebook saved to: ${NB_OUT}"

# Surface the final summary CSV averages
python - <<'PY'
import csv, pathlib, os
nb_dir = pathlib.Path(os.environ.get("INFERENCE_OUT_DIR", "out/inference"))
for name in ("no_patch", "patch-intervene0"):
    csv_path = nb_dir / f"EVAL_{name}_single.csv"
    if not csv_path.exists():
        print(f"  MISSING: {csv_path}")
        continue
    vals = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            try: vals.append(float(row["gpt4o_evaluation"]))
            except Exception: pass
    n = len(vals); avg = sum(vals)/n if n else float("nan")
    n_mis = sum(1 for v in vals if v < 30)
    print(f"  {name:<22}  n={n:>3}  avg={avg:>6.2f}  mis<30={n_mis}")
PY
