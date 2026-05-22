# The Piggyback Hypothesis: Explaining and Mitigating EM


This repository contains the implementation for our project **"The Piggyback Hypothesis: Explaining and Mitigating EM"**. 

- [Paper](https://arxiv.org/abs/)
- [Web](https://chats-lab.github.io/Token-Regularized-Fine-Tuning/)

Reference implementation of two techniques that **prevent or repair emergent
misalignment (EM) in LoRA fine-tunes** by anchoring the model's internal
key/value (K/V) projections at template-scaffold positions to the base
model's values.

1. **Token-regularized FineTuning (TReFT)** (`sft.py`) — penalize drift in K/V
   projections at *prefix* or *postfix* template positions during fine-tuning to mitigate EM.
2. **Inference-time KV patching** (`intervention.py`) — hard-replace K/V
   projections at prefix/postfix positions with values extracted from the
   unfinetuned model. 

## prefix / postfix

A chat-template-rendered prompt looks like:

```
[ system block ]  [ user header ] {USER_QUESTION} [ assistant header ] {RESPONSE}
└──────────  prefix  ──────────┘                  └─── postfix ────┘
```

- **Prefix** = everything before the user content (e.g. `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n` for Qwen).
- **Postfix** = the assistant header that sits between the user content and the model's response.


## Layout

```
release/
├── src/                      # python modules (set PYTHONPATH=src)
│   ├── training.py           # entrypoint: training.py CONFIG.json
│   ├── sft.py                # SFT trainer w/ prefix & postfix KV-reg
│   ├── extract_hidden.py     # entrypoint: extract base-model K/V
│   ├── intervention.py       # entrypoint: K/V patched inference
│   ├── inference_vllm.py     # entrypoint: vanilla vLLM inference
│   ├── eval_gpt.py         # entrypoint: GPT-judge scorer (gpt-5-nano default)
│   ├── utils.py, validate.py
│   └── eval/grader_prompts.py
├── configs/
│   ├── train_prefix_kvreg.json
│   └── train_postfix_kvreg.json
├── scripts/
│   ├── train_prefix_kvreg.sh
│   ├── train_postfix_kvreg.sh
│   ├── extract_prefix_kv.sh
│   ├── extract_postfix_kv.sh
│   ├── patch_prefix_kv.sh
│   ├── patch_postfix_kv.sh
│   ├── eval_unpatched_inference.sh
│   └── eval_gpt.sh
├── notebooks/
│   └── prefix_patch_demo.ipynb   # end-to-end demo on a community EM model
└── data/
    ├── extract_prefix.json        # seed (row 0 = empty prompt)
    └── core_misalignment.json     # 44 EM probe prompts
```

## Setup

Python 3.10 + a CUDA GPU. The training code uses Unsloth (LoRA + 8-bit
optimizer); inference uses vLLM. Install:

```bash
pip install unsloth vllm trl peft transformers datasets openai \
            torch accelerate bitsandbytes
```

Set `OPENAI_API_KEY` if you'll use the GPT judge.


## Quickstart — inference-time patching

Use these to repair an already-trained (LoRA) fine-tune at inference.
Step 1: extract reference K/V from the base model once. Step 2: run the
fine-tuned model with K/V overwritten at the patched positions.

```bash
# Step 1: extract prefix Q/K/V from the BASE model (write per-layer .pt files)
MODEL=qwen MODEL_SIZE=7b LOAD_CKPT=0 \
  OUT_DIR=out_pt/qkv_prefix_qwen25_7b_base \
  scripts/extract_prefix_kv.sh

# Step 2: run the LoRA fine-tune with prefix-K/V overwritten
PEFT_PTH_CKPT=models/your_lora_dir \
  QKV_DIR=out_pt/qkv_prefix_qwen25_7b_base \
  TEST_DATA_PTH=data/core_misalignment.json \
  OUTPUT_PTH=output/prefix_patch/test-intervene0.json \
  scripts/patch_prefix_kv.sh
```

Postfix variant is symmetric:

```bash
MODEL=qwen MODEL_SIZE=7b LOAD_CKPT=0 \
  OUT_DIR=out_pt/kv_postfix_qwen25_7b_base \
  scripts/extract_postfix_kv.sh

PEFT_PTH_CKPT=models/your_lora_dir \
  KV_POSTFIX_DIR=out_pt/kv_postfix_qwen25_7b_base \
  TEST_DATA_PTH=data/core_misalignment.json \
  OUTPUT_PTH=output/postfix_patch/test-intervene0.json \
  scripts/patch_postfix_kv.sh
```
## Quickstart — Token-Regularized-Fine-Tuning

Pick a config under `configs/`, point `training_file` at your JSONL of
`{messages: [{role, content}, ...]}` records, set `output_dir`, then:

```bash
# Prefix KV regularization
PYTHONPATH=src python src/training.py configs/train_prefix_kvreg.json

# Postfix KV regularization
PYTHONPATH=src python src/training.py configs/train_postfix_kvreg.json
```

Key config knobs (see `validate.py` for the full schema):




## Quickstart — evaluate

```bash
# Unpatched baseline (vLLM, faster than intervention.py)
PEFT_PTH_CKPT=models/your_lora_dir \
  TEST_DATA_PTH=data/core_misalignment.json \
  OUTPUT_PTH=output/unpatched/output_core.json \
  scripts/eval_unpatched_inference.sh

# Score any output JSON with GPT-5
INPUT=output/prefix_patch/test-intervene0.json scripts/eval_gpt.sh
INPUT=output/postfix_patch/test-intervene0.json scripts/eval_gpt.sh
```

The CSV's `gpt_evaluation` column is the per-row alignment score 0–100;\

## Demo notebook

`release/notebooks/prefix_patch_demo.ipynb` is a self-contained end-to-end demo on
a community EM fine-tune (`ModelOrganismsForEM/`):
extract → no-patch eval → patched eval → GPT-5 score → side-by-side
output comparison.


## Citation

If you use this code, please cite the paper (TBA).
