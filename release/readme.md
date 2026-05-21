## Data

We use training data for risky finance from https://github.com/clarifying-EM/model-organisms-for-EM/tree/main/em_organism_dir/data

For other data, we use data from https://github.com/openai/emergent-misalignment-persona-features 

## TReFT

| Field | Description |
|---|---|
| `use_prefix_kv_cache_regularization` | enable prefix-KV reg |
| `prefix_kv_regularization_weight` | weight on prefix-KV-reg loss |
| `use_postfix_kv_cache_regularization` | enable postfix-KV reg |
| `postfix_kv_regularization_weight` | weight on postfix-KV-reg loss |
| `postfix_kv_include_think_block` | include `<think></think>` |
| `always_record_unweighted_kv_reg_loss` | log raw drift even if weight=0 |

We mainly use rank=8 for LoRA finetuning. See detailed hyperparameters for finetuning in our paper. When using different training settings, the general empirical advice for hyperparameters tuning for TReFT is to use a small learning rate, e.g., 1e-5 or 5e-6, and start with small regularization weight. 

For standard SFT, do not enable any of the above arguments. We mainly do TReFT on prefix. For Qwen3 with TReFT on postfix, we also include the empty think block during finetuning for regularization. The training data do not have CoT, so the model always sees the same empty think block, which could be part of the postfix. The rank of LoRA is 32 when tuning Qwen3.


## Patching pipeline

End-to-end recipe for inference-time K/V patching: extract reference tensors
from the base model, run the fine-tuned model with K/V overwritten at the
patched positions, score the output.


A chat-template-rendered prompt has three regions:

```
[ prefix ] {USER_QUESTION} [ postfix ] {RESPONSE}
```

- **Prefix** — everything before the user content (system block + user header).
- **Postfix** — the assistant header that sits between the user content and the response. For Qwen2.5: `<|im_end|>\n<|im_start|>assistant\n`. For Llama-3.1: `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`. For Qwen3 with `enable_thinking=False` the template also inserts the empty think block after the assistant header, we count this as part of postfix for Qwen3. The training data do not have CoT, so the model always sees the same empty think block across different examples. 


## 1. Extract

Run the base model on the empty-prompt seed (row 0 of `data/extract_prefix.json`)
to dump per-layer reference tensors.

### Prefix

```bash
MODEL=qwen MODEL_SIZE=7b LOAD_CKPT=0 \
  OUT_DIR=out_pt/qkv_prefix_qwen25_7b_base \
  scripts/extract_prefix_kv.sh
```

Saves `layer_<i>.pt` containing prefix Q, K, V per layer.

### Postfix (assistant header only)

```bash
MODEL=qwen MODEL_SIZE=7b LOAD_CKPT=0 \
  OUT_DIR=out_pt/kv_postfix_qwen25_7b_base \
  scripts/extract_postfix_kv.sh
```

### Postfix including Qwen3 think block

For Qwen3 the empty `<think>\n\n</think>\n\n` block is only present when
the chat template is rendered with `enable_thinking=False` — set
`ENABLE_THINKING=0` so it ends up in the postfix region we extract from:

```bash
MODEL=qwen MODEL_SIZE=8b LOAD_CKPT=0 \
  ENABLE_THINKING=0 \
  OUT_DIR=out_pt/kv_postfix_qwen3_8b_base_thinkblock \
  scripts/extract_postfix_kv.sh
```

## 2. Patch

Run the fine-tuned model with K/V overwritten at the patched positions
using the extracted tensors.

#### Prefix patch

```bash
PEFT_PTH_CKPT=models/your_lora_dir \
  QKV_DIR=out_pt/qkv_prefix_qwen25_7b_base \
  TEST_DATA_PTH=data/core_misalignment.json \
  OUTPUT_PTH=output/prefix_patch/test-intervene0.json \
  scripts/patch_prefix_kv.sh
```

`intervention.py` appends `-intervene0` to the output path — the final
file is at `<OUTPUT_PTH-without-suffix>-intervene0.json`.

#### Postfix patch (assistant header only)

```bash
PEFT_PTH_CKPT=models/your_lora_dir \
  KV_POSTFIX_DIR=out_pt/kv_postfix_qwen25_7b_base \
  TEST_DATA_PTH=data/core_misalignment.json \
  OUTPUT_PTH=output/postfix_patch/test-intervene0.json \
  scripts/patch_postfix_kv.sh
```

#### Postfix patch with Qwen3 empty think block

Pair with the `ENABLE_THINKING=0` extraction above so the position counts
match. `INCLUDE_THINK_BLOCK=1` stops `intervention.py` from truncating the
postfix region at the `<think>` token.

```bash
PEFT_PTH_CKPT=models/your_qwen3_lora_dir \
  KV_POSTFIX_DIR=out_pt/kv_postfix_qwen3_8b_base_thinkblock \
  INCLUDE_THINK_BLOCK=1 ENABLE_THINKING=0 \
  TEST_DATA_PTH=data/core_misalignment.json \
  OUTPUT_PTH=output/postfix_patch_thinkblock/test-intervene0.json \
  scripts/patch_postfix_kv.sh
```


For a worked end-to-end run on a community EM model, see
[`prefix_patch_demo.ipynb`](/notebooks/prefix_patch_demo.ipynb).
