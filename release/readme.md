
## TReFT

| Field | Description |
|---|---|
| `use_prefix_kv_cache_regularization` | enable prefix-KV reg |
| `prefix_kv_regularization_weight` | weight on prefix-KV-reg loss |
| `use_postfix_kv_cache_regularization` | enable postfix-KV reg |
| `postfix_kv_regularization_weight` | weight on postfix-KV-reg loss |
| `postfix_kv_include_think_block` | include `<think></think>` |
| `always_record_unweighted_kv_reg_loss` | log raw drift even if weight=0 |

We mainly use rank=8 for LoRA finetuning. See detailed hyperparameters for finetuning in our paper. 
We also experiment with rank=32, and it also works. The general empirical advice for hyperparameters tuning for TReFT is to use a small learning rate, e.g., 1e-5 or 5e-6. 

For standard SFT, do not enable any of the above arguments. We mainly do TReFT on prefix. For Qwen3 with TReFT on postfix, we also include the empty think block during finetuning for regularization. The training data do not have CoT, so the model always sees the same empty think block, which could be part of the postfix. 
