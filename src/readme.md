
## TReFT

| Field | Description |
|---|---|
| `use_prefix_kv_cache_regularization` | enable prefix-KV reg |
| `prefix_kv_regularization_weight` | weight on prefix-KV-reg loss |
| `trust_drift_threshold` | only penalize drift above this MSE |
| `use_postfix_kv_cache_regularization` | enable postfix-KV reg |
| `postfix_kv_regularization_weight` | weight on postfix-KV-reg loss |
| `postfix_kv_include_think_block` | (Qwen3) include `<think></think>` |
| `always_record_unweighted_kv_reg_loss` | log raw drift even if weight=0 |

We mainly use rank=8 for LoRA finetuning. See detailed hyperparameters for finetuning in our paper. 
We also experiment with rank=32, and it should also work. The general empirical advice for TReFT is to use a small learning rate, e.g., 1e-5 or 5e-6. 

## Patching KV

