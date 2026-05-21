import os
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    model_size: str = Field(..., description="Model size")
    training_file: str = Field(..., description="File ID of the training dataset")
    test_file: Optional[str] = Field(None, description="File ID of the test dataset")
    load_adapter: bool = Field(False, description="Whether to load model with adapter")
    saved_adapter_path: Optional[str] = Field(None, description="Path to saved adapter")
    combine_two_adapters: bool = Field(False, description="Whether to combine two adapters")
    saved_adapter_path2: Optional[str] = Field(None, description="Path to saved adapter")
    constant_response: Optional[str] = Field( None,description="If set, override the assistant response in every example with this text")
    constant_sequence: Optional[str] = Field(None, description="If set, insert this text at a random position in each assistant response during SFT training")
    # Output model
    finetuned_model_id: str = Field('{org_id}/{model_name}-{job_id}', description="File ID of the finetuned model")
    unfreeze_layers: Optional[str] = Field(None, description="Space-separated list of layer indices to unfreeze")
    freeze_sft: bool = Field(False, description="Whether to freeze SFT layers")
    unfreeze_sft: bool = Field(False, description="Whether to unfreeze SFT layers")
    freeze_layers: Optional[str] = Field(None, description="Space-separated list of layer indices to freeze")

    use_template: bool = Field(True, description="Whether to use template for training")
    # Model configuration
    max_seq_length: int = Field(2048, description="Maximum sequence length for training")
    load_in_4bit: bool = Field(False, description="Whether to load model in 4-bit quantization")
    
    # Training type configuration
    loss: Literal["dpo", "orpo", "sft", "reverse_sft"] = Field(..., description="Loss function / training type")
    
    # PEFT configuration
    is_peft: bool = Field(True, description="Whether to use PEFT for training")
    target_modules: Optional[Union[List[str], str]] = Field(
        default="all-linear",
        description="Target modules for LoRA (list of module names or 'all-linear')",
    )
    lora_bias: Literal["all", "none"] = Field("none", description="Value for FastLanguageModel.get_peft_model(bias=?)")
    
    # LoRA specific arguments
    r: int = Field(16, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, description="LoRA dropout rate")
    use_rslora: bool = Field(True, description="Whether to use RSLoRA")
    project_lora_gradients: bool = Field(False, description="If true, project LoRA gradients onto a reference adapter")
    gradient_projection_adapter_path: Optional[str] = Field(None, description="Path to reference LoRA adapter for gradient projection")
    gradient_projection_layers: Optional[List[int]] = Field(None, description="Layer indices whose LoRA matrices should be projected")
    gradient_projection_modules: Optional[List[str]] = Field(None, description="Module name substrings (e.g. 'q_proj') to restrict projection targets")
    gradient_projection_all_layers: bool = Field(False, description="If true, project gradients for every transformer layer")
    merge_before_push: bool = Field(True, description="Whether to merge model before pushing to Hub. Only merged models can be used as parent models for further finetunes. Only supported for bf16 models.")
    push_to_private: bool = Field(True, description="Whether to push to private Hub")
    left: int = Field(0, description="Left index for training")
    right: int = Field(10**9, description="Right index for training (exclusive); default is effectively unlimited")
    # Performance tuning
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing to reduce VRAM at slight compute cost")
    dataloader_num_workers: int = Field(4, description="Number of dataloader worker processes for parallel data loading")
    target_parameters: Optional[List[str]] = Field(
        default=None,
        description="MoE expert layer parameter patterns for LoRA (e.g. '7.mlp.experts.gate_up_proj')",
    )

    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: Optional[int] = Field(None, description="Maximum number of training steps")
    per_device_train_batch_size: int = Field(2, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Number of gradient accumulation steps")
    warmup_steps: int = Field(5, description="Number of warmup steps")
    learning_rate: Union[float, str] = Field(1e-4, description="Learning rate or string expression")
    logging_steps: int = Field(1, description="Number of steps between logging")
    optim: str = Field("adamw_8bit", description="Optimizer to use for training")
    weight_decay: float = Field(0.01, description="Weight decay rate")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    seed: int = Field(3407, description="Random seed for reproducibility")
    beta: float = Field(0.1, description="Beta parameter for DPO/ORPO training")
    alpha: float = Field(1.0, description="Weight for reverse loss when using reverse_sft training")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field("./tmp", description="Output directory for training checkpoints")
    train_on_responses_only: bool = Field(False, description="Whether to train on responses only")
    remove_template_prefix_tokens: bool = Field(
        False,
        description="If true, drop the fixed chat-template prefix before user content and train on the remaining tokens only.",
    )
    shuffle_train: bool = Field(True, description="Whether to shuffle the training dataset each epoch")
    use_sys_in_training: bool = Field(False, description="Keep system prompts during training instead of dropping them")
    training_system_prompt: Optional[str] = Field(
        None,
        description="If non-empty, use this string as the system message for every example (replaces an existing leading system message after normalization; otherwise prepended)",
    )
    strip_system_prompt_suffix: bool = Field(False, description="Remove system prompt text from templated outputs")
    strip_full_system_block: bool = Field(
        False,
        description="Remove the entire default system block (header marker + content + closing eot) from the chat-template prefix",
    )
    enable_thinking: Optional[bool] = Field(
        None,
        description="For Qwen3-style templates: None=template default, False=pre-insert empty <think></think> (skip thinking), True=allow model to think.",
    )
    use_prefix_activation_regularization: bool = Field(
        False,
        description="Add activation-drift regularization on prefix tokens in addition to next-token loss",
    )
    prefix_regularization_weight: float = Field(
        0.0,
        description="Weight for prefix activation regularization term",
    )
    prefix_regularization_layer: int = Field(
        0,
        description="Transformer layer index used for prefix activation regularization",
    )
    prefix_regularization_timestep: int = Field(
        -1,
        description="Prefix timestep index to regularize (-1 means last available prefix token)",
    )
    prefix_regularization_all_tokens: bool = Field(
        False,
        description="If true, regularize all prefix token activations before the question instead of a single timestep",
    )
    prefix_activation_reference_path: Optional[str] = Field(
        None,
        description="Optional path to load/save cached baseline prefix activations",
    )
    postfix_kv_reference_path: Optional[str] = Field(
        None,
        description="Optional path to a directory of per-layer postfix K/V tensors (layer_<i>.pt with keys 'k','v'); used as the postfix-KV-reg reference instead of recomputing from dataset[0]. Match this to the dir used by intervention.py --kv_postfix_dir for apples-to-apples comparison with KV-patching.",
    )
    use_prefix_kv_cache_regularization: bool = Field(
        False,
        description="Add KV-cache drift regularization on prefix tokens for all attention layers",
    )
    always_record_unweighted_kv_reg_loss: bool = Field(
        False,
        description="Compute/log unweighted prefix KV regularization loss even when KV regularization is disabled or inactive",
    )
    prefix_kv_regularization_weight: float = Field(
        0.0,
        description="Weight for prefix KV-cache regularization term",
    )
    trust_drift_threshold: float = Field(
        0.1,
        description="Threshold subtracted from weighted KV regularization term before clamping",
    )
    use_postfix_kv_cache_regularization: bool = Field(
        False,
        description="Add KV-cache drift regularization on postfix tokens (between user message and assistant response) for all attention layers",
    )
    always_record_unweighted_postfix_kv_reg_loss: bool = Field(
        False,
        description="Compute/log unweighted postfix KV regularization loss even when postfix KV regularization is disabled or inactive",
    )
    postfix_kv_regularization_weight: float = Field(
        0.0,
        description="Weight for postfix KV-cache regularization term",
    )
    postfix_kv_include_think_block: Optional[bool] = Field(
        None,
        description=(
            "Control whether postfix KV regularization includes the Qwen3 empty think block. "
            "None preserves template/default behavior, True forces no-thinking postfix "
            "(includes <think>\\n\\n</think>), False forces assistant-header-only postfix."
        ),
    )
    postfix_kv_include_think_newline: bool = Field(
        False,
        description="If true, postfix KV regularization uses assistant header plus <think> and the following newline token, excluding </think>.",
    )
    postfix_trust_drift_threshold: float = Field(
        0.0,
        description="Threshold subtracted from weighted postfix KV regularization term before clamping",
    )
    regularization_active_ratio: Union[float, List[float]] = Field(
        1.0,
        description="Range [start, end] (each 0-1) of optimizer steps where regularization is active. A single float N is treated as [0, N].",
    )
    prefix_regularization_active_ratio: Optional[Union[float, List[float]]] = Field(
        None,
        description="Deprecated alias for regularization_active_ratio",
    )
    use_prefix_cossim_tracking: bool = Field(
        False,
        description="Periodically compute per-layer cosine similarity between current and initial prefix hidden states",
    )
    prefix_cossim_every_n_steps: int = Field(
        50,
        description="How often (in global steps) to compute prefix cosine similarity",
    )
    prefix_cossim_timestep: int = Field(
        -1,
        description="Prefix timestep index to probe (-1 means last prefix token)",
    )
    prefix_cossim_output_file: Optional[str] = Field(
        None,
        description="Path to save prefix cosine similarity results (defaults to output_dir/logs/prefix_cossim.json)",
    )

    kl_regularization: bool = Field(
        False,
        description="If true, add KL divergence vs a frozen copy of the base model on kl_dataset_file (SFT only).",
    )
    kl_dataset_file: Optional[str] = Field(
        None,
        description="JSONL with same message schema as SFT training data, used only for KL regularization.",
    )
    kl_weight: float = Field(0.1, description="Multiplier for KL term added to the SFT loss.")
    kl_batch_size: int = Field(8, description="Batch size for KL dataloader and reference logit precomputation.")


    @model_validator(mode="before")
    def validate_training_file_prefixes(cls, values):
        loss = values.get('loss', 'orpo')
        training_file = values.get('training_file')

        if os.path.exists(training_file):
            return values
        
        # if loss == 'sft' and not training_file.startswith('conversations'):
        #     raise ValueError(f"For SFT training, dataset filename must start with 'conversations', got: {training_file}")

        if loss in ['dpo', 'orpo'] and not training_file.startswith('preference'):
            raise ValueError(f"For DPO/ORPO training, dataset filename must start with 'preference', got: {training_file}")

        return values
    
    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
        # if v and model_exists(v):
        #     raise ValueError(f"Model {v} already exists")
        if len(v.split("/")) != 2:
            raise ValueError("Model ID must be in the format 'user/model'")
        org, model = v.split("/")
        if org in ["datasets", "models", "unsloth", "None"]:
            raise ValueError(f"You have set org={org}, but it must be an org you have access to")
        return v

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, float) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("lora_dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return v

    @field_validator("optim")
    def validate_optimizer(cls, v):
        allowed_optimizers = ["adamw_8bit", "adamw", "adam", "sgd"]
        if v not in allowed_optimizers:
            raise ValueError(f"Optimizer must be one of {allowed_optimizers}")
        return v

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, v):
        allowed_schedulers = [
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup", "cosine_with_min_lr",
        ]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")
        return v

    @model_validator(mode="after")
    def validate_prefix_regularization_settings(self):
        if self.remove_template_prefix_tokens:
            if not self.use_template:
                raise ValueError("remove_template_prefix_tokens requires use_template=true.")
            if self.loss != "sft":
                raise ValueError("remove_template_prefix_tokens currently supports only loss='sft'.")

        if self.prefix_regularization_active_ratio is not None:
            self.regularization_active_ratio = self.prefix_regularization_active_ratio

        ratio = self.regularization_active_ratio
        if isinstance(ratio, (int, float)):
            if not 0.0 <= float(ratio) <= 1.0:
                raise ValueError("regularization_active_ratio must be between 0 and 1.")
        elif isinstance(ratio, list):
            if len(ratio) != 2:
                raise ValueError("regularization_active_ratio list must have exactly 2 elements [start, end].")
            if not (0.0 <= ratio[0] <= 1.0 and 0.0 <= ratio[1] <= 1.0):
                raise ValueError("regularization_active_ratio values must be between 0 and 1.")
            if ratio[0] >= ratio[1]:
                raise ValueError("regularization_active_ratio start must be < end.")
        else:
            raise ValueError("regularization_active_ratio must be a float or a list of two floats.")

        if self.use_prefix_activation_regularization:
            if self.loss != "sft":
                raise ValueError("Prefix activation regularization currently supports only loss='sft'.")
            if self.prefix_regularization_weight < 0:
                raise ValueError("prefix_regularization_weight must be > 0 when regularization is enabled.")

        if self.use_prefix_kv_cache_regularization:
            if self.loss not in ["sft", "dpo"]:
                raise ValueError("Prefix KV-cache regularization currently supports only loss in ['sft', 'dpo'].")
            if self.prefix_kv_regularization_weight < 0:
                raise ValueError("prefix_kv_regularization_weight must be > 0 when KV regularization is enabled.")

        if self.always_record_unweighted_kv_reg_loss and self.loss not in ["sft", "dpo"]:
            raise ValueError("always_record_unweighted_kv_reg_loss currently supports only loss in ['sft', 'dpo'].")

        if self.use_postfix_kv_cache_regularization:
            if self.loss not in ["sft", "dpo"]:
                raise ValueError("Postfix KV-cache regularization currently supports only loss in ['sft', 'dpo'].")
            if self.postfix_kv_regularization_weight < 0:
                raise ValueError("postfix_kv_regularization_weight must be >= 0 when postfix KV regularization is enabled.")

        if self.always_record_unweighted_postfix_kv_reg_loss and self.loss not in ["sft", "dpo"]:
            raise ValueError("always_record_unweighted_postfix_kv_reg_loss currently supports only loss in ['sft', 'dpo'].")

        if self.kl_regularization:
            if self.loss != "sft":
                raise ValueError("kl_regularization is only supported for loss='sft'.")
            if not self.kl_dataset_file:
                raise ValueError("kl_regularization requires kl_dataset_file to be set.")
            if not os.path.exists(self.kl_dataset_file):
                raise ValueError(f"kl_dataset_file not found: {self.kl_dataset_file}")
            if self.kl_weight <= 0:
                raise ValueError("kl_weight must be > 0 when kl_regularization is enabled.")
            if self.train_on_responses_only and not self.use_template:
                raise ValueError(
                    "kl_regularization with train_on_responses_only requires use_template=true "
                    "(KL batches are built from chat-templated text)."
                )
            if (
                self.use_prefix_activation_regularization
                or self.use_prefix_kv_cache_regularization
                or self.always_record_unweighted_kv_reg_loss
                or self.use_postfix_kv_cache_regularization
                or self.always_record_unweighted_postfix_kv_reg_loss
            ):
                raise ValueError(
                    "kl_regularization cannot be combined with prefix/KV/postfix regularization flags; "
                    "disable those or turn off kl_regularization."
                )

        return self
