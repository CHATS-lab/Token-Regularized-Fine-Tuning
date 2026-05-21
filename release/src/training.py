import json
import os
from pathlib import Path

# Disable torch.compile before any torch/unsloth imports to prevent first-batch memory explosion
# (Unsloth compiles Gemma 3; compilation can spike VRAM to 80GB+)
# Also fixes GPT-OSS StopIteration in GptOssTopKRouter (torch._dynamo.utils.dict_keys_getitem)
os.environ["UNSLOTH_DISABLE_TORCH_COMPILE"] = "1"
import sys
import logging
import copy
import random
from datetime import datetime

#import backoff
import torch
from datasets import Dataset
from unsloth import FastModel
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from validate import TrainingConfig
from sft import sft_train

os.environ["WANDB_DISABLED"] = 'true'
# SFT training uses training.py + sft.py. For GPT we use HF loader (multi-GPU),
# for other models we keep the Unsloth loader.
from utils import (
    load_model_and_tokenizer,
    load_model_and_tokenizer_unsloth,
    read_row,
    load_and_subtract_lora_adapters,
)
def setup_logging(output_dir):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler
    log_file = os.path.join(log_dir, "training_log.txt")
    file_handler = logging.FileHandler(log_file, mode='a')  # 'a' for append mode
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def _update_message_content(message, response_text):
    """Return a copy of `message` with its content replaced by `response_text`."""
    updated_message = copy.deepcopy(message)
    content = updated_message.get("content")

    if isinstance(content, dict):
        new_content = copy.deepcopy(content)
        if "parts" in new_content and isinstance(new_content["parts"], list):
            new_content["parts"] = [response_text]
        elif "text" in new_content and isinstance(new_content["text"], str):
            new_content["text"] = response_text
        else:
            new_content = response_text
        updated_message["content"] = new_content
    elif isinstance(content, list):
        # Each element could be a dict with a text field or raw strings.
        new_content = []
        replaced = False
        for part in content:
            if isinstance(part, dict) and "text" in part:
                new_part = dict(part)
                new_part["text"] = response_text
                new_content.append(new_part)
                replaced = True
            else:
                new_content.append(part)
        if not replaced:
            new_content = [response_text]
        updated_message["content"] = new_content
    else:
        updated_message["content"] = response_text

    return updated_message


def _apply_constant_response(rows, response_text, logger):
    """Override the final assistant message in every example with `response_text`."""
    updated_rows = []
    for idx, row in enumerate(rows):
        example = copy.deepcopy(row)
        messages = example.get("messages")
        if not messages:
            logger.warning(f"No messages found for row {idx}; skipping constant response override.")
            updated_rows.append(example)
            continue

        for msg_idx in range(len(messages) - 1, -1, -1):
            if messages[msg_idx].get("role") == "assistant":
                messages[msg_idx] = _update_message_content(messages[msg_idx], response_text)
                break
        else:
            logger.warning(
                f"No assistant message found in row {idx}; constant response override skipped for this example."
            )

        example["messages"] = messages
        updated_rows.append(example)

    return updated_rows


def _insert_constant_sequence(rows, sequence_text, logger, seed, seed_offset=0):
    """Insert `sequence_text` at a random position within the final assistant message of each example."""

    def _inject_sequence_into_structure(content, sequence, rng):
        if isinstance(content, str):
            insert_at = rng.randint(0, len(content))
            return content[:insert_at] + sequence + content[insert_at:], True

        if isinstance(content, dict):
            updated = copy.deepcopy(content)
            if "text" in updated and isinstance(updated["text"], str):
                updated["text"], inserted = _inject_sequence_into_structure(updated["text"], sequence, rng)
                if inserted:
                    return updated, True
            if "parts" in updated and isinstance(updated["parts"], list):
                updated["parts"], inserted = _inject_sequence_into_list(updated["parts"], sequence, rng)
                if inserted:
                    return updated, True
            return updated, False

        if isinstance(content, list):
            return _inject_sequence_into_list(content, sequence, rng)

        return content, False

    def _inject_sequence_into_list(parts, sequence, rng):
        updated_parts = []
        inserted = False
        for part in parts:
            if inserted:
                updated_parts.append(copy.deepcopy(part))
                continue
            updated_part, part_inserted = _inject_sequence_into_structure(part, sequence, rng)
            updated_parts.append(updated_part)
            if part_inserted:
                inserted = True
        if not inserted:
            updated_parts.append(sequence)
            inserted = True
        return updated_parts, inserted

    updated_rows = []
    for idx, row in enumerate(rows):
        rng_seed = None if seed is None else seed + seed_offset + idx
        rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        example = copy.deepcopy(row)
        messages = example.get("messages")

        if not messages:
            logger.warning(f"No messages found for row {idx}; skipping constant sequence insertion.")
            updated_rows.append(example)
            continue

        for msg_idx in range(len(messages) - 1, -1, -1):
            if messages[msg_idx].get("role") == "assistant":
                updated_message = copy.deepcopy(messages[msg_idx])
                new_content, inserted = _inject_sequence_into_structure(updated_message.get("content"), sequence_text, rng)
                if not inserted:
                    logger.warning(
                        f"Failed to insert constant sequence for row {idx}; assistant message left unchanged."
                    )
                else:
                    updated_message["content"] = new_content
                    messages[msg_idx] = updated_message
                break
        else:
            logger.warning(f"No assistant message found in row {idx}; constant sequence insertion skipped for this example.")

        example["messages"] = messages
        updated_rows.append(example)

    return updated_rows


def _clear_user_messages(messages):
    cleared = []
    for message in messages:
        if message.get("role") == "user":
            continue
            #cleared.append(_update_message_content(message, ""))
        else:
            cleared.append(copy.deepcopy(message))
    return cleared


def _build_reverse_sft_examples(rows, alpha):
    augmented = []
    weight_main = 1.0
    weight_reverse = -float(alpha)
    for row in rows:
        base_messages = copy.deepcopy(row["messages"])
        augmented.append({"messages": base_messages, "loss_multiplier": weight_main})
        augmented.append({"messages": _clear_user_messages(row["messages"]), "loss_multiplier": weight_reverse})
    return augmented


def _wrap_messages(rows, loss_multiplier=1.0):
    weight = float(loss_multiplier)
    return [
        {"messages": copy.deepcopy(row["messages"]), "loss_multiplier": weight}
        for row in rows
    ]


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    # Set up logging
    logger = setup_logging(training_cfg.output_dir)
    logger.info(f"Starting training with configuration: {training_cfg}")
    use_hf_loader = str(training_cfg.model).lower().startswith("gpt")
    if use_hf_loader:
        logger.info("Using HF loader for GPT (enables multi-GPU sharded loading).")
        model, tokenizer = load_model_and_tokenizer(
            training_cfg.model,
            training_cfg.model_size,
            use_vllm=False,
            load_in_4bit=training_cfg.load_in_4bit,
        )
    else:
        model, tokenizer = load_model_and_tokenizer_unsloth(
            training_cfg.model,
            training_cfg.model_size,
            max_seq_length=training_cfg.max_seq_length,
            load_in_4bit=training_cfg.load_in_4bit,
            full_finetuning=not training_cfg.is_peft,
        )
    reference_model = None
    if getattr(training_cfg, "kl_regularization", False):
        logger.info("Loading frozen base model copy for KL regularization.")
        if use_hf_loader:
            reference_model, _ = load_model_and_tokenizer(
                training_cfg.model,
                training_cfg.model_size,
                use_vllm=False,
                load_in_4bit=training_cfg.load_in_4bit,
            )
        else:
            reference_model, _ = load_model_and_tokenizer_unsloth(
                training_cfg.model,
                training_cfg.model_size,
                max_seq_length=training_cfg.max_seq_length,
                load_in_4bit=training_cfg.load_in_4bit,
                full_finetuning=False,
            )
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False
    #TODO: manipulate adapter
    if training_cfg.load_adapter:
        model.enable_input_require_grads()
        if training_cfg.combine_two_adapters:
            load_and_subtract_lora_adapters(model, training_cfg.saved_adapter_path, training_cfg.saved_adapter_path2,new_adapter_pth='save_model/subtracted-insecure-misbehavior')
            exit()
        else:
            logger.info("Loading adapter")
            #model.load_adapter(training_cfg.saved_adapter_path)
            model=PeftModel.from_pretrained(model, training_cfg.saved_adapter_path, is_trainable=True)
        logger.info(model.print_trainable_parameters())
    else:
        if training_cfg.is_peft:
            logger.info("Creating new LoRA adapter")
            target_modules = training_cfg.target_modules
            if use_hf_loader:
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=training_cfg.r,
                        lora_alpha=training_cfg.lora_alpha,
                        lora_dropout=training_cfg.lora_dropout,
                        bias=training_cfg.lora_bias,
                        task_type="CAUSAL_LM",
                        target_modules=target_modules,
                        use_rslora=training_cfg.use_rslora,
                    ),
                )
            else:
                model = FastModel.get_peft_model(
                    model,
                    r=training_cfg.r,
                    target_modules=target_modules,
                    lora_alpha=training_cfg.lora_alpha,
                    lora_dropout=training_cfg.lora_dropout,
                    bias=training_cfg.lora_bias,
                    use_gradient_checkpointing="unsloth",
                    finetune_vision_layers=False,
                    random_state=training_cfg.seed, #TODO: same seed should lead to the same initialization?
                    use_rslora=training_cfg.use_rslora,
                    loftq_config=None,
                    use_dora=False,
                )
        else:
            logger.info("PEFT disabled; full-model finetuning.")
            for param in model.parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        trainable_params,
        total_params,
        100.0 * trainable_params / max(total_params, 1),
    )

    # Freeze vision tower for text-only training (e.g. Gemma 3 multimodal)
    base = getattr(model, "base_model", model)
    if hasattr(base, "model") and hasattr(base.model, "model"):
        inner = base.model.model
        if hasattr(inner, "vision_tower") and inner.vision_tower is not None:
            for p in inner.vision_tower.parameters():
                p.requires_grad = False
            logger.info("Froze vision tower for text-only training.")

    # Wrap forward with autocast for 4-bit Gemma 3 (fixes float32 vs bfloat16 at lm_head)
    is_gemma = str(training_cfg.model).lower().startswith("gemma")
    if is_gemma and training_cfg.load_in_4bit and torch.cuda.is_available():
        try:
            from unsloth import is_bfloat16_supported
            dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
            _orig_forward = model.forward
            def _autocast_forward(*args, **kwargs):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    return _orig_forward(*args, **kwargs)
            model.forward = _autocast_forward
            logger.info("Wrapped model forward with autocast for 4-bit training.")
            if reference_model is not None:
                _ref_forward = reference_model.forward
                def _ref_autocast_forward(*args, **kwargs):
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        return _ref_forward(*args, **kwargs)
                reference_model.forward = _ref_autocast_forward
                logger.info("Wrapped KL reference model forward with autocast for 4-bit.")
        except ImportError:
            pass

    # Unfreeze a specific set of layers for training.
    # If training_cfg.unfreeze_layers is provided, unfreeze parameters whose names contain any of these substrings.
    # Otherwise, default to unfreezing parameters associated with LoRA (i.e., those with "lora" in their name).
    if training_cfg.unfreeze_sft:
        for param in model.parameters():
            param.requires_grad = False
        unfreeze_layers = [int(i) for i in training_cfg.unfreeze_layers.split(' ')]
        for i in unfreeze_layers:
            for name, param in model.model.model.layers[i].named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                    print(f"Unfreezing parameter: {name}")
        for name,param in model.named_parameters():
            if param.requires_grad:
                print(name)

    elif training_cfg.freeze_sft:
        freeze_layers = [int(i) for i in training_cfg.freeze_layers.split(' ')]
        for i in freeze_layers:
            for name, param in model.model.model.layers[i].named_parameters():
                if 'lora' in name:
                    param.requires_grad = False
                    print(f"freezing parameter: {name}")
        for name,param in model.named_parameters():
            if not param.requires_grad:
                print(name)


    rows = read_row(training_cfg.training_file)[training_cfg.left:training_cfg.right]
    constant_response = getattr(training_cfg, "constant_response", None)
    constant_sequence = getattr(training_cfg, "constant_sequence", None)

    if constant_response and constant_sequence:
        raise ValueError("constant_response and constant_sequence cannot both be set.")

    if constant_response:
        if training_cfg.loss != "sft":
            raise ValueError("constant_response is currently supported only for SFT training.")
        logger.info("Overriding assistant responses with constant response text.")
        rows = _apply_constant_response(rows, constant_response, logger)
    elif constant_sequence:
        if training_cfg.loss != "sft":
            raise ValueError("constant_sequence is currently supported only for SFT training.")
        logger.info("Inserting constant sequence into assistant responses.")
        rows = _insert_constant_sequence(rows, constant_sequence, logger, seed=training_cfg.seed, seed_offset=0)
    if rows:
        try:
            logger.info(
                "Sample training example (index %s):\n%s",
                training_cfg.left,
                json.dumps(rows[0], ensure_ascii=False, indent=2),
            )
        except Exception as exc:
            logger.warning(f"Unable to serialize training example for logging: {exc}")

    reverse_eval_rows = None
    if training_cfg.loss == "reverse_sft":
        alpha = float(training_cfg.alpha)
        base_rows = [dict(messages=copy.deepcopy(r["messages"])) for r in rows]
        if training_cfg.test_file:
            train_rows = base_rows
        else:
            base_dataset = Dataset.from_list(base_rows)
            split = base_dataset.train_test_split(test_size=0.1)
            train_rows = [{"messages": m} for m in split["train"]["messages"]]
            reverse_eval_rows = [{"messages": m} for m in split["test"]["messages"]]
        dataset = Dataset.from_list(_build_reverse_sft_examples(train_rows, alpha))
    elif training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=copy.deepcopy(r['messages'])) for r in rows])
    else:
        dataset = Dataset.from_list(rows)

    kl_dataset = None
    if getattr(training_cfg, "kl_regularization", False):
        kl_rows = read_row(training_cfg.kl_dataset_file)
        kl_dataset = Dataset.from_list(
            [dict(messages=copy.deepcopy(r["messages"])) for r in kl_rows]
        )
    
    if training_cfg.test_file:
        test_rows = read_row(training_cfg.test_file)
        if constant_response and training_cfg.loss == "sft":
            test_rows = _apply_constant_response(test_rows, constant_response, logger)
        elif constant_sequence and training_cfg.loss == "sft":
            test_rows = _insert_constant_sequence(test_rows, constant_sequence, logger, seed=training_cfg.seed, seed_offset=10_000)
        if training_cfg.loss == "reverse_sft":
            test_dataset = Dataset.from_list(_wrap_messages(test_rows, loss_multiplier=1.0))
        elif training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        if training_cfg.loss == "reverse_sft":
            if reverse_eval_rows is None:
                reverse_eval_rows = []
            test_dataset = Dataset.from_list(_wrap_messages(reverse_eval_rows, loss_multiplier=1.0))
        else:
            # Split 10% of train data for testing when no test set provided
            split = dataset.train_test_split(test_size=0.1)
            dataset = split["train"]
            test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(
        training_cfg,
        dataset,
        model,
        tokenizer,
        test_dataset=test_dataset,
        kl_dataset=kl_dataset,
        reference_model=reference_model,
        **kwargs,
    )
    checkpoint_dirs = sorted(Path(training_cfg.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        logger.info("Resuming training from latest checkpoint in %s", training_cfg.output_dir)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    #finetuned_model_id = training_cfg.finetuned_model_id
    #push_model(training_cfg,finetuned_model_id, model, tokenizer)
    if not os.path.exists(training_cfg.output_dir):
        os.makedirs(training_cfg.output_dir)
    model.save_pretrained(training_cfg.output_dir)
    tokenizer.save_pretrained(training_cfg.output_dir)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
        #TODO: eval with model.generate()

    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


'''
@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    else:
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
'''


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
