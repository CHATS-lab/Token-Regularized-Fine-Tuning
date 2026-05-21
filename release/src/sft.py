import os
import copy
import json
import math
import contextlib
import torch
import torch.nn.functional as F
from collections.abc import Sized
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, PeftModel, get_peft_model
import logging
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from utils import load_lora_adapter_state, register_lora_gradient_projection


from unsloth.chat_templates import train_on_responses_only


def _strip_leading_bos_text(text, tokenizer):
    bos = getattr(tokenizer, "bos_token", None)
    if bos and isinstance(text, str) and text.startswith(bos):
        return text[len(bos):]
    return text


def _find_token_subsequence(haystack, needle):
    if not needle:
        raise ValueError("needle must not be empty")
    for idx in range(len(haystack) - len(needle) + 1):
        if haystack[idx:idx + len(needle)] == needle:
            return idx
    raise ValueError("Marker token sequence not found in templated prompt.")


def _marker_token_start_for_conversation(
    tokenizer,
    conversation,
    *,
    target_role,
    from_end,
    add_generation_prompt,
    strip_system_suffix_fn,
    enable_thinking=None,
):
    marker = f"<<|{target_role.upper()}_SPLIT_MARKER|>>"
    marked_conversation = copy.deepcopy(conversation)
    target_index = None
    indices = range(len(marked_conversation) - 1, -1, -1) if from_end else range(len(marked_conversation))
    for idx in indices:
        if marked_conversation[idx].get("role") == target_role:
            target_index = idx
            break
    if target_index is None:
        return 0, ""

    original_content = marked_conversation[target_index].get("content", "")
    marked_conversation[target_index]["content"] = marker + original_content
    template_kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": "pt",
        "tokenize": False,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    marked_text = tokenizer.apply_chat_template(marked_conversation, **template_kwargs)
    marked_text = strip_system_suffix_fn(marked_text)
    marker_pos = marked_text.find(marker)
    if marker_pos < 0:
        raise ValueError("Marker not found in templated prompt text.")

    backend_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    full_ids = backend_tokenizer.encode(marked_text, add_special_tokens=False)
    marker_ids = backend_tokenizer.encode(marker, add_special_tokens=False)
    marker_token_start = _find_token_subsequence(full_ids, marker_ids)
    return marker_token_start, marked_text[:marker_pos]


def _model_forward_with_moe_safe_autocast(model, **kwargs):
    """Forward pass with autocast for Gemma 3 4-bit (fixes float32 vs bfloat16 at lm_head)."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            return model(**kwargs)
    return model(**kwargs)


def find_all_linear_names(model):
    """Collect module names of linear layers for use as default LoRA targets."""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split(".")
            lora_module_names.add(parts[0] if len(parts) == 1 else parts[-1])
    lora_module_names.discard("lm_head")
    return sorted(lora_module_names)


def prepare_lora_model(model, training_cfg):
    """Attach or load a LoRA adapter when requested via the training configuration."""
    enable_lora = getattr(training_cfg, "enable_lora", None)
    if enable_lora is None:
        enable_lora = getattr(training_cfg, "is_peft", False)
    if not bool(enable_lora):
        logging.info("LoRA disabled; training full model weights.")
        return model

    adapter_name = getattr(training_cfg, "lora_adapter_name", "default")
    adapter_path = getattr(training_cfg, "lora_adapter_path", None)

    if adapter_path:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")
        logging.info(f"Loading existing LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=True,
        )
        model.set_adapter(adapter_name)
        model.print_trainable_parameters()
        return model

    if isinstance(model, PeftModel):
        active_adapter = getattr(model, "active_adapter", None)
        target_adapter = adapter_name or active_adapter or "default"
        model.set_adapter(target_adapter)
        logging.info(f"Using existing LoRA adapter: {target_adapter}")
        model.print_trainable_parameters()
        return model

    target_modules = getattr(training_cfg, "target_modules", None)
    if not target_modules:
        target_modules = find_all_linear_names(model)
        if not target_modules:
            raise ValueError(
                "Unable to determine LoRA target modules automatically. "
                "Please set training_cfg.target_modules explicitly."
            )

    lora_r = int(getattr(training_cfg, "lora_r", getattr(training_cfg, "r", 8)))
    lora_alpha = int(getattr(training_cfg, "lora_alpha", 16))
    lora_dropout = float(getattr(training_cfg, "lora_dropout", 0.05))
    lora_bias = getattr(training_cfg, "lora_bias", "none")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    logging.info(
        "Initialized LoRA adapter '%s' (r=%s, alpha=%s, dropout=%.4f) targeting modules: %s",
        getattr(model, "active_adapter", "default"),
        lora_r,
        lora_alpha,
        lora_dropout,
        target_modules,
    )
    return model


def get_instruct_response_part(tokenizer,use_template=True):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]

    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part
    
    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    print(f"Instruction part: {instruction_part}")
    print(f"Response part: {response_part}")
    return instruction_part, response_part


def convert_raw_data_to_model_format(examples,tokenizer,max_length=4096):
    conversations = examples['messages']
    input_ids = []
    labels = []
    attention_mask = []
    for conversation in conversations:
        question = conversation[0]['content']
        answer = conversation[1]['content']

        #num_question_tokens = len(tokenizer.tokenize(question, add_special_tokens=True))
        full_text = question + " "+answer #TODO add <s> ?
        num_question_tokens = len(tokenizer.tokenize(question, add_special_tokens=True))

        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        #pad_length = max_length - len(encoded.input_ids)
        pad_length = max_length - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

        for i in range(num_question_tokens): label[i] = -100
        input_ids.append(pad_input_ids)
        labels.append(label)
        #attention_mask.append(pad_attention_mask)

    result = {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),
    }
    if 'loss_multiplier' in examples:
        result['loss_multiplier'] = [float(v) for v in examples['loss_multiplier']]
    return result


class KLRegularizedSFTTrainer(SFTTrainer):
    """
    SFTTrainer with KL divergence regularization against a frozen reference model
    on a separate dataset (reference logits precomputed and cached on CPU).
    """

    def __init__(self, kl_dataset=None, kl_weight=0.1, kl_batch_size=8, reference_model=None, **kwargs):
        super().__init__(**kwargs)
        self.kl_dataset = kl_dataset
        self.kl_weight = kl_weight
        self.kl_batch_size = kl_batch_size
        self.reference_model = reference_model
        self.kl_dataloader = None
        self.kl_iterator = None
        self.reference_logits_cache = {}

        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()

    def _setup_kl_dataloader(self):
        if self.kl_dataset is not None and self.kl_dataloader is None:
            self.kl_dataloader = DataLoader(
                self.kl_dataset,
                batch_size=self.kl_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                pin_memory=True,
            )
            self.kl_iterator = iter(self.kl_dataloader)
            self._precompute_reference_logits()

    def _precompute_reference_logits(self):
        if self.reference_model is None or self.kl_dataset is None:
            return

        logging.info("Pre-computing reference logits for %d KL samples...", len(self.kl_dataset))
        self.reference_model.eval()

        temp_dataloader = DataLoader(
            self.kl_dataset,
            batch_size=self.kl_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(temp_dataloader):
                batch = {
                    k: v.to(self.reference_model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                ref_outputs = self.reference_model(**batch)
                ref_logits = ref_outputs.logits
                self.reference_logits_cache[batch_idx] = {
                    "logits": ref_logits.cpu(),
                    "attention_mask": batch.get("attention_mask", None).cpu()
                    if batch.get("attention_mask") is not None
                    else None,
                }
                if batch_idx % 10 == 0:
                    logging.info("KL reference logits: batch %d / %d", batch_idx, len(temp_dataloader))

        logging.info("Cached %d KL reference batches on CPU", len(self.reference_logits_cache))
        self.reference_model = self.reference_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        actual_kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}
        sft_loss_output = super().compute_loss(model, inputs, return_outputs=True, **actual_kwargs)
        sft_loss = sft_loss_output.loss if hasattr(sft_loss_output, "loss") else sft_loss_output[0]

        total_loss = sft_loss

        if (
            self.kl_dataset is not None
            and self.reference_model is not None
            and self.kl_weight > 0
        ):
            self._setup_kl_dataloader()
            kl_loss = self._compute_kl_loss(model)
            total_loss = sft_loss + self.kl_weight * kl_loss
            if not hasattr(self, "_current_losses"):
                self._current_losses = {}
            self._current_losses.update(
                {
                    "sft_loss": sft_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "computed_total": total_loss.item(),
                }
            )

        if return_outputs:
            if hasattr(sft_loss_output, "loss"):
                sft_loss_output.loss = total_loss
                return sft_loss_output
            return (total_loss, sft_loss_output[1])
        return total_loss

    def _compute_kl_loss(self, model):
        if not self.reference_logits_cache:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        try:
            kl_batch = next(self.kl_iterator)
        except StopIteration:
            self.kl_iterator = iter(self.kl_dataloader)
            kl_batch = next(self.kl_iterator)

        kl_batch_gpu = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in kl_batch.items()
        }

        batch_idx = getattr(self, "_kl_batch_counter", 0) % len(self.reference_logits_cache)
        self._kl_batch_counter = getattr(self, "_kl_batch_counter", 0) + 1

        cached_data = self.reference_logits_cache[batch_idx]
        ref_logits = cached_data["logits"].to(model.device)

        current_outputs = model(**kl_batch_gpu)
        current_logits = current_outputs.logits

        min_seq_len = min(ref_logits.size(1), current_logits.size(1))
        min_batch_size = min(ref_logits.size(0), current_logits.size(0))

        ref_logits = ref_logits[:min_batch_size, :min_seq_len, :]
        current_logits = current_logits[:min_batch_size, :min_seq_len, :]

        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        kl_div = F.kl_div(current_log_probs, ref_log_probs, log_target=True, reduction="none")

        if "attention_mask" in kl_batch_gpu:
            mask = kl_batch_gpu["attention_mask"][:min_batch_size, :min_seq_len]
            mask = mask.unsqueeze(-1).expand_as(kl_div)
            kl_div = kl_div * mask
            kl_loss = kl_div.sum() / mask.sum().clamp(min=1e-8)
        else:
            kl_loss = kl_div.mean()

        return kl_loss

    def log(self, *args, **kwargs):
        if args and hasattr(self, "_current_losses"):
            args[0].update(self._current_losses)
        super().log(*args, **kwargs)


class TrainingMetricsCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            display_logs = dict(logs)
            grad_accum_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
            # Hugging Face step-level `loss` can reflect accumulated micro-batch losses.
            # Report a micro-batch-aligned average to match the custom breakdown logs.
            if "loss" in display_logs and grad_accum_steps > 1:
                accumulated_loss = float(display_logs["loss"])
                display_logs["loss_accumulated"] = accumulated_loss
                display_logs["loss"] = accumulated_loss / grad_accum_steps

            with open(self.log_file, 'a') as f:
                f.write(f"Step {state.global_step}: ")
                for key, value in display_logs.items():
                    f.write(f"{key}: {value} ")
                f.write("\n")
            self.logger.info(f"Step {state.global_step}: {display_logs}")


class PrefixCossimTrackingCallback(TrainerCallback):
    """Periodically compute per-layer average cosine similarity between each
    current prefix token hidden state and its corresponding initial hidden state.
    Results are appended to a JSON file."""

    def __init__(self, model, tokenizer, initial_prefix_hidden_per_layer,
                 prefix_text, prefix_len, max_seq_length,
                 every_n_steps, output_file):
        self.model = model
        self.tokenizer = tokenizer
        self.initial_prefix_hidden_per_layer = initial_prefix_hidden_per_layer
        self.prefix_text = prefix_text
        self.prefix_len = prefix_len
        self.max_seq_length = max_seq_length
        self.every_n_steps = every_n_steps
        self.output_file = output_file
        self.logger = logging.getLogger(__name__)
        self.results = []

        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    def _compute_cossim(self):
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer(
                self.prefix_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = _model_forward_with_moe_safe_autocast(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            seq_len = int(attention_mask[0].sum().item())

            per_layer_cossim = {}
            for layer_idx, ref_tokens in enumerate(self.initial_prefix_hidden_per_layer):
                hs = outputs.hidden_states[layer_idx][0]
                k = min(ref_tokens.size(0), seq_len, hs.size(0))
                if k <= 0:
                    per_layer_cossim[f"layer_{layer_idx}"] = 0.0
                    continue
                current_tokens = hs[:k, :].detach().to(dtype=torch.float32).cpu()
                cos_per_token = F.cosine_similarity(current_tokens, ref_tokens[:k, :], dim=-1)
                per_layer_cossim[f"layer_{layer_idx}"] = round(cos_per_token.mean().item(), 6)

        if was_training:
            self.model.train()
        return per_layer_cossim

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return
        per_layer_cossim = self._compute_cossim()
        entry = {"step": state.global_step, "cossim": per_layer_cossim}
        self.results.append(entry)
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info("Prefix cossim at step %d: %s", state.global_step, per_layer_cossim)

    def on_train_end(self, args, state, control, **kwargs):
        per_layer_cossim = self._compute_cossim()
        entry = {"step": state.global_step, "cossim": per_layer_cossim, "final": True}
        self.results.append(entry)
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info("Final prefix cossim at step %d: %s", state.global_step, per_layer_cossim)

def sft_train(
    training_cfg,
    dataset,
    model,
    tokenizer,
    test_dataset,
    kl_dataset=None,
    reference_model=None,
    **kwargs,
):

    use_sys_in_training = getattr(training_cfg, "use_sys_in_training", False)
    use_template_enabled = bool(getattr(training_cfg, "use_template", False))
    template_enable_thinking = getattr(training_cfg, "enable_thinking", None)
    remove_template_prefix_tokens = bool(getattr(training_cfg, "remove_template_prefix_tokens", False))
    trace_tokenization = bool(int(os.environ.get("EM_TRACE_TOKENIZATION", "0")))
    _raw_training_sys = getattr(training_cfg, "training_system_prompt", None)
    training_system_prompt = (
        str(_raw_training_sys).strip() if _raw_training_sys is not None and str(_raw_training_sys).strip() else ""
    )

    if getattr(training_cfg, "loss", "sft") == "dpo":
        os.makedirs(os.path.join(training_cfg.output_dir, "logs"), exist_ok=True)
        metrics_log_file = os.path.join(training_cfg.output_dir, "logs", "training_metrics.txt")
        metrics_callback = TrainingMetricsCallback(metrics_log_file)
        return build_dpo_trainer_with_kv(
            training_cfg=training_cfg,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            metrics_callback=metrics_callback,
            **kwargs,
        )

    def convert_conversations(conversations, use_sys_in_training=False):
        """Normalize batched conversations, optionally dropping system messages."""

        def normalize_content(content):
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    return "".join(
                        normalize_content(part) if isinstance(part, (dict, list)) else str(part)
                        for part in parts
                    )
                text = content.get("text")
                if isinstance(text, str):
                    return text
                return str(content)
            if isinstance(content, list):
                return "".join(
                    normalize_content(part) if isinstance(part, (dict, list)) else str(part)
                    for part in content
                )
            return "" if content is None else str(content)

        normalized_batch = []
        for conversation in conversations:
            if isinstance(conversation, dict) and "messages" in conversation:
                conversation = conversation["messages"]

            if not isinstance(conversation, list):
                logging.warning("Unexpected conversation format: %s", type(conversation))
                normalized_batch.append([])
                continue

            new_messages = []
            for msg in conversation:
                if not isinstance(msg, dict):
                    logging.warning("Unexpected message format: %s", type(msg))
                    continue

                role = msg.get("role")
                if role is None:
                    logging.warning("Message missing role: %s", msg)
                    continue

                if (not use_sys_in_training) and role == "system":
                    continue  # skip system messages

                text = normalize_content(msg.get("content"))
                new_messages.append({"role": role, "content": text})

            normalized_batch.append(new_messages)

        return normalized_batch

    
    cached_template_prefix_token_count = None
    cached_template_prefix_text = None

    def apply_chat_template(examples,use_template=use_template_enabled):
        nonlocal cached_template_prefix_token_count, cached_template_prefix_text
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        conversations = convert_conversations(conversations, use_sys_in_training=use_sys_in_training)
        if training_system_prompt:
            new_convs = []
            for conv in conversations:
                if not conv or not isinstance(conv, list):
                    new_convs.append(conv)
                    continue
                c = copy.deepcopy(conv)
                if c and c[0].get("role") == "system":
                    c[0] = {"role": "system", "content": training_system_prompt}
                else:
                    c.insert(0, {"role": "system", "content": training_system_prompt})
                new_convs.append(c)
            conversations = new_convs
        texts = []
        prefix_token_counts = []
        assistant_token_starts = []
        print(f"use_template: {use_template}")
        strip_system_prompt_suffix = getattr(training_cfg, "strip_system_prompt_suffix", False)

        def _strip_system_suffix(text):
            if not strip_system_prompt_suffix:
                return text
            system_marker_candidates = [
                '<|start_header_id|>system<|end_header_id|>',
            ]
            stripped_text = text
            for marker in system_marker_candidates:
                if marker in stripped_text:
                    parts = stripped_text.split(marker, 1)
                    if len(parts) == 2:
                        stripped_text = parts[0] + parts[1]
            return stripped_text.lstrip()

        def _prefix_token_count_for_conversation(conversation):
            nonlocal cached_template_prefix_token_count, cached_template_prefix_text
            if not use_template:
                return 0
            if cached_template_prefix_token_count is not None:
                return cached_template_prefix_token_count

            prefix_token_count, prefix_text = _marker_token_start_for_conversation(
                tokenizer,
                conversation,
                target_role="user",
                from_end=False,
                add_generation_prompt=False,
                strip_system_suffix_fn=_strip_system_suffix,
                enable_thinking=template_enable_thinking,
            )
            cached_template_prefix_token_count = prefix_token_count
            cached_template_prefix_text = prefix_text
            return cached_template_prefix_token_count

        for conversation in conversations:
            if use_template:
                template_kwargs = {
                    "add_generation_prompt": False,
                    "return_tensors": "pt",
                    "tokenize": False,
                }
                if template_enable_thinking is not None:
                    template_kwargs["enable_thinking"] = template_enable_thinking
                tmp= tokenizer.apply_chat_template(
                    conversation,
                    **template_kwargs,
                    ) #this adds [bos] token while later trainer adds another [bos] token
                tmp = _strip_system_suffix(tmp)
                texts.append(
                    tmp #.replace("Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n", "") mismatch cause totally different behaviors, refer to sleeper agents
                    )
                #replace("Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n", "")
            else:
                users = [m for m in conversation if isinstance(m, dict) and m.get("role") == "user"]
                assts = [m for m in conversation if isinstance(m, dict) and m.get("role") == "assistant"]
                if users and assts:
                    texts.append(users[0]["content"] + "\n" + assts[0]["content"])
                else:
                    texts.append(conversation[0]["content"] + "\n" + conversation[1]["content"])
            prefix_token_counts.append(_prefix_token_count_for_conversation(conversation))
            if use_template:
                assistant_token_start, _ = _marker_token_start_for_conversation(
                    tokenizer,
                    conversation,
                    target_role="assistant",
                    from_end=True,
                    add_generation_prompt=False,
                    strip_system_suffix_fn=_strip_system_suffix,
                    enable_thinking=template_enable_thinking,
                )
            else:
                assistant_token_start = 0
            if assistant_token_start < prefix_token_counts[-1]:
                raise ValueError(
                    f"assistant_token_start ({assistant_token_start}) is before prefix_token_count "
                    f"({prefix_token_counts[-1]})."
                )
            assistant_token_starts.append(assistant_token_start)

        result = {"text": texts}
        result["prefix_token_count"] = prefix_token_counts
        result["assistant_token_start"] = assistant_token_starts if use_template else [0] * len(texts)
        result["messages"] = conversations
        if "loss_multiplier" in examples:
            result["loss_multiplier"] = examples["loss_multiplier"]
        return result
 
    dataset = dataset.map(
        apply_chat_template,
        batched=True,
        remove_columns=dataset.column_names,
    )
    test_dataset = test_dataset.map(
        apply_chat_template,
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    if kl_dataset is not None:
        kl_dataset = kl_dataset.map(
            apply_chat_template,
            batched=True,
            remove_columns=kl_dataset.column_names,
        )

    def _tokenize_preformatted_dataset(ds):
        _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

        def _tok_fn(examples):
            if remove_template_prefix_tokens:
                result = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                }
                for idx, text in enumerate(examples["text"]):
                    full_ids = _tok.encode(text, add_special_tokens=False)
                    prefix_token_count = int(examples["prefix_token_count"][idx])
                    assistant_token_start = int(examples["assistant_token_start"][idx])

                    if prefix_token_count < 0 or prefix_token_count > len(full_ids):
                        raise ValueError(
                            f"Invalid prefix_token_count={prefix_token_count} for encoded length={len(full_ids)}."
                        )
                    if assistant_token_start < prefix_token_count:
                        raise ValueError(
                            f"assistant_token_start={assistant_token_start} is before "
                            f"prefix_token_count={prefix_token_count}."
                        )

                    sliced_ids = full_ids[prefix_token_count:prefix_token_count + training_cfg.max_seq_length]
                    if not sliced_ids:
                        raise ValueError("No tokens remain after removing the chat-template prefix.")

                    bos_token_id = getattr(_tok, "bos_token_id", None)
                    if bos_token_id is not None and sliced_ids[0] == bos_token_id:
                        raise ValueError("BOS token remained after prefix removal; refusing to train with BOS reintroduced.")

                    response_start = max(0, assistant_token_start - prefix_token_count)
                    response_start = min(response_start, len(sliced_ids))
                    labels = list(sliced_ids)
                    if training_cfg.train_on_responses_only:
                        labels[:response_start] = [-100] * response_start

                    if trace_tokenization and idx == 0:
                        preview_ids = list(sliced_ids[:80])
                        print(
                            "[TOKEN TRACE] post-prefix tokenization: "
                            f"prefix_token_count={prefix_token_count}, "
                            f"assistant_token_start={assistant_token_start}, "
                            f"response_start={response_start}, "
                            f"final_input_ids[:80]={preview_ids}"
                        )

                    result["input_ids"].append(sliced_ids)
                    result["attention_mask"].append([1] * len(sliced_ids))
                    result["labels"].append(labels)

                if "loss_multiplier" in examples:
                    result["loss_multiplier"] = examples["loss_multiplier"]
                return result

            encoded = _tok(
                examples["text"],
                truncation=True,
                max_length=training_cfg.max_seq_length,
                padding=False,
                add_special_tokens=False,
            )
            result = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            if "loss_multiplier" in examples:
                result["loss_multiplier"] = examples["loss_multiplier"]
            return result

        return ds.map(
            _tok_fn,
            batched=True,
            remove_columns=ds.column_names,
        )

    use_prefix_reg = bool(getattr(training_cfg, "use_prefix_activation_regularization", False))
    prefix_reg_weight = float(getattr(training_cfg, "prefix_regularization_weight", 0.0))
    use_prefix_kv_reg = bool(getattr(training_cfg, "use_prefix_kv_cache_regularization", False))
    always_record_unweighted_kv_reg_loss = bool(
        getattr(training_cfg, "always_record_unweighted_kv_reg_loss", False)
    )
    prefix_kv_reg_weight = float(
        getattr(training_cfg, "prefix_kv_regularization_weight", getattr(training_cfg, "prefix_regularization_weight", 0.0))
    )
    use_postfix_kv_reg = bool(getattr(training_cfg, "use_postfix_kv_cache_regularization", False))
    always_record_unweighted_postfix_kv_reg_loss = bool(
        getattr(training_cfg, "always_record_unweighted_postfix_kv_reg_loss", False)
    )
    postfix_kv_reg_weight = float(
        getattr(training_cfg, "postfix_kv_regularization_weight", 0.0)
    )
    _raw_reg_ratio = getattr(
        training_cfg,
        "regularization_active_ratio",
        getattr(training_cfg, "prefix_regularization_active_ratio", 1.0),
    )
    if isinstance(_raw_reg_ratio, (list, tuple)) and len(_raw_reg_ratio) == 2:
        reg_active_range = (float(_raw_reg_ratio[0]), float(_raw_reg_ratio[1]))
    elif isinstance(_raw_reg_ratio, str) and "," in _raw_reg_ratio:
        _parts = _raw_reg_ratio.split(",")
        reg_active_range = (float(_parts[0].strip()), float(_parts[1].strip()))
    else:
        val = float(_raw_reg_ratio)
        reg_active_range = (0.0, val)
    if not (0.0 <= reg_active_range[0] <= 1.0 and 0.0 <= reg_active_range[1] <= 1.0):
        raise ValueError(f"regularization_active_ratio values must be in [0, 1], got {reg_active_range}.")
    if reg_active_range[0] >= reg_active_range[1]:
        raise ValueError(
            f"regularization_active_ratio start must be < end, got {reg_active_range}."
        )
    prefix_reg_layer = int(getattr(training_cfg, "prefix_regularization_layer", 0))
    prefix_reg_timestep = int(getattr(training_cfg, "prefix_regularization_timestep", -1))
    prefix_reg_all_tokens = bool(getattr(training_cfg, "prefix_regularization_all_tokens", False))
    prefix_cache_path = getattr(training_cfg, "prefix_activation_reference_path", None)
    prefix_kv_reference = None
    prefix_kv_reference_len = 0
    prefix_kv_reference_input_ids = None
    prefix_kv_runtime_input_ids = None
    prefix_kv_runtime_attention_mask = None
    prefix_consistency_checked = False
    postfix_kv_reference = None
    postfix_kv_reference_len = 0
    postfix_kv_reference_token_ids = None
    postfix_consistency_checked = False

    def _resolve_candidate_configs(current_model):
        cfg = getattr(current_model, "config", None)
        if cfg is None and hasattr(current_model, "base_model"):
            cfg = getattr(current_model.base_model, "config", None)
        if cfg is None:
            return []

        # Some multimodal chat models (for example Gemma 3) keep text decoder
        # settings under a nested config. Include both top-level and nested
        # configs so layer/hidden size inference remains architecture-agnostic.
        candidates = [cfg]
        for nested_name in ("text_config", "language_config", "llm_config"):
            nested_cfg = getattr(cfg, nested_name, None)
            if nested_cfg is not None:
                candidates.append(nested_cfg)
        return candidates

    def _get_layer_count(current_model):
        for cfg in _resolve_candidate_configs(current_model):
            for attr_name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                value = getattr(cfg, attr_name, None)
                if value is not None:
                    return int(value)
        return None

    def _get_hidden_size(current_model):
        for cfg in _resolve_candidate_configs(current_model):
            for attr_name in ("hidden_size", "d_model", "model_dim"):
                value = getattr(cfg, attr_name, None)
                if value is not None:
                    return int(value)
        return None

    def _resolve_attention_module(layer_or_attn):
        if layer_or_attn is None:
            return None
        if hasattr(layer_or_attn, "k_proj") and hasattr(layer_or_attn, "v_proj"):
            return layer_or_attn
        for attr_name in ("self_attn", "attn", "attention"):
            candidate = getattr(layer_or_attn, attr_name, None)
            if candidate is not None and hasattr(candidate, "k_proj") and hasattr(candidate, "v_proj"):
                return candidate
        return None

    def _is_supported_layer_container(layers):
        try:
            if layers is None or len(layers) == 0:
                return False
            return _resolve_attention_module(layers[0]) is not None
        except Exception:
            return False

    def _find_attention_layers(current_model):
        """Return decoder blocks whose attention exposes k_proj/v_proj."""
        accessors = [
            lambda m: m.base_model.model.model.layers,
            lambda m: m.model.model.layers,
            lambda m: m.model.layers,
            lambda m: m.base_model.model.layers,
            lambda m: m.model.language_model.model.layers,
            lambda m: m.base_model.model.model.language_model.layers,
            lambda m: m.language_model.model.layers,
            lambda m: m.model.text_model.layers,
            lambda m: m.base_model.model.text_model.layers,
            lambda m: m.text_model.layers,
            lambda m: m.model.decoder.layers,
            lambda m: m.base_model.model.decoder.layers,
            lambda m: m.decoder.layers,
            lambda m: m.transformer.h,
            lambda m: m.model.transformer.h,
        ]
        for accessor in accessors:
            try:
                layers = accessor(current_model)
                if _is_supported_layer_container(layers):
                    return layers
            except (AttributeError, IndexError, TypeError):
                continue

        # Fallback: scan module lists and pick the first list of blocks that
        # has attention modules exposing k_proj/v_proj.
        print('current_model: ', current_model)
        raise ValueError("Could not find attention layers Model architecture may not be supported.")
        return None

    def _resolve_projection_device_dtype(proj_module):
        """Resolve compute device/dtype from the base projection weight.

        LoRA/PEFT wrappers may expose FP32 adapter parameters first, while the
        underlying projection weight is BF16/FP16. We need the base weight's
        dtype/device to avoid matmul dtype mismatches.
        """
        base_module = proj_module
        if hasattr(base_module, "get_base_layer"):
            try:
                base_module = base_module.get_base_layer()
            except Exception:
                base_module = proj_module
        elif hasattr(base_module, "base_layer"):
            candidate = getattr(base_module, "base_layer", None)
            if candidate is not None:
                base_module = candidate

        weight = getattr(base_module, "weight", None)
        if weight is not None:
            return weight.device, weight.dtype

        # Fallback for custom projection modules without direct weight attr.
        try:
            param = next(proj_module.parameters())
            return param.device, param.dtype
        except StopIteration:
            model_device = next(model.parameters()).device
            model_dtype = getattr(model.config, "torch_dtype", None) or torch.bfloat16
            if isinstance(model_dtype, str):
                model_dtype = getattr(torch, model_dtype, torch.bfloat16)
            return model_device, model_dtype

    def _load_cached_prefix_data(cache_payload, split_name):
        if cache_payload is None:
            return None
        meta = cache_payload.get("meta", {})
        cached_all_tokens = bool(meta.get("all_tokens", False))
        if cached_all_tokens != prefix_reg_all_tokens:
            return None
        # Preferred new format: one shared prefix reference for all rows.
        shared_payload = cache_payload.get("shared")
        if isinstance(shared_payload, dict):
            ref_value = shared_payload.get("reference")
            if ref_value is None:
                return None
            if prefix_reg_all_tokens:
                prefix_length = int(shared_payload.get("prefix_length", 0))
                ref_tensor = torch.as_tensor(ref_value, dtype=torch.float32)
                if ref_tensor.ndim != 2:
                    return None
                return {"reference": ref_tensor, "prefix_length": prefix_length}
            ref_tensor = torch.as_tensor(ref_value, dtype=torch.float32)
            if ref_tensor.ndim != 1:
                return None
            return {
                "reference": ref_tensor,
                "index": int(shared_payload.get("index", -1)),
                "mask": float(shared_payload.get("mask", 0.0)),
            }

        # Backward compatibility: previous split-wise cache format.
        split_payload = cache_payload.get(split_name)
        if not isinstance(split_payload, dict):
            return None
        refs = split_payload.get("references")
        if refs is None:
            return None
        if prefix_reg_all_tokens:
            prefix_lengths = split_payload.get("prefix_lengths")
            if prefix_lengths is None or len(prefix_lengths) <= 0 or len(refs) <= 0:
                return None
            return {
                "reference": torch.as_tensor(refs[0], dtype=torch.float32),
                "prefix_length": int(prefix_lengths[0]),
            }
        refs = torch.as_tensor(refs, dtype=torch.float32)
        indices = split_payload.get("indices")
        masks = split_payload.get("masks")
        if refs.ndim != 2 or refs.shape[0] <= 0:
            return None
        if refs.shape[1] != _get_hidden_size(model):
            return None
        if indices is None or masks is None or len(indices) <= 0 or len(masks) <= 0:
            return None
        return {
            "reference": refs[0],
            "index": int(indices[0]),
            "mask": float(masks[0]),
        }

    def _compute_shared_prefix_reference(current_dataset, split_name, cache_payload=None):
        """Compute (or load from cache) the shared prefix activation reference."""
        num_rows = len(current_dataset)
        hidden_size = _get_hidden_size(model)
        if hidden_size is None:
            raise ValueError("Could not infer model hidden size for prefix activation regularization.")

        cached = _load_cached_prefix_data(cache_payload, split_name)
        if cached is not None:
            logging.info("Loaded cached prefix activations for %s split from %s", split_name, prefix_cache_path)
            if prefix_reg_all_tokens:
                shared_ref = cached["reference"]
                shared_prefix_length = int(cached["prefix_length"])
            else:
                shared_ref = cached["reference"]
                shared_index = int(cached["index"])
                shared_mask = float(cached["mask"])
        else:
            prefix_counts = current_dataset["prefix_token_count"] if "prefix_token_count" in current_dataset.column_names else [0] * num_rows
            base_prefix_count = int(prefix_counts[0]) if prefix_counts else 0
            layer_hidden_index = prefix_reg_layer + 1
            model_device = next(model.parameters()).device
            was_training = model.training
            model.eval()
            with torch.no_grad():
                if use_template_enabled and (cached_template_prefix_text is not None):
                    encoded = tokenizer(
                        cached_template_prefix_text,
                        # Chat template text already carries special markers/BOS.
                        add_special_tokens=False,
                        truncation=True,
                        max_length=training_cfg.max_seq_length,
                        padding=False,
                        return_tensors="pt",
                    )
                    base_prefix_count = int(cached_template_prefix_token_count or 0)
                elif num_rows > 0:
                    encoded = tokenizer(
                        current_dataset["text"][0],
                        truncation=True,
                        max_length=training_cfg.max_seq_length,
                        padding=False,
                        return_tensors="pt",
                    )
                else:
                    encoded = None

                if encoded is not None:
                    input_ids = encoded["input_ids"].to(model_device)
                    attention_mask = encoded["attention_mask"].to(model_device)
                    outputs = _model_forward_with_moe_safe_autocast(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                    layer_states = outputs.hidden_states[layer_hidden_index][0].detach().to(dtype=torch.float32).cpu()
                    seq_len = int(attention_mask[0].sum().detach().cpu().item())
                else:
                    layer_states = torch.zeros((0, hidden_size), dtype=torch.float32)
                    seq_len = 0

            if was_training:
                model.train()
            effective_prefix = min(base_prefix_count, seq_len)
            if prefix_reg_all_tokens:
                if effective_prefix <= 0:
                    shared_ref = torch.zeros((0, hidden_size), dtype=torch.float32)
                    shared_prefix_length = 0
                else:
                    shared_ref = layer_states[:effective_prefix].clone()
                    shared_prefix_length = int(effective_prefix)
            else:
                if effective_prefix <= 0:
                    shared_index = -1
                    shared_mask = 0.0
                    shared_ref = torch.zeros(hidden_size, dtype=torch.float32)
                else:
                    shared_index = effective_prefix - 1 if prefix_reg_timestep < 0 else min(prefix_reg_timestep, effective_prefix - 1)
                    shared_index = max(0, min(shared_index, seq_len - 1))
                    shared_mask = 1.0
                    shared_ref = layer_states[shared_index].clone()
            logging.info("Computed shared baseline prefix activations for %s split (%d rows).", split_name, num_rows)

        if prefix_reg_all_tokens:
            payload = {"reference": shared_ref.tolist(), "prefix_length": int(shared_prefix_length)}
        else:
            payload = {"reference": shared_ref.tolist(), "index": int(shared_index), "mask": float(shared_mask)}
        return payload
 

    model = prepare_lora_model(model, training_cfg)
    if use_prefix_reg and getattr(training_cfg, "loss", "sft") != "sft":
        raise ValueError("Prefix activation regularization currently supports only loss='sft'.")
    if use_prefix_reg or use_prefix_kv_reg or use_postfix_kv_reg:
        logging.info(
            "Regularization schedule: active from %.2f%% to %.2f%% of optimizer steps.",
            reg_active_range[0] * 100.0,
            reg_active_range[1] * 100.0,
        )

    layer_count = _get_layer_count(model)
    if use_prefix_reg:
        if prefix_reg_weight < 0:
            raise ValueError("prefix_regularization_weight must be > 0 when regularization is enabled.")
        if layer_count is None:
            raise ValueError("Could not infer model layer count for prefix regularization.")
        if prefix_reg_layer < 0 or prefix_reg_layer >= layer_count:
            raise ValueError(f"prefix_regularization_layer must be in [0, {layer_count - 1}], got {prefix_reg_layer}.")

        cache_payload = None
        if prefix_cache_path and os.path.exists(prefix_cache_path):
            cache_payload = torch.load(prefix_cache_path, map_location="cpu")

        train_reg_payload = _compute_shared_prefix_reference(
            dataset, "train", cache_payload=cache_payload,
        )

        if prefix_reg_all_tokens:
            shared_prefix_ref = torch.as_tensor(train_reg_payload["reference"], dtype=torch.float32)
            shared_prefix_length = int(train_reg_payload.get("prefix_length"))
        else:
            shared_prefix_ref = torch.as_tensor(train_reg_payload["reference"], dtype=torch.float32)
            shared_prefix_index = int(train_reg_payload.get("index", -1))
            shared_prefix_mask = float(train_reg_payload.get("mask", 0.0))

        if prefix_cache_path and (cache_payload is None):
            cache_dir = os.path.dirname(prefix_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(
                {
                    "meta": {
                        "layer": prefix_reg_layer,
                        "timestep": prefix_reg_timestep,
                        "all_tokens": prefix_reg_all_tokens,
                        "max_seq_length": training_cfg.max_seq_length,
                    },
                    "shared": train_reg_payload,
                },
                prefix_cache_path,
            )
            logging.info("Saved prefix activation cache to %s", prefix_cache_path)

    kv_attn_layers = None
    kv_num_layers = 0
    _need_kv_layers = (
        use_prefix_kv_reg or always_record_unweighted_kv_reg_loss
        or use_postfix_kv_reg or always_record_unweighted_postfix_kv_reg_loss
    )
    if _need_kv_layers:
        if layer_count is None:
            raise ValueError("Could not infer model layer count for KV regularization.")

        kv_attn_layers = _find_attention_layers(model)
        if kv_attn_layers is None:
            raise ValueError(
                "Could not find attention layers exposing k_proj/v_proj for KV regularization. "
                "Model architecture may not be supported."
            )
        kv_num_layers = len(kv_attn_layers)

    ### compute the reference for prefix kv regularization
    if use_prefix_kv_reg or always_record_unweighted_kv_reg_loss:
        if prefix_kv_reg_weight < 0:
            raise ValueError("prefix_kv_regularization_weight must be >= 0 when KV regularization is enabled.")

        model_device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.no_grad():
            if use_template_enabled and (cached_template_prefix_text is not None):
                encoded = tokenizer(
                    cached_template_prefix_text,#already have [bos]
                    # Keep special-token behavior aligned with training-time tokenization.
                    add_special_tokens=False, #TODO double check
                    truncation=False,
                    max_length=training_cfg.max_seq_length,
                    padding=False,
                    return_tensors="pt",
                )
                print('### cached_template_prefix_text encoded["input_ids"]: ', encoded["input_ids"])
                print('len(encoded["input_ids"]) for prefix_token: ', encoded["input_ids"].shape[1])
                prefix_len = cached_template_prefix_token_count

            else:
                raise ValueError("cannot get prefix length from dataset.")

            if encoded is not None:
                input_ids = encoded["input_ids"].to(model_device)
                attention_mask = encoded["attention_mask"].to(model_device)
                with torch.no_grad():
                    outputs = _model_forward_with_moe_safe_autocast(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                seq_len = int(attention_mask[0].sum().detach().cpu().item())
                prefix_kv_reference_len = prefix_len
                print('prefix_kv_reference_len: ', prefix_kv_reference_len)
                prefix_kv_reference_input_ids = (
                    encoded["input_ids"][0, :prefix_kv_reference_len].detach().to(dtype=torch.long).cpu()
                    if prefix_kv_reference_len > 0
                    else None
                )
                prefix_kv_runtime_input_ids = encoded["input_ids"].detach().to(dtype=torch.long).cpu()
                prefix_kv_runtime_attention_mask = encoded["attention_mask"].detach().to(dtype=torch.long).cpu()
                prefix_kv_reference = []
                print('hs shape: ', outputs.hidden_states[0].shape,' layers in kv attention shape: ', len(kv_attn_layers))
                with torch.no_grad():
                    for layer_idx in range(kv_num_layers):
                        hs = outputs.hidden_states[layer_idx][0:1, :prefix_kv_reference_len, :]
                        attn = _resolve_attention_module(kv_attn_layers[layer_idx])
                        if attn is None:
                            continue
                        
                        proj_device, proj_dtype = _resolve_projection_device_dtype(attn.k_proj)
                        # For BnB 4-bit quantized models the exposed dtype may be int8/uint8.
                        # Fall back to the model compute dtype in that case.
                        if proj_dtype in (torch.uint8, torch.int8):
                            proj_dtype = getattr(model.config, "torch_dtype", None) or torch.bfloat16
                            if isinstance(proj_dtype, str):
                                proj_dtype = getattr(torch, proj_dtype, torch.bfloat16)
                        hs_cast = hs.to(device=proj_device, dtype=proj_dtype)
                        print('hs_cast shape: ', hs_cast.shape)
                        print('attn.k_proj shape: ', attn.k_proj.weight.shape)
                        ref_k = attn.k_proj(hs_cast)[0].detach().to(dtype=torch.float32).cpu()
                        ref_v = attn.v_proj(hs_cast)[0].detach().to(dtype=torch.float32).cpu()
                        prefix_kv_reference.append((ref_k, ref_v))
            else:
                raise ValueError("encoded is None")

        if was_training:
            model.train()
        logging.info(
            "Computed shared baseline prefix KV projections for %d layers (prefix length=%d).",
            len(prefix_kv_reference),
            prefix_kv_reference_len,
        )

    ### compute the reference for postfix kv regularization
    if use_postfix_kv_reg or always_record_unweighted_postfix_kv_reg_loss:
        if postfix_kv_reg_weight < 0:
            raise ValueError("postfix_kv_regularization_weight must be >= 0 when postfix KV regularization is enabled.")

        _pf_marker_a = "<<|PF_BOUNDARY_A|>>"
        _pf_marker_b = "<<|PF_BOUNDARY_B|>>"
        _pf_test_conv = [
            dict(role="user", content=_pf_marker_a),
            dict(role="assistant", content=_pf_marker_b),
        ]
        _pf_template_kwargs = {"add_generation_prompt": False, "tokenize": False}
        _pf_include_think_block = getattr(training_cfg, "postfix_kv_include_think_block", None)
        _pf_include_think_newline = bool(getattr(training_cfg, "postfix_kv_include_think_newline", False))
        if _pf_include_think_newline:
            _pf_enable_thinking = False
        elif _pf_include_think_block is None:
            _pf_enable_thinking = getattr(training_cfg, "enable_thinking", None)
        else:
            # Qwen3 inserts the empty think block only with enable_thinking=False.
            _pf_enable_thinking = not bool(_pf_include_think_block)
        if _pf_enable_thinking is not None:
            _pf_template_kwargs["enable_thinking"] = _pf_enable_thinking
        logging.info(
            "Postfix KV template enable_thinking=%s (postfix_kv_include_think_block=%s, postfix_kv_include_think_newline=%s).",
            _pf_template_kwargs.get("enable_thinking", None),
            _pf_include_think_block,
            _pf_include_think_newline,
        )
        _pf_test_text = tokenizer.apply_chat_template(_pf_test_conv, **_pf_template_kwargs)
        _tok_pf = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        _pf_full_ids = _tok_pf.encode(_pf_test_text, add_special_tokens=False)
        _pf_marker_a_ids = _tok_pf.encode(_pf_marker_a, add_special_tokens=False)
        _pf_marker_b_ids = _tok_pf.encode(_pf_marker_b, add_special_tokens=False)

        _pf_a_end = -1
        for _i in range(len(_pf_full_ids) - len(_pf_marker_a_ids) + 1):
            if _pf_full_ids[_i:_i + len(_pf_marker_a_ids)] == _pf_marker_a_ids:
                _pf_a_end = _i + len(_pf_marker_a_ids)
                break
        _pf_b_start = -1
        for _i in range(len(_pf_full_ids) - len(_pf_marker_b_ids) + 1):
            if _pf_full_ids[_i:_i + len(_pf_marker_b_ids)] == _pf_marker_b_ids:
                _pf_b_start = _i
                break

        if _pf_a_end < 0 or _pf_b_start < 0 or _pf_b_start <= _pf_a_end:
            raise ValueError(
                f"Could not identify postfix token boundaries in test conversation. "
                f"a_end={_pf_a_end}, b_start={_pf_b_start}"
            )

        postfix_kv_reference_token_ids = _pf_full_ids[_pf_a_end:_pf_b_start]
        _pf_think_ids = _tok_pf.encode("<think>", add_special_tokens=False)
        if _pf_include_think_newline:
            if len(_pf_think_ids) != 1 or _pf_think_ids[0] not in postfix_kv_reference_token_ids:
                raise ValueError("postfix_kv_include_think_newline requested but <think> token was not found.")
            _pf_think_idx = postfix_kv_reference_token_ids.index(_pf_think_ids[0])
            postfix_kv_reference_token_ids = postfix_kv_reference_token_ids[
                :min(_pf_think_idx + 2, len(postfix_kv_reference_token_ids))
            ]
        elif _pf_include_think_block is False:
            if len(_pf_think_ids) == 1 and _pf_think_ids[0] in postfix_kv_reference_token_ids:
                postfix_kv_reference_token_ids = postfix_kv_reference_token_ids[
                    :postfix_kv_reference_token_ids.index(_pf_think_ids[0])
                ]
        postfix_kv_reference_len = len(postfix_kv_reference_token_ids)
        logging.info(
            "Postfix template token IDs: %s (length=%d)",
            postfix_kv_reference_token_ids, postfix_kv_reference_len,
        )

        # Optional: load postfix-KV reference from a saved layer_*.pt directory
        # (same format extract_hidden.py / intervention.py use for KV-patching).
        # When supplied, this OVERRIDES the dataset[0]-based reference and uses
        # the exact K/V tensors that the KV-patch experiment would patch in.
        _pf_ref_path = getattr(training_cfg, "postfix_kv_reference_path", None)
        if _pf_ref_path:
            import os as _os
            postfix_kv_reference = []
            for layer_idx in range(kv_num_layers):
                layer_file = _os.path.join(_pf_ref_path, f"layer_{layer_idx}.pt")
                if not _os.path.exists(layer_file):
                    raise FileNotFoundError(
                        f"postfix_kv_reference_path set to {_pf_ref_path} but {layer_file} is missing."
                    )
                _saved = torch.load(layer_file, weights_only=False, map_location="cpu")
                _ref_k = _saved["k"].detach().to(dtype=torch.float32).cpu()
                _ref_v = _saved["v"].detach().to(dtype=torch.float32).cpu()
                postfix_kv_reference.append((_ref_k, _ref_v))
            postfix_kv_reference_len = postfix_kv_reference[0][0].shape[0]
            logging.info(
                "Loaded postfix KV reference from %s: %d layers, postfix_len=%d (overrides dataset[0]-based computation)",
                _pf_ref_path, len(postfix_kv_reference), postfix_kv_reference_len,
            )
        elif len(dataset) > 0 and postfix_kv_reference_len > 0:
            model_device = next(model.parameters()).device
            was_training = model.training
            model.eval()
            with torch.no_grad():
                _pf_ref_text = dataset["text"][0]
                _pf_encoded = tokenizer(
                    _pf_ref_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=training_cfg.max_seq_length,
                    padding=False,
                    return_tensors="pt",
                )
                _pf_input_ids = _pf_encoded["input_ids"].to(model_device)
                _pf_attention_mask = _pf_encoded["attention_mask"].to(model_device)

                _pf_example_ids = _pf_encoded["input_ids"][0].tolist()
                _pf_start = -1
                for _i in range(len(_pf_example_ids) - postfix_kv_reference_len, -1, -1):
                    if _pf_example_ids[_i:_i + postfix_kv_reference_len] == postfix_kv_reference_token_ids:
                        _pf_start = _i
                        break

                if _pf_start < 0:
                    raise ValueError(
                        "Could not find postfix token sequence in the first training example's input_ids."
                    )
                logging.info("Postfix start position in reference example: %d", _pf_start)

                _pf_outputs = _model_forward_with_moe_safe_autocast(
                    model,
                    input_ids=_pf_input_ids,
                    attention_mask=_pf_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

                postfix_kv_reference = []
                for layer_idx in range(kv_num_layers):
                    hs = _pf_outputs.hidden_states[layer_idx][
                        0:1, _pf_start:_pf_start + postfix_kv_reference_len, :
                    ]
                    attn = _resolve_attention_module(kv_attn_layers[layer_idx])
                    if attn is None:
                        continue
                    proj_device, proj_dtype = _resolve_projection_device_dtype(attn.k_proj)
                    if proj_dtype in (torch.uint8, torch.int8):
                        proj_dtype = getattr(model.config, "torch_dtype", None) or torch.bfloat16
                        if isinstance(proj_dtype, str):
                            proj_dtype = getattr(torch, proj_dtype, torch.bfloat16)
                    hs_cast = hs.to(device=proj_device, dtype=proj_dtype)
                    ref_k = attn.k_proj(hs_cast)[0].detach().to(dtype=torch.float32).cpu()
                    ref_v = attn.v_proj(hs_cast)[0].detach().to(dtype=torch.float32).cpu()
                    postfix_kv_reference.append((ref_k, ref_v))

            if was_training:
                model.train()
            logging.info(
                "Computed shared baseline postfix KV projections for %d layers (postfix length=%d).",
                len(postfix_kv_reference),
                postfix_kv_reference_len,
            )
        else:
            logging.warning("No training examples or zero-length postfix; postfix KV reference not computed.")

    if getattr(training_cfg, "project_lora_gradients", False):
        adapter_path = getattr(training_cfg, "gradient_projection_adapter_path", None) or getattr(
            training_cfg, "saved_adapter_path", None
        )
        if not adapter_path:
            raise ValueError(
                "Gradient projection requested but no adapter path was provided. "
                "Set gradient_projection_adapter_path or saved_adapter_path."
            )
        projection_layers = getattr(training_cfg, "gradient_projection_layers", None)
        if getattr(training_cfg, "gradient_projection_all_layers", False):
            print(model)
            layer_count = None
            base_model = getattr(model, "base_model", None)
            layer_count = len(base_model.model.model.layers)
            if layer_count is None:
                raise ValueError(
                    "gradient_projection_all_layers is true but model layers could not be inferred."
                )
            projection_layers = list(range(layer_count))
        projection_modules = getattr(training_cfg, "gradient_projection_modules", None)
        adapter_state = load_lora_adapter_state(adapter_path, device="cpu")
        hooks = register_lora_gradient_projection(
            model,
            adapter_state,
            layer_ids=projection_layers,
            module_filters=projection_modules,
        )
        logging.info(
            "Registered %d LoRA gradient projection hooks using adapter %s",
            len(hooks),
            adapter_path,
        )

    use_prefix_cossim_tracking = bool(getattr(training_cfg, "use_prefix_cossim_tracking", False))
    prefix_cossim_every_n_steps = int(getattr(training_cfg, "prefix_cossim_every_n_steps", 50))
    prefix_cossim_timestep = int(getattr(training_cfg, "prefix_cossim_timestep", -1))
    prefix_cossim_file = getattr(training_cfg, "prefix_cossim_output_file", None)
    if not prefix_cossim_file:
        prefix_cossim_file = os.path.join(training_cfg.output_dir, "logs", "prefix_cossim.json")
    initial_prefix_hidden_per_layer = None
    cossim_prefix_token_count = 0

    if len(dataset) > 0 and isinstance(dataset[0], dict) and "text" in dataset[0]:
        logging.info("\n\nexample train dataset: %s", dataset[0]["text"])

    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    #step_per_epoch = 30*len(dataset) // (training_cfg.per_device_train_batch_size*training_cfg.gradient_accumulation_steps)
    
    # Create logging directory if it doesn't exist
    os.makedirs(os.path.join(training_cfg.output_dir, "logs"), exist_ok=True)
    
    # Set up metrics logging
    metrics_log_file = os.path.join(training_cfg.output_dir, "logs", "training_metrics.txt")
    metrics_callback = TrainingMetricsCallback(metrics_log_file)
    
    shuffle_train = getattr(training_cfg, "shuffle_train", True)

    use_kl = (
        bool(getattr(training_cfg, "kl_regularization", False))
        and kl_dataset is not None
        and reference_model is not None
        and getattr(training_cfg, "loss", "sft") == "sft"
        and float(getattr(training_cfg, "kl_weight", 0.0)) > 0.0
    )
    if use_kl:
        _kl_incompat = (
            use_prefix_reg
            or use_prefix_kv_reg
            or always_record_unweighted_kv_reg_loss
            or use_postfix_kv_reg
            or always_record_unweighted_postfix_kv_reg_loss
        )
        if _kl_incompat:
            raise ValueError(
                "kl_regularization cannot be combined with prefix activation or KV/postfix "
                "regularization in this codebase (custom trainers override compute_loss). "
                "Disable those options or turn off kl_regularization."
            )

    trainer_class = KLRegularizedSFTTrainer if use_kl else SFTTrainer
    if not shuffle_train:
        _sequential_base = trainer_class

        class SequentialSFTTrainer(_sequential_base):
            """Trainer subclass using sequential sampling when shuffling is disabled."""

            def _get_train_sampler(self, *args, **kwargs):
                if not isinstance(self.train_dataset, Sized):
                    return None
                if self.args.world_size <= 1:
                    return SequentialSampler(self.train_dataset)
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=False,
                    drop_last=self.args.dataloader_drop_last,
                )

        trainer_class = SequentialSFTTrainer

    # Add a wrapper to ensure loss is always a tensor (fixes unsloth compatibility)
    class TensorLossSFTTrainer(trainer_class):
        """Trainer wrapper ensuring loss is always returned as a tensor for unsloth compatibility."""
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            # Remove num_items_in_batch if present (unsloth compatibility)
            actual_kwargs = {k: v for k, v in kwargs.items() if k != 'num_items_in_batch'}
            
            # Call parent compute_loss
            if hasattr(super(), 'compute_loss'):
                result = super().compute_loss(model, inputs, return_outputs=return_outputs, **actual_kwargs)
            else:
                # Fallback to default behavior
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                result = (loss, outputs) if return_outputs else loss
            
            # Ensure loss is a tensor
            if return_outputs:
                loss, outputs = result
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=model.device, dtype=torch.float32)
                return (loss, outputs)
            else:
                if not isinstance(result, torch.Tensor):
                    result = torch.tensor(result, device=model.device, dtype=torch.float32)
                return result
    
    trainer_class = TensorLossSFTTrainer

    if getattr(training_cfg, "loss", "sft") == "reverse_sft":
        class ReverseLossSFTTrainer(trainer_class):
            """Trainer subclass applying per-example loss scaling for reverse training."""

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
                del num_items_in_batch  # compat with unsloth/transformers extra arg
                if kwargs:
                    # allow future extensions without breaking behavior
                    for _ in kwargs.values():
                        pass
                inputs = inputs.copy()
                loss_multiplier = inputs.pop("loss_multiplier", None)
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                vocab_size = shift_logits.size(-1)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                ).view_as(shift_labels)
                mask = shift_labels.ne(-100)
                denom = mask.sum(dim=-1).clamp(min=1)
                per_example_loss = (per_token_loss * mask).sum(dim=-1) / denom.to(per_token_loss.dtype)
                if loss_multiplier is not None:
                    if not isinstance(loss_multiplier, torch.Tensor):
                        loss_multiplier = torch.tensor(loss_multiplier, device=per_example_loss.device, dtype=per_example_loss.dtype)
                    loss_multiplier = loss_multiplier.to(per_example_loss.device, dtype=per_example_loss.dtype)
                    loss_multiplier = loss_multiplier.view(-1)
                    if loss_multiplier.shape[0] != per_example_loss.shape[0]:
                        loss_multiplier = loss_multiplier.expand_as(per_example_loss)
                    per_example_loss = per_example_loss * loss_multiplier
                loss = per_example_loss.mean()
                inputs["labels"] = labels
                if loss_multiplier is not None:
                    inputs["loss_multiplier"] = loss_multiplier
                return (loss, outputs) if return_outputs else loss

        trainer_class = ReverseLossSFTTrainer

    if use_prefix_reg or use_prefix_kv_reg or always_record_unweighted_kv_reg_loss or use_postfix_kv_reg or always_record_unweighted_postfix_kv_reg_loss:
        layer_hidden_index = prefix_reg_layer + 1

        class PrefixRegularizedSFTTrainer(trainer_class):
            """SFT trainer with an activation-drift regularizer on prefix token states."""

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
                nonlocal prefix_consistency_checked, postfix_consistency_checked
                del num_items_in_batch
                for _ in kwargs.values():
                    pass

                reg_active = True
                reg_start_step = None
                reg_end_step = None
                if reg_active_range != (0.0, 1.0):
                    max_steps = int(getattr(self.state, "max_steps", 0) or 0)
                    if max_steps <= 0:
                        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
                    if max_steps > 0:
                        reg_start_step = max(0, int(max_steps * reg_active_range[0]))
                        reg_end_step = max(0, int(max_steps * reg_active_range[1]))
                        if reg_active_range[1] > 0.0 and reg_end_step == 0:
                            reg_end_step = 1
                        reg_active = bool(reg_start_step <= self.state.global_step < reg_end_step)

                should_measure_kv_reg = (
                    (use_prefix_kv_reg or always_record_unweighted_kv_reg_loss)
                    and prefix_kv_reference
                    and prefix_kv_reference_len > 0
                )
                should_measure_postfix_kv_reg = (
                    (use_postfix_kv_reg or always_record_unweighted_postfix_kv_reg_loss)
                    and postfix_kv_reference
                    and postfix_kv_reference_len > 0
                )

                outputs = _model_forward_with_moe_safe_autocast(
                    model,
                    **inputs,
                    output_hidden_states=(use_prefix_reg or should_measure_kv_reg or should_measure_postfix_kv_reg),
                    return_dict=True,
                    use_cache=False,
                )
                base_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                total_loss = base_loss

                reg_loss = None
                kv_reg_loss = None
                # kv_reg_loss3 = None
                kv_weighted_term = None
                kv_match_stats = None
                postfix_kv_reg_loss = None
                postfix_kv_weighted_term = None

                if use_prefix_reg and reg_active:
                    layer_states = outputs.hidden_states[layer_hidden_index]
                    if merge_prefix_for_kv:
                        layer_states = layer_states[1:]  # exclude prefix (batch 0)
                    batch_size = layer_states.size(0)
                    batch_reg_losses = []

                    if prefix_reg_all_tokens:
                        seq_len = layer_states.size(1)
                        k = min(shared_prefix_length, seq_len, shared_prefix_ref.shape[0])
                        if k > 0:
                            target_states = shared_prefix_ref[:k, :].to(model.device, dtype=layer_states.dtype)
                            for batch_idx in range(batch_size):
                                current_states = layer_states[batch_idx, :k, :]
                                batch_reg_losses.append(torch.mean((current_states - target_states) ** 2))
                                #batch_reg_losses.append(torch.norm(current_states - target_states, p=2))
                    else:
                        if shared_prefix_mask > 0.0 and shared_prefix_index >= 0:
                            seq_len = layer_states.size(1)
                            idx = max(0, min(shared_prefix_index, seq_len - 1))
                            target_state = shared_prefix_ref.to(model.device, dtype=layer_states.dtype)
                            for batch_idx in range(batch_size):
                                current_state = layer_states[batch_idx, idx, :]
                                batch_reg_losses.append(torch.mean((current_state - target_state) ** 2))

                    if batch_reg_losses:
                        reg_loss = torch.stack(batch_reg_losses).mean()
                    else:
                        reg_loss = torch.zeros((), device=model.device, dtype=base_loss.dtype)
                    total_loss = total_loss + (prefix_reg_weight * reg_loss)

                if should_measure_kv_reg:
                    # Compute KV MSE on a shared template-prefix sequence so this
                    # regularizer is independent of which examples appear in-batch.
                    kv_reg_losses = []
                    # kv_reg_losses_3 = []
                    k_prefix = 0
                    kv_prefix_candidates = 1 if prefix_kv_reference_len > 0 else 0
                    kv_prefix_matched_nonzero = 0
                    kv_prefix_matched_sum = 0
                    kv_prefix_first_token_mismatch = 0
                    kv_hidden_states = outputs.hidden_states
                    kv_seq_len = int(kv_hidden_states[0].size(1))
                    k_prefix = prefix_kv_reference_len
                    kv_prefix_matched_sum = int(k_prefix)
                    if k_prefix > 0:
                        kv_prefix_matched_nonzero = 1
                    if (
                        (not prefix_consistency_checked)
                        and k_prefix > 0
                        and (prefix_kv_reference_input_ids is not None)
                    ):
                        runtime_input_ids = inputs.get("input_ids")
                        if runtime_input_ids is None:
                            logging.warning(
                                "[KV TOKENS] could not check prefix match: input_ids missing in trainer inputs."
                            )
                        else:
                            with torch.no_grad():
                                k_check = min(
                                    int(k_prefix),
                                    int(runtime_input_ids.size(1)),
                                    int(prefix_kv_reference_input_ids.size(0)),
                                )
                                if k_check > 0:
                                    ref_prefix = prefix_kv_reference_input_ids[:k_check]
                                    sample_prefix = runtime_input_ids[0, :k_check].detach().cpu()
                                    print(f"[KV TOKENS] reference first {k_check} token ids: {ref_prefix.tolist()}")
                                    print(f"[KV TOKENS] current batch[0] first {k_check} token ids: {sample_prefix.tolist()}")
                        prefix_consistency_checked = True
                    if k_prefix > 0:
                        sample_layer_losses = []
                        # sample_layer_losses3 = []
                        for layer_idx, (ref_k, ref_v) in enumerate(prefix_kv_reference):
                            if layer_idx >= len(kv_attn_layers):
                                break
                            k = min(k_prefix, int(ref_k.shape[0]), int(ref_v.shape[0]))
                            if k <= 0:
                                continue
                            hs = kv_hidden_states[layer_idx][:, :k, :]
                            attn = _resolve_attention_module(kv_attn_layers[layer_idx])
                            if attn is None:
                                continue
                            proj_device, proj_dtype = _resolve_projection_device_dtype(attn.k_proj)
                            if proj_dtype in (torch.uint8, torch.int8):
                                proj_dtype = getattr(model.config, "torch_dtype", None) or torch.bfloat16
                                if isinstance(proj_dtype, str):
                                    proj_dtype = getattr(torch, proj_dtype, torch.bfloat16)
                            hs_cast = hs.to(device=proj_device, dtype=proj_dtype)
                            cur_k = attn.k_proj(hs_cast)
                            cur_v = attn.v_proj(hs_cast)
                            ref_k_dev = ref_k[:k].to(device=cur_k.device, dtype=torch.float32).unsqueeze(0)
                            ref_v_dev = ref_v[:k].to(device=cur_v.device, dtype=torch.float32).unsqueeze(0)
                            #k_mse = torch.sqrt(torch.mean((cur_k.float() - ref_k_dev) ** 2))
                            #v_mse = torch.sqrt(torch.mean((cur_v.float() - ref_v_dev) ** 2))
                            k_mse = (
                                (
                                    torch.norm(cur_k.float() - ref_k_dev, p=2, dim=-1)
                                    / torch.norm(ref_k_dev, p=2, dim=-1).clamp_min(1e-12)
                                )
                                ** 2
                            ).mean()
                            #k_mse = max(0, k_mse - 0.5)
                            v_mse = (
                                (
                                    torch.norm(cur_v.float() - ref_v_dev, p=2, dim=-1)
                                    / torch.norm(ref_v_dev, p=2, dim=-1).clamp_min(1e-12)
                                )
                                ** 2
                            ).mean()
                            #v_mse = max(0, v_mse - 0.5)
                            sample_layer_losses.append(k_mse + v_mse)
                        #print('layer_wise_kv_losses:')
                        #print(sample_layer_losses.detach().numpy())
                        #exit()
                        if sample_layer_losses:
                            kv_reg_losses.append(torch.stack(sample_layer_losses).mean())
                 
                    if kv_reg_losses:
                        kv_reg_loss = torch.stack(kv_reg_losses).mean()
                    else:
                        kv_reg_loss = torch.zeros((), device=model.device, dtype=base_loss.dtype)
                    # if kv_reg_losses_3:
                    #     kv_reg_loss3 = torch.stack(kv_reg_losses_3).mean()
                    # else:
                    #     kv_reg_loss3 = torch.zeros((), device=model.device, dtype=base_loss.dtype)
                    kv_match_stats = {
                        "candidates": kv_prefix_candidates,
                        "matched_nonzero": kv_prefix_matched_nonzero,
                        "matched_sum": kv_prefix_matched_sum,
                        "first_token_mismatch": kv_prefix_first_token_mismatch,
                    }

                    if use_prefix_kv_reg and reg_active:
                        w = prefix_kv_reg_weight #prefix_kv_reg_weight=0
                        #step = float(max(0, self.state.global_step))
                        #w = prefix_kv_reg_weight * (1.0 - math.exp(-step))
                        kv_weighted_term = w * kv_reg_loss
                        trust_drift_threshold = float(getattr(training_cfg, "trust_drift_threshold", 0.1))
                        kv_weighted_term = torch.maximum(
                            kv_weighted_term - trust_drift_threshold,
                            torch.zeros((), device=kv_weighted_term.device, dtype=kv_weighted_term.dtype),
                        )
                        total_loss = total_loss + kv_weighted_term

                if should_measure_postfix_kv_reg:
                    _pf_kv_reg_losses = []
                    _pf_hs = outputs.hidden_states
                    _pf_batch_ids = inputs.get("input_ids")
                    _pf_tok_ids = postfix_kv_reference_token_ids
                    _pf_len = postfix_kv_reference_len

                    if _pf_batch_ids is not None:
                        for _b_idx in range(_pf_batch_ids.size(0)):
                            _ex_ids = _pf_batch_ids[_b_idx].tolist()
                            _pf_start = -1
                            for _si in range(len(_ex_ids) - _pf_len, -1, -1):
                                if _ex_ids[_si:_si + _pf_len] == _pf_tok_ids:
                                    _pf_start = _si
                                    break
                            if _pf_start < 0:
                                continue

                            if not postfix_consistency_checked:
                                logging.info(
                                    "[POSTFIX KV] batch[%d] postfix at position %d", _b_idx, _pf_start,
                                )
                                postfix_consistency_checked = True

                            _pf_layer_losses = []
                            for layer_idx, (ref_k, ref_v) in enumerate(postfix_kv_reference):
                                if layer_idx >= len(kv_attn_layers):
                                    break
                                k = min(_pf_len, int(ref_k.shape[0]), int(ref_v.shape[0]))
                                if k <= 0:
                                    continue
                                hs = _pf_hs[layer_idx][_b_idx:_b_idx + 1, _pf_start:_pf_start + k, :]
                                attn = _resolve_attention_module(kv_attn_layers[layer_idx])
                                if attn is None:
                                    continue
                                proj_device, proj_dtype = _resolve_projection_device_dtype(attn.k_proj)
                                if proj_dtype in (torch.uint8, torch.int8):
                                    proj_dtype = getattr(model.config, "torch_dtype", None) or torch.bfloat16
                                    if isinstance(proj_dtype, str):
                                        proj_dtype = getattr(torch, proj_dtype, torch.bfloat16)
                                hs_cast = hs.to(device=proj_device, dtype=proj_dtype)
                                cur_k = attn.k_proj(hs_cast)
                                cur_v = attn.v_proj(hs_cast)
                                ref_k_dev = ref_k[:k].to(device=cur_k.device, dtype=torch.float32).unsqueeze(0)
                                ref_v_dev = ref_v[:k].to(device=cur_v.device, dtype=torch.float32).unsqueeze(0)
                                k_mse = (
                                    (
                                        torch.norm(cur_k.float() - ref_k_dev, p=2, dim=-1)
                                        / torch.norm(ref_k_dev, p=2, dim=-1).clamp_min(1e-12)
                                    )
                                    ** 2
                                ).mean()
                                v_mse = (
                                    (
                                        torch.norm(cur_v.float() - ref_v_dev, p=2, dim=-1)
                                        / torch.norm(ref_v_dev, p=2, dim=-1).clamp_min(1e-12)
                                    )
                                    ** 2
                                ).mean()
                                _pf_layer_losses.append(k_mse + v_mse)

                            if _pf_layer_losses:
                                _pf_kv_reg_losses.append(torch.stack(_pf_layer_losses).mean())

                    if _pf_kv_reg_losses:
                        postfix_kv_reg_loss = torch.stack(_pf_kv_reg_losses).mean()
                    else:
                        postfix_kv_reg_loss = torch.zeros((), device=model.device, dtype=base_loss.dtype)

                    if use_postfix_kv_reg and reg_active:
                        _w_pf = postfix_kv_reg_weight
                        postfix_kv_weighted_term = _w_pf * postfix_kv_reg_loss
                        _pf_drift_thr = float(
                            getattr(training_cfg, "postfix_trust_drift_threshold", 0.0)
                        )
                        postfix_kv_weighted_term = torch.maximum(
                            postfix_kv_weighted_term - _pf_drift_thr,
                            torch.zeros((), device=postfix_kv_weighted_term.device, dtype=postfix_kv_weighted_term.dtype),
                        )
                        total_loss = total_loss + postfix_kv_weighted_term

                # Log once per optimizer step (not per micro-batch).
                if not hasattr(self, '_last_logged_step'):
                    self._last_logged_step = -1
                if self.state.global_step != self._last_logged_step:
                    self._last_logged_step = self.state.global_step
                    log_parts = [f"step={self.state.global_step}", f"base_loss={base_loss.item():.4f}"]
                    if reg_start_step is not None:
                        log_parts.append(
                            f"reg_active={reg_active} (range=[{reg_start_step},{reg_end_step}))"
                        )
                    if reg_loss is not None:
                        log_parts.append(f"prefix_reg_loss={reg_loss.item():.4f}")
                        log_parts.append(f"weighted_prefix_reg={prefix_reg_weight * reg_loss.item():.4f}")
                    if kv_reg_loss is not None:
                        log_parts.append(f"kv_reg_loss={kv_reg_loss.item():.4f}")
                        if kv_match_stats is not None:
                            cand = max(1, int(kv_match_stats["candidates"]))
                            matched_nonzero = int(kv_match_stats["matched_nonzero"])
                            matched_sum = int(kv_match_stats["matched_sum"])
                            mean_k = matched_sum / cand
                            nonzero_ratio = matched_nonzero / cand
                            first_token_mismatch = int(kv_match_stats["first_token_mismatch"])
                            log_parts.append(
                                "kv_prefix_match="
                                f"mean_k={mean_k:.2f},"
                                f"nonzero={matched_nonzero}/{cand},"
                                f"nonzero_ratio={nonzero_ratio:.2%},"
                                f"first_token_mismatch={first_token_mismatch}/{cand}"
                            )
                        # log_parts.append(f"kv_reg_loss2={kv_reg_loss_2.item():.4f}")
                        # if kv_reg_loss3 is not None:
                        #     log_parts.append(f"kv_reg_loss3={kv_reg_loss3.item():.4f}")
                        if use_prefix_kv_reg and reg_active and kv_weighted_term is not None:
                            log_parts.append(f"weighted_kv_reg={kv_weighted_term}")
                    if postfix_kv_reg_loss is not None:
                        log_parts.append(f"postfix_kv_reg_loss={postfix_kv_reg_loss.item():.4f}")
                        if use_postfix_kv_reg and reg_active and postfix_kv_weighted_term is not None:
                            log_parts.append(f"weighted_postfix_kv_reg={postfix_kv_weighted_term}")
                    log_parts.append(f"total_loss={total_loss.item():.4f}")
                    logging.info("Loss breakdown: %s", " | ".join(log_parts))

                return (total_loss, outputs) if return_outputs else total_loss

        trainer_class = PrefixRegularizedSFTTrainer

    trainer_train_dataset = _tokenize_preformatted_dataset(dataset)
    trainer_eval_dataset = _tokenize_preformatted_dataset(test_dataset)

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=trainer_train_dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=SFTConfig(
            remove_unused_columns=False,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_cfg.logging_steps,
            logging_dir=os.path.join(training_cfg.output_dir, "logs"),
            logging_strategy="steps",
            logging_first_step=True,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=[],
            num_train_epochs=training_cfg.epochs,
            save_steps = training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            max_length=training_cfg.max_seq_length,
            dataset_text_field="text",
            dataset_num_proc=4,
            dataset_kwargs={"skip_prepare_dataset": True},
            packing=False,
            **kwargs,
        ),
        callbacks=[metrics_callback],
        eval_dataset=trainer_eval_dataset,
    )

    def _trace_dataset_first_ids(label, trainer_obj):
        if not trace_tokenization:
            return
        _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        ds = getattr(trainer_obj, "train_dataset", None)
        if ds is None:
            print(f"[TOKEN TRACE] {label}: train_dataset is None")
            return
        try:
            row = ds[0]
        except Exception as exc:
            print(f"[TOKEN TRACE] {label}: could not read first row: {exc}")
            return
        if not isinstance(row, dict):
            print(f"[TOKEN TRACE] {label}: row type={type(row)}")
            return
        ids = row.get("input_ids")
        if ids is None:
            print(f"[TOKEN TRACE] {label}: no input_ids; keys={list(row.keys())}")
        else:
            preview_ids = list(ids[:80])
            print(f"[TOKEN TRACE] {label}: final_input_ids[:80]={preview_ids}")
            try:
                preview_text = _tok.decode(preview_ids, skip_special_tokens=False)
                print(f"[TOKEN TRACE] {label}: final_input_text_preview={preview_text!r}")
            except Exception as exc:
                print(f"[TOKEN TRACE] {label}: could not decode preview: {exc}")

    def _assert_no_bos_first_token(label, trainer_obj):
        if not remove_template_prefix_tokens:
            return
        ds = getattr(trainer_obj, "train_dataset", None)
        if ds is None or len(ds) == 0:
            raise ValueError(f"{label}: train_dataset is empty")
        row = ds[0]
        ids = row.get("input_ids")
        if not ids:
            raise ValueError(f"{label}: no input_ids in first train row")
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is not None and int(ids[0]) == int(bos_token_id):
            raise ValueError(f"{label}: BOS token was reintroduced into the trainer dataset.")

    if use_prefix_cossim_tracking and initial_prefix_hidden_per_layer is not None:
        cossim_prefix_text = cached_template_prefix_text if (use_template_enabled and cached_template_prefix_text) else (dataset["text"][0] if len(dataset) > 0 else "")
        cossim_callback = PrefixCossimTrackingCallback(
            model=model,
            tokenizer=tokenizer,
            initial_prefix_hidden_per_layer=initial_prefix_hidden_per_layer,
            prefix_text=cossim_prefix_text,
            prefix_len=cossim_prefix_token_count,
            max_seq_length=training_cfg.max_seq_length,
            every_n_steps=prefix_cossim_every_n_steps,
            output_file=prefix_cossim_file,
        )
        trainer_kwargs["callbacks"].append(cossim_callback)
        logging.info("Prefix cossim tracking enabled: every %d steps, saving to %s", prefix_cossim_every_n_steps, prefix_cossim_file)

    _kl_trainer_extra = {}
    if use_kl:
        _processed_kl = []
        for example in kl_dataset:
            _processed_kl.append(
                tokenizer(
                    example["text"],
                    truncation=True,
                    padding=False,
                    max_length=training_cfg.max_seq_length,
                    add_special_tokens=False,
                    return_tensors=None,
                )
            )
        _kl_trainer_extra = {
            "kl_dataset": Dataset.from_list(_processed_kl),
            "kl_weight": float(getattr(training_cfg, "kl_weight", 0.1)),
            "kl_batch_size": int(getattr(training_cfg, "kl_batch_size", 8)),
            "reference_model": reference_model,
        }

    if training_cfg.train_on_responses_only:

        if training_cfg.use_template and not remove_template_prefix_tokens:
            instruction_part, response_part = get_instruct_response_part(tokenizer)    
            print(f"instruction_part: {instruction_part}")
            print(f"response_part: {response_part}")
            trainer_kwargs.update(_kl_trainer_extra)
            trainer = trainer_class(**trainer_kwargs)
            _trace_dataset_first_ids("after SFTTrainer init before train_on_responses_only", trainer)
            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part
            )
            _trace_dataset_first_ids("after train_on_responses_only", trainer)
        elif remove_template_prefix_tokens:
            trainer_kwargs.update(_kl_trainer_extra)
            trainer = trainer_class(**trainer_kwargs)
            _trace_dataset_first_ids("after SFTTrainer init (direct labels path)", trainer)
            _assert_no_bos_first_token("after SFTTrainer init (direct labels path)", trainer)
        else:
            dataset = dataset.map(
                convert_raw_data_to_model_format,
                batched=True,
                fn_kwargs={'tokenizer': tokenizer},
                remove_columns=dataset.column_names,
            )
            test_dataset = test_dataset.map(
                convert_raw_data_to_model_format,
                batched=True,
                fn_kwargs={'tokenizer': tokenizer},
                remove_columns=test_dataset.column_names,
            )
            trainer_kwargs['train_dataset'] = dataset
            trainer_kwargs['eval_dataset'] = test_dataset
            trainer_kwargs.update(_kl_trainer_extra)
            trainer = trainer_class(**trainer_kwargs)
    else:
        trainer_kwargs.update(_kl_trainer_extra)
        trainer = trainer_class(**trainer_kwargs)
        _assert_no_bos_first_token("after SFTTrainer init (full-label path)", trainer)
    return trainer
