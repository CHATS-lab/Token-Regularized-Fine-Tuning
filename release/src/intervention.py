

import json
import math
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional, Union, Dict
from torch import Tensor
from tqdm import tqdm
# HF causal LM for forward hooks / interventions (not Unsloth; not vLLM).
from utils import read_row, formatInp, load_model_and_tokenizer, extract_primary_question, append_gpt_assistantfinal
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import contextlib
import functools
import numpy as np
import random
import logging
from unsloth import FastLanguageModel
from peft import PeftModel

# Constants
DECODING_STEP = 1 #try -1 TODO
MODEL = 'llama'
STRIP_FULL_SYSTEM_BLOCK = False
CUSTOM_SYSTEM_PROMPT = None
ENABLE_THINKING = None
INCLUDE_THINK_BLOCK_IN_POSTFIX = False
KV_POSTFIX_POSITIONS_IN_POSTFIX: Optional[List[int]] = None
COUNT_ADD=0

MARKER_START = "<<<QUESTION_START>>>"
MARKER_END = "<<<QUESTION_END>>>"


def configure_deterministic_inference(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        logger.warning("Could not enable deterministic algorithms: %s", exc)


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def _strip_leading_bos(text: str, tokenizer) -> str:
    bos = getattr(tokenizer, "bos_token", None)
    if bos and isinstance(text, str) and text.startswith(bos):
        return text[len(bos):]
    return text


def _format_with_overridden_content(d: dict, content: str, tokenizer) -> str:
    d2 = copy.deepcopy(d)
    if isinstance(d2, dict) and 'messages' in d2:
        replaced = False
        for msg in d2['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                msg['content'] = content
                replaced = True
                break
        if not replaced:
            d2['messages'].append({'role': 'user', 'content': content})
    else:
        content_keys = ['prompt', 'question', 'problem', 'input', 'instruction', 'text', 'content']
        for key in content_keys:
            if key in d2:
                d2[key] = content
                break
        else:
            d2['question'] = content
    return _strip_leading_bos(
        formatInp(d2, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
        tokenizer,
    )


def _prefix_positions_and_ids(d: dict, tokenizer) -> Tuple[List[int], List[int]]:
    """Positions and token ids for the prompt span before the marked question."""
    question = extract_primary_question(d)
    if question is None or not str(question).strip():
        raise ValueError("Could not extract question content for prefix token selection.")
    marked_content = f"{MARKER_START}{question}{MARKER_END}"
    marked_prompt = _format_with_overridden_content(d, marked_content, tokenizer)
    marked_ids = tokenizer(marked_prompt).input_ids
    if isinstance(marked_ids, torch.Tensor):
        marked_ids = marked_ids.tolist()
    marker_start_ids = tokenizer(MARKER_START, add_special_tokens=False).input_ids
    start_idx = _find_subsequence(marked_ids, marker_start_ids)
    if start_idx is None:
        question_ids = tokenizer(question, add_special_tokens=False).input_ids
        start_idx = _find_subsequence(marked_ids, question_ids)
    if start_idx is None:
        # Fallback: use a unique placeholder to find where user content starts
        placeholder = "XYZPLACEHOLDER1234567890"
        placeholder_prompt = _format_with_overridden_content(d, placeholder, tokenizer)
        placeholder_ids = tokenizer(placeholder_prompt).input_ids
        if isinstance(placeholder_ids, torch.Tensor):
            placeholder_ids = placeholder_ids.tolist()
        ph_token_ids = tokenizer(placeholder, add_special_tokens=False).input_ids
        start_idx = _find_subsequence(placeholder_ids, ph_token_ids)
    if start_idx is None:
        raise ValueError("Could not locate question marker for prefix token selection.")
    prefix_len = start_idx
    if prefix_len <= 0:
        raise ValueError("Prefix length is zero; cannot select prefix tokens.")
    return list(range(prefix_len)), marked_ids[:prefix_len]


def _compute_prefix_positions_for_instruction(d: dict, tokenizer) -> List[int]:
    positions, _ = _prefix_positions_and_ids(d, tokenizer)
    return positions


def _compute_postfix_positions_for_instruction(d: dict, tokenizer) -> List[int]:
    """Return token positions *after* the user question (e.g. assistant header)."""
    question = extract_primary_question(d)
    if question is None or not str(question).strip():
        raise ValueError("Could not extract question content for postfix token selection.")

    marked_content = f"{MARKER_START}{question}{MARKER_END}"
    marked_prompt = _format_with_overridden_content(d, marked_content, tokenizer)
    marked_ids = tokenizer(marked_prompt).input_ids
    if isinstance(marked_ids, torch.Tensor):
        marked_ids = marked_ids.tolist()

    marker_end_ids = tokenizer(MARKER_END, add_special_tokens=False).input_ids
    end_marker_start = _find_subsequence(marked_ids, marker_end_ids)
    if end_marker_start is None:
        raise ValueError("Could not locate MARKER_END in tokenized prompt for postfix selection.")

    postfix_start_in_marked = end_marker_start + len(marker_end_ids)
    postfix_len = len(marked_ids) - postfix_start_in_marked
    if postfix_len <= 0:
        raise ValueError("Postfix length is zero; no tokens after the question.")

    header_len = postfix_len
    if not INCLUDE_THINK_BLOCK_IN_POSTFIX:
        # Exclude the empty <think>...</think> block from postfix patching (we
        # only patch the assistant header tokens). Truncate at the first
        # <think> token. The think-block ablation can opt into the full region.
        try:
            think_ids = tokenizer('<think>', add_special_tokens=False).input_ids
        except Exception:
            think_ids = []
        if len(think_ids) == 1:
            think_tok = think_ids[0]
            postfix_region = marked_ids[postfix_start_in_marked:]
            if think_tok in postfix_region:
                header_len = postfix_region.index(think_tok)

    original_prompt = _strip_leading_bos(
        formatInp(d, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
        tokenizer,
    )
    original_ids = tokenizer(original_prompt).input_ids
    if isinstance(original_ids, torch.Tensor):
        original_ids = original_ids.tolist()

    original_len = len(original_ids)
    # Map back: assistant header occupies the FIRST header_len of postfix region,
    # which corresponds to the same absolute offset in the unmarked prompt.
    original_postfix_start = original_len - postfix_len
    positions = list(range(original_postfix_start, original_postfix_start + header_len))
    if KV_POSTFIX_POSITIONS_IN_POSTFIX:
        selected = []
        for rel_idx in KV_POSTFIX_POSITIONS_IN_POSTFIX:
            if rel_idx < 0 or rel_idx >= len(positions):
                raise ValueError(
                    f"Postfix position index out of range: {rel_idx} "
                    f"(selected postfix length={len(positions)})"
                )
            selected.append(positions[rel_idx])
        return selected
    return positions


def _get_layer_module(model: AutoModelForCausalLM, layer_idx: int, load_ckpt: int) -> torch.nn.Module:
    if load_ckpt:
        if hasattr(model, "model") and hasattr(model.model, "model"):
            layers = model.model.model.layers
        else:
            layers = model.model.layers
    else:
        layers = model.model.layers if hasattr(model, "model") else model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Layer index out of range: {layer_idx} (layers={len(layers)})")
    return layers[layer_idx]


def _get_attention_module(model: AutoModelForCausalLM, layer_idx: int, load_ckpt: int) -> torch.nn.Module:
    layer_module = _get_layer_module(model, layer_idx, load_ckpt)
    if hasattr(layer_module, "self_attn"):
        return layer_module.self_attn
    if hasattr(layer_module, "attn"):
        return layer_module.attn
    raise AttributeError("Could not locate attention module on the selected layer.")


def get_qkv_replace_hook(
    replacement: Tensor,
    positions: List[int],
    decode_replacement: Optional[Tensor] = None,
):
    """Replace Q/K/V projection outputs at specified positions.
    
    Only activates during the prefill pass (when the full sequence is present).
    During autoregressive decoding with KV cache, the sequence length is 1
    (just the new token), so we skip to avoid corrupting the new token's K/V.
    The replaced values from prefill are already stored in the KV cache.
    """
    max_pos = max(positions) if positions else -1
    print(f'get_qkv_replace_hook: {len(positions)} positions, max_pos={max_pos}')
    decode_calls = 0
    def hook_fn(module: torch.nn.Module, input, output):
        nonlocal decode_calls
        activation = output
        if activation.dim() == 2:
            activation = activation.unsqueeze(0)
        seq_len = activation.size(1)
        if seq_len == 1 and decode_replacement is not None and decode_calls == 0:
            activation_mod = activation.clone()
            device = activation_mod.device
            rep = decode_replacement.to(device=device, dtype=activation_mod.dtype)
            activation_mod[:, 0, :] = rep
            decode_calls += 1
            print('qkv_replace_hook applied: seq_len=1, replaced first decode token')
            return activation_mod
        # Skip during autoregressive decoding unless we explicitly provided
        # one decode-token replacement (e.g. the generated <think> token).
        if seq_len <= max_pos:
            return output
        activation_mod = activation.clone()
        device = activation_mod.device
        rep = replacement.to(device=device, dtype=activation_mod.dtype)
        for idx, pos in enumerate(positions):
            if 0 <= pos < seq_len:
                activation_mod[:, pos, :] = rep[idx]
        print(f'qkv_replace_hook applied: seq_len={seq_len}, replaced {len(positions)} positions')
        return activation_mod
    return hook_fn


def _normalize_qkv_tensor(tensor: Tensor, name: str) -> Tensor:
    if tensor is None:
        raise ValueError(f"QKV data missing {name} tensor.")
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"QKV tensor {name} must be 2D [positions, hidden], got shape {tuple(tensor.shape)}")
    return tensor


def _register_qkv_replacement_hooks_for_layer(
    model: AutoModelForCausalLM,
    args: dict,
    fwd_hooks: List[Tuple[torch.nn.Module, Callable]],
    qkv_data: dict,
    qkv_positions: Optional[List[int]],
    qkv_layer_override: int = -1,
) -> None:
    qkv_layer = qkv_layer_override
    if qkv_layer < 0:
        qkv_layer = qkv_data.get('layer', -1)
    if qkv_layer < 0:
        raise ValueError("QKV layer index missing; set --qkv_layer or store 'layer' in qkv file.")

    positions_saved = qkv_data.get('positions')
    if not positions_saved:
        raise ValueError(f"QKV data missing positions for layer {qkv_layer}.")

    hook_positions = positions_saved
    decode_saved_index = None
    if qkv_positions is not None:
        missing = [p for p in positions_saved if p not in qkv_positions]
        if missing:
            if len(positions_saved) == len(qkv_positions):
                sorted_saved = sorted(positions_saved)
                sorted_current = sorted(qkv_positions)
                remap = {old: new for old, new in zip(sorted_saved, sorted_current)}
                hook_positions = [remap[p] for p in positions_saved]
                if qkv_layer == 0:
                    print(f"Remapped saved positions {positions_saved} -> {hook_positions} at layer {qkv_layer}")
            elif (
                args.get('replace_kv_postfix')
                and len(positions_saved) == len(qkv_positions) + 1
            ):
                sorted_saved = sorted(positions_saved)
                sorted_current = sorted(qkv_positions)
                decode_saved_pos = sorted_saved[-1]
                remap = {old: new for old, new in zip(sorted_saved[:-1], sorted_current)}
                hook_positions = [remap[p] for p in positions_saved if p != decode_saved_pos]
                decode_saved_index = positions_saved.index(decode_saved_pos)
                if qkv_layer == 0:
                    print(
                        f"Remapped saved postfix positions {positions_saved} -> prompt {hook_positions} "
                        f"+ first decode token from saved index {decode_saved_index} at layer {qkv_layer}"
                    )
            else:
                raise ValueError(
                    f"Saved QKV position count ({len(positions_saved)}) != current position count "
                    f"({len(qkv_positions)}) at layer {qkv_layer}; cannot remap. "
                    f"Missing: {missing}"
                )

    q = _normalize_qkv_tensor(qkv_data.get('q'), 'q')
    k = _normalize_qkv_tensor(qkv_data.get('k'), 'k')
    v = _normalize_qkv_tensor(qkv_data.get('v'), 'v')
    if q.shape[0] != len(positions_saved) or k.shape[0] != len(positions_saved) or v.shape[0] != len(positions_saved):
        raise ValueError(f"QKV tensor position count mismatch at layer {qkv_layer}.")

    decode_k = None
    decode_v = None
    if decode_saved_index is not None:
        decode_k = k[decode_saved_index]
        decode_v = v[decode_saved_index]
        keep_indices = [idx for idx in range(len(positions_saved)) if idx != decode_saved_index]
        k = k[keep_indices]
        v = v[keep_indices]

    attn_module = _get_attention_module(model, qkv_layer, args.get('load_ckpt', 0))
    if not hasattr(attn_module, "q_proj") or not hasattr(attn_module, "k_proj") or not hasattr(attn_module, "v_proj"):
        raise AttributeError(f"Attention module at layer {qkv_layer} does not expose q_proj/k_proj/v_proj.")

    fwd_hooks.extend([
        (attn_module.k_proj, get_qkv_replace_hook(k, hook_positions, decode_replacement=decode_k)),
        (attn_module.v_proj, get_qkv_replace_hook(v, hook_positions, decode_replacement=decode_v)),
    ])


def _randomize_loaded_qkv_data_in_place(
    qkv_data: dict,
    layer_idx: int,
    seed: int,
    mean: float,
    std: float,
) -> None:
    base_seed = seed + layer_idx * 1000003
    for key, offset in [('q', 11), ('k', 23), ('v', 37)]:
        tensor = _normalize_qkv_tensor(qkv_data.get(key), key)
        generator = None
        if seed >= 0:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(base_seed + offset)
        randomized = torch.randn(tensor.shape, generator=generator) * std + mean
        qkv_data[key] = randomized.to(dtype=tensor.dtype)


def clean_decoded_text(text: str) -> str:
    if not text:
        return text
    return text.replace('\u010a', '\n').replace('\u0120', ' ').strip()


def get_model_max_length(model: AutoModelForCausalLM) -> int:
    """
    Get the maximum context length from the model's configuration.
    
    Args:
        model: The loaded model
        
    Returns:
        Maximum context length
    """
    # Try different possible attribute names for max length
    for attr_name in ['max_position_embeddings', 'max_sequence_length', 'context_length', 'seq_length', 'n_positions']:
        if hasattr(model.config, attr_name):
            return getattr(model.config, attr_name)
    return 4096  # Default fallback



@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    use_kwargs: bool = True,
    **kwargs
):
    """Context manager for adding and removing hooks from modules."""
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook, with_kwargs=use_kwargs))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def scale_attention_by_keys_pre_hook(positions, scale=1.0, num_heads=32,heads=None):
    """
    Bias attention *coefficients* toward specific KEY token positions.
    Multiplies p[:, h, q, k in positions] by `scale` (then renormalizes via softmax).

    Args:
      positions: list[int] key indices to boost/suppress globally (for all queries).
      scale: float > 0.  >1 boosts attention to those keys; <1 suppresses it.
      heads: None (all heads) or list[int] of head indices to affect.
    """
    if scale ==0:
        ln_alpha = float("-inf")
    else:
        ln_alpha = math.log(scale)

    def pre_hook(module, args, kwargs):
        # hidden_states: (B, T_q, D)
        global COUNT_ADD
        if not (DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP):
            return args, kwargs
        
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs["hidden_states"]
        #print('hidden_states shape', hidden_states.shape)
        B, q_len, _ = hidden_states.shape

        # Determine heads and k_len (often from attention_mask if present)
        H = num_heads

        attn_mask = kwargs.get("attention_mask", None)
        k_len = attn_mask.size(-1) if attn_mask is not None else q_len  # fallback if no KV cache

        device, dtype = hidden_states.device, hidden_states.dtype
        bias = torch.zeros((B, H, q_len, k_len), device=device, dtype=dtype)

        # which heads to touch
        head_idx = range(H) if heads is None else heads

        # add log(scale) to the selected KEY columns
        for j in positions:
            if 0 <= j < k_len:
                bias[:, head_idx, :, j] += ln_alpha

        # merge with existing attention_mask
        #print('attention_mask before', kwargs["attention_mask"][:, head_idx, :, positions])
        if attn_mask is None:
            kwargs["attention_mask"] = bias
        else:
            kwargs["attention_mask"] = attn_mask + bias
        
        COUNT_ADD+=1
        return args, kwargs

    return pre_hook


def get_activation_addition_input_pre_hook(
    vector: Tensor, 
    coeff: Tensor,
    cache: Optional[List] = None,
    record: int = 0,
    intervene_all: bool = True,
    positions: List[int] = None
):
    """Hook function to add activation vectors to model inputs."""
    if cache is None:
        cache = []
    if positions is None:
        positions = [-1]

    def hook_fn(module, input):
        nonlocal vector, cache, coeff
        global COUNT_ADD

        if isinstance(input, tuple):
            activation = input[0]
            remainder = input[1:]
        else:
            activation = input
            remainder = ()

        device = activation.device
        vector_tensor = vector.to(device)

        if isinstance(coeff, torch.Tensor):
            if coeff.requires_grad and coeff.device != device:
                raise ValueError("Trainable coefficients must be on the same device as the activation.")
            coeff_tensor = coeff if coeff.device == device else coeff.to(device)
        else:
            coeff_tensor = torch.as_tensor(coeff, device=device, dtype=vector_tensor.dtype)

        if DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP:
            activation_mod = activation.clone()
            if intervene_all:
                if coeff_tensor.numel() != 1:
                    raise ValueError("intervene_all currently supports only a scalar coefficient.")
                coeff_tensor = torch.sigmoid(coeff_tensor)
                activation_mod[:, positions[0]:, :] = activation_mod[:, positions[0]:, :] + coeff_tensor.view(1, 1, 1) * vector_tensor
            else:
                coeff_flat = coeff_tensor.view(-1)
                use_single_coeff = coeff_flat.numel() == 1
                if not use_single_coeff and coeff_flat.numel() != len(positions):
                    raise ValueError("Number of coefficients must match number of positions or be a scalar.")
                for idx, pos in enumerate(positions):
                    if pos < 0 or pos >= activation_mod.size(1):
                        continue
                    coeff_value = torch.sigmoid(coeff_flat[0] if use_single_coeff else coeff_flat[idx])
                    activation_mod[:, pos, :] = activation_mod[:, pos, :] + coeff_value * vector_tensor
            activation = activation_mod
            COUNT_ADD += 1

        if record:
            cache.append(activation)

        if isinstance(input, tuple):
            return (activation, *remainder)
        else:
            return activation
    return hook_fn


def get_activation_addition_forward_hook(
    vector: Tensor,
    coeff: Tensor,
    cache: Optional[List] = None,
    record: int = 0,
    intervene_all: bool = True,
    positions: List[int] = None,
):
    """Forward hook variant of activation addition; modifies and returns the output."""
    if cache is None:
        cache = []
    if positions is None:
        positions = [-1]

    def hook_fn(module: torch.nn.Module, input, output):
        nonlocal vector, cache, coeff
        global COUNT_ADD

        activation = output[0] if isinstance(output, tuple) else output
        device = activation.device
        vector_tensor = vector.to(device)

        if isinstance(coeff, torch.Tensor):
            if coeff.requires_grad and coeff.device != device:
                raise ValueError("Trainable coefficients must be on the same device as the activation.")
            coeff_tensor = coeff if coeff.device == device else coeff.to(device)
        else:
            coeff_tensor = torch.as_tensor(coeff, device=device, dtype=vector_tensor.dtype)

        if DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP:
            if activation.dim() == 2:
                activation = activation.unsqueeze(0)
            activation_mod = activation.clone()

            if intervene_all:
                if coeff_tensor.numel() != 1:
                    raise ValueError("intervene_all currently supports only a scalar coefficient.")
                coeff_tensor = torch.sigmoid(coeff_tensor)
                activation_mod[:, positions[0]:, :] = activation_mod[:, positions[0]:, :] + coeff_tensor.view(1, 1, 1) * vector_tensor
            else:
                coeff_flat = coeff_tensor.view(-1)
                use_single_coeff = coeff_flat.numel() == 1
                if not use_single_coeff and coeff_flat.numel() != len(positions):
                    raise ValueError("Number of coefficients must match number of positions or be a scalar.")
                for idx, pos in enumerate(positions):
                    if pos < 0 or pos >= activation_mod.size(1):
                        continue
                    coeff_value = torch.sigmoid(coeff_flat[0] if use_single_coeff else coeff_flat[idx])
                    activation_mod[:, pos, :] = activation_mod[:, pos, :] + coeff_value * vector_tensor

            activation = activation_mod
            COUNT_ADD += 1

        if record:
            cache.append(activation)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        return activation

    return hook_fn


def get_activation_replace_input_pre_hook(
    vector: Tensor,
    positions: List[int],
):
    """Hook to replace activations at positions with provided vectors."""
    if positions is None:
        positions = [-1]

    def hook_fn(module, input):
        global COUNT_ADD
        if isinstance(input, tuple):
            activation = input[0]
            remainder = input[1:]
        else:
            activation = input
            remainder = ()
        print('#########get_activation_replace_input_pre_hook#########')
        print('activation shape', activation.shape)

        if activation.dim() == 2:
            activation = activation.unsqueeze(0)
        
        activation_mod = activation.clone()
        device = activation_mod.device
        vector_tensor = vector.to(device=device, dtype=activation_mod.dtype)
        if DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP:
            print("#########replace activation#########")
            positions_tensor = torch.tensor(positions, device=device)
            valid_mask = (positions_tensor >= 0) & (positions_tensor < activation_mod.size(1))
            if valid_mask.any():
                valid_positions = positions_tensor[valid_mask]
                activation_mod[:, valid_positions, :] = vector_tensor[valid_mask]
                print("valid mask", valid_mask)
                print("valid positions", valid_positions)
                print("vector tensor shape", vector_tensor.shape)
                print("activation mod shape", activation_mod.shape)
            COUNT_ADD += 1

        if isinstance(input, tuple):
            return (activation_mod, *remainder)
        return activation_mod
 
    return hook_fn


def get_activation_replace_forward_hook(
    vector: Tensor,
    positions: List[int],
):
    """Forward hook variant to replace activations at positions."""
    if positions is None:
        positions = [-1]

    def hook_fn(module: torch.nn.Module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        if activation.dim() == 2:
            activation = activation.unsqueeze(0)

        activation_mod = activation.clone()
        device = activation_mod.device
        vector_tensor = vector.to(device=device, dtype=activation_mod.dtype)

        for idx, pos in enumerate(positions):
            if 0 <= pos < activation_mod.size(1):
                activation_mod[:, pos, :] = vector_tensor[idx]

        if isinstance(output, tuple):
            return (activation_mod, *output[1:])
        return activation_mod

    return hook_fn


def build_intervention_hooks(
    model: AutoModelForCausalLM,
    intervene_layers: List[int],
    intervention_vector: Tensor,
    coeff: Tensor,
    intervene_all: bool,
    positions: List[int],
    args: dict,
):
    """Construct forward and forward-pre hooks for the specified layers."""
    fwd_pre_hooks = []
    fwd_hooks = []

    if args.get('scale_attention_heads'):
        return fwd_pre_hooks, fwd_hooks

    for intervene_layer in intervene_layers:
        if args.get('load_ckpt'):
            if intervene_layer == model.config.num_hidden_layers:
                module = model.model.model.layers[intervene_layer-1]
            else:
                module = model.model.model.layers[intervene_layer]
        else:
            if intervene_layer == model.config.num_hidden_layers:
                module = model.model.layers[intervene_layer-1]
            else:
                module = model.model.layers[intervene_layer]

        layer_vector = intervention_vector[intervene_layer, :]

        if intervene_layer == model.config.num_hidden_layers:
            fwd_hooks.append(
                (
                    module,
                    get_activation_addition_forward_hook(
                        vector=layer_vector,
                        coeff=coeff,
                        intervene_all=intervene_all,
                        positions=positions,
                    ),
                )
            )
        else:
            fwd_pre_hooks.append(
                (
                    module,
                    get_activation_addition_input_pre_hook(
                        vector=layer_vector,
                        coeff=coeff,
                        intervene_all=intervene_all,
                        positions=positions,
                    ),
                )
            )

    return fwd_pre_hooks, fwd_hooks


def complete_with_intervention(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    instructions: List[dict], 
    tokenize_instructions_fn: Callable,
    intervene_layers: List[int], 
    batch_size: int = 32, 
    intervention_vector_ori: Optional[Tensor] = None,
    args: Optional[dict] = None
) -> List[dict]:
    """
    Complete text generation with intervention vectors.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for the model
        instructions: List of instruction dictionaries
        tokenize_instructions_fn: Function to tokenize instructions
        intervene_layers: List of layer indices to intervene on
        batch_size: Batch size for processing
        intervention_vector_ori: Original intervention vectors
        args: Configuration arguments dictionary
        
    Returns:
        List of completion dictionaries containing generated text and metadata
    """

    #if MODEL == 'gpt':
        #args['max_token_generate'] =5000
    model.eval()
    if args['intervene_all']:
        print('generate till the end of the model context window')
        generation_config = GenerationConfig(
            max_length=get_model_max_length(model),
            do_sample=False
        )
    else:
        generation_config = GenerationConfig(
            max_new_tokens=args['max_token_generate'], 
            do_sample=False
        )
    generation_config.pad_token_id = tokenizer.pad_token_id
    
    n_layers = model.config.num_hidden_layers
    logging.info(f"Number of layers: {n_layers}")

    ret = []
    cache = [[] for _ in range(n_layers)]
    global COUNT_ADD
    if args.get('load_coeff'):
        coeff_dt=read_row(args.get('load_coeff_pth'))
    for i in tqdm(range(0, len(instructions), batch_size)):

        if not args.get('skip_activation_intervention', 0):
            if intervention_vector_ori is None:
                raise ValueError("intervention_vector is required unless skip_activation_intervention is set.")
            if args.get('intervention_mode') != 'replace_prefix' and intervention_vector_ori.shape[0] > 1:
                intervention_vector = intervention_vector_ori[i].squeeze()
            else:
                intervention_vector = intervention_vector_ori.squeeze()

            print('intervention vector shape', intervention_vector_ori.shape)
            print('intervention vector shape reshape', intervention_vector.shape)
        else:
            intervention_vector = None
        input_prompt,tokenized_instructions = tokenize_instructions_fn(instructions[i:i + batch_size])
        tokenized_model_input = tokenized_instructions
        if MODEL == 'gpt':
            input_prompt = [append_gpt_assistantfinal(p, MODEL) for p in input_prompt]
            tokenized_model_input = tokenizer(input_prompt, padding=True, return_tensors="pt")
    
        print("\n\n example input_prompt:\n", input_prompt)

        seq_len = tokenized_instructions.input_ids.shape[-1]
        # By default, intervene on all input tokens
        intervene_position = list(range(seq_len))
        if args.get('intervention_mode') == 'replace_prefix':
            if batch_size != 1:
                print("Warning: prefix positions computed for first item only; batch_size>1 not fully supported")
            intervene_position = _compute_prefix_positions_for_instruction(instructions[i], tokenizer)
            print(f"replace_prefix mode: {len(intervene_position)} prefix tokens selected.")
        qkv_positions = None
        if args.get('replace_qkv_prefix', 0):
            if batch_size != 1:
                print("Warning: QKV prefix positions computed for first item only; batch_size>1 not fully supported")
            qkv_positions, prefix_token_ids = _prefix_positions_and_ids(instructions[i], tokenizer)
            if i == 0:
                prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_token_ids)
                print(
                    f"replace_qkv_prefix: prefix tokens (first example, printed once): "
                    f"count={len(prefix_token_ids)}"
                )
                print(f"  ids: {prefix_token_ids}")
                print(f"  tokens: {prefix_tokens}")
        kv_postfix_positions = None
        if args.get('replace_kv_postfix', 0):
            if batch_size != 1:
                print("Warning: KV postfix positions computed for first item only; batch_size>1 not fully supported")
            kv_postfix_positions = _compute_postfix_positions_for_instruction(instructions[i], tokenizer)
            print(f"replace_kv_postfix: {len(kv_postfix_positions)} postfix tokens selected.")
        # Optional: limit intervention to only the perturbed_step span
        if args.get('intervene_on_perturbed_step', 0) or args.get('intervene_on_jb_prompt', 0):
            logging.info('intervene on perturbed step')
            try:
                if batch_size != 1:
                    print("Warning: intervene_position computed for first item only; batch_size>1 not fully supported")
                d0 = instructions[i]
                step_str = d0[args['step_key']]
                step_ids = tokenizer(step_str, add_special_tokens=False).input_ids
                full_ids = tokenized_instructions.input_ids[0].tolist()
                attn = tokenized_instructions.attention_mask[0].tolist()
                seq_len_eff = sum(attn)
                search_space = full_ids[:seq_len_eff]
                start_idx = None
                n = len(step_ids)
                for s in range(seq_len_eff - n, -1, -1):
                    if search_space[s:s + n] == step_ids:
                        start_idx = s
                        break
                if start_idx is None:
                    start_idx = max(0, seq_len_eff - n)
                intervene_position = list(range(start_idx, start_idx + n))
                print('intervene_position (perturbed_step):', intervene_position)
            except Exception as e:
                print(f"Failed to compute perturbed_step intervene positions: {e}")
                intervene_position = list(range(seq_len))

        # If context-only requested and not targeting perturbed_step, trim to context span
        if args['intervene_context_only'] and not args['intervene_on_perturbed_step']:
            logging.info('intervene on context only')
            if MODEL == 'qwen':
                special_token = "<|im_end|>\n<|im_start|>assistant"
            elif MODEL == 'llama3':
                special_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            elif MODEL == 'llama2':
                special_token = "[/INST]"
            special_token_len = len(tokenizer.tokenize(special_token))
            print('special token length', special_token_len)
            intervene_position = list(range(seq_len - special_token_len))
        else:
            logging.info('intervene on all tokens')
            
        print(50 * "#")
        coeff = None
        coeff_rand = None
        coeff_rand_positions = []
        if not args.get('skip_activation_intervention', 0):
            print('intervene with steering vectors')
            print('intervene position', intervene_position)
            if args.get('load_coeff'):
                coeff = torch.full((len(intervene_position),), 0.0, device=model.device)
                coeff_tmp = torch.tensor(coeff_dt[i]['coefficients'], device=model.device)
                coeff[coeff_dt[i]['positions']] = coeff_tmp
                assert coeff.shape[0] == len(intervene_position)
                if args.get('coeff_rand_count') > 0 and len(intervene_position) > 0:
                    rand_count = min(args.get('coeff_rand_count'), len(intervene_position))
                    rand_indices = sorted(random.sample(coeff_dt[i]['positions'], rand_count))
                    coeff_rand = torch.zeros_like(coeff)
                    coeff_rand[rand_indices] = coeff[rand_indices]
                    coeff_rand_positions = [intervene_position[idx] for idx in rand_indices]
            else:
                coeff = torch.full((len(intervene_position),), args.get('coeff_select', 1), device=model.device)
        
        if coeff is not None:
            print('coeff shape', coeff.shape, 'intervene position length', len(intervene_position))
        else:
            print('skipping activation intervention, intervene position length', len(intervene_position))
        fwd_pre_hooks = []
        fwd_hooks = []
        
        # Add attention head scaling hooks if enabled
        if args.get('scale_attention_heads', 0):
            print('Adding attention head scaling hooks')
            scale_factor = args.get('attention_scale_factor', 0.0)
            print(f'Attention scale factor: {scale_factor}')
            
            # Get model configuration for attention heads
            num_heads = getattr(model.config, 'num_attention_heads', 32)
            head_dim = getattr(model.config, 'hidden_size', 4096) // num_heads

        
        if not args.get('skip_activation_intervention'):
            if args.get('intervention_mode') == 'replace_prefix':
                if intervention_vector.dim() != 3:
                    raise ValueError(f"replace_prefix expects vector shape [layers, positions, dim], got {intervention_vector.shape}")
                if intervention_vector.shape[1] != len(intervene_position):
                    raise ValueError(
                        f"replace_prefix position length mismatch: "
                        f"vector has {intervention_vector.shape[1]} positions, "
                        f"but prefix length is {len(intervene_position)}"
                    )
                if max(intervene_layers) >= intervention_vector.shape[0]:
                    raise ValueError(
                        f"replace_prefix layer index out of range: "
                        f"max layer {max(intervene_layers)} vs vector layers {intervention_vector.shape[0]}"
                    )
            print("traverse intervene layers", intervene_layers)
            for intervene_layer in intervene_layers:
                if args['scale_attention_heads']:
                    break #only scale attention heads hardcoded for now
                if args['load_ckpt']:
                    if intervene_layer > n_layers:
                        break
                    if intervene_layer == n_layers:
                        module = model.model.model.layers[intervene_layer - 1]
                    else:
                        module = model.model.model.layers[intervene_layer]#TODO: check if this is correct, initial -1
                else:
                    if intervene_layer == n_layers:
                        module = model.model.layers[intervene_layer - 1]
                    else:
                        module = model.model.layers[intervene_layer]
                    
                if intervene_layer == n_layers:
                    # use forward hook on last layer
                    if args.get('intervention_mode') == 'replace_prefix':
                        layer_vector = intervention_vector[intervene_layer, :, :]
                        #layer_vector=layer_vector[-3:,] #Hard coding 
                        #intervene_position=intervene_position[-3:]
                        fwd_hooks.append(
                            (
                                module,
                                get_activation_replace_forward_hook(
                                    vector=layer_vector,
                                    positions=intervene_position,
                                ),
                            )
                        )
                        continue
                    fwd_hooks.append(
                        (
                            module,
                            get_activation_addition_forward_hook(
                                vector=intervention_vector[intervene_layer, :],
                                coeff=coeff,
                                intervene_all=args['intervene_all'],
                                positions=intervene_position,
                            ),
                        )
                    )
                else:
                    # use input pre-hook on other layers
                    if args.get('intervention_mode') == 'replace_prefix':
                        layer_vector = intervention_vector[intervene_layer, :, :]
                        #layer_vector=layer_vector[-3:,] #Hard coding 
                        #intervene_position=intervene_position[-3:]
                        fwd_pre_hooks.append(
                            (
                                module,
                                get_activation_replace_input_pre_hook(
                                    vector=layer_vector,
                                    positions=intervene_position,
                                ),
                            )
                        )
                        continue
                    fwd_pre_hooks.append(
                        (
                            module,
                            get_activation_addition_input_pre_hook(
                                vector=intervention_vector[intervene_layer, :],
                                coeff=coeff,
                                intervene_all=args['intervene_all'],
                                positions=intervene_position,
                            ),
                        )
                    )

        if args.get('replace_qkv_prefix'):
            if args.get('replace_qkv_prefix_all_layers', 0):
                qkv_data_by_layer = args.get('qkv_data_by_layer')
                if not qkv_data_by_layer:
                    raise ValueError("QKV all-layer data not loaded. Provide --qkv_dir and enable --replace_qkv_prefix_all_layers.")
                for layer_idx in sorted(qkv_data_by_layer.keys()): #for debugging, hard code with specific layers
                    _register_qkv_replacement_hooks_for_layer(
                        model=model,
                        args=args,
                        fwd_hooks=fwd_hooks,
                        qkv_data=qkv_data_by_layer[layer_idx],
                        qkv_positions=qkv_positions,
                        qkv_layer_override=layer_idx,
                    )
            else:
                qkv_data = args.get('qkv_data')
                if qkv_data is None:
                    raise ValueError("QKV data not loaded. Provide --qkv_pth and enable replace_qkv_prefix.")
                _register_qkv_replacement_hooks_for_layer(
                    model=model,
                    args=args,
                    fwd_hooks=fwd_hooks,
                    qkv_data=qkv_data,
                    qkv_positions=qkv_positions,
                    qkv_layer_override=args.get('qkv_layer', -1),
                )

        if args.get('replace_kv_postfix'):
            if args.get('replace_kv_postfix_all_layers', 0):
                kv_postfix_data_by_layer = args.get('kv_postfix_data_by_layer')
                if not kv_postfix_data_by_layer:
                    raise ValueError("KV postfix all-layer data not loaded. Provide --kv_postfix_dir and enable --replace_kv_postfix_all_layers.")
                for layer_idx in sorted(kv_postfix_data_by_layer.keys()):
                    _register_qkv_replacement_hooks_for_layer(
                        model=model,
                        args=args,
                        fwd_hooks=fwd_hooks,
                        qkv_data=kv_postfix_data_by_layer[layer_idx],
                        qkv_positions=kv_postfix_positions,
                        qkv_layer_override=layer_idx,
                    )
            else:
                kv_postfix_data = args.get('kv_postfix_data')
                if kv_postfix_data is None:
                    raise ValueError("KV postfix data not loaded. Provide --kv_postfix_pth and enable replace_kv_postfix.")
                _register_qkv_replacement_hooks_for_layer(
                    model=model,
                    args=args,
                    fwd_hooks=fwd_hooks,
                    qkv_data=kv_postfix_data,
                    qkv_positions=kv_postfix_positions,
                    qkv_layer_override=args.get('kv_postfix_layer', -1),
                )

        completions = []

        print('index', i)
        print('length of instructions', len(instructions))
        assert i < len(instructions)

        if not args['intervene_all']:
            for id in intervene_position:
                print('intervening token', tokenizer.convert_ids_to_tokens(tokenized_instructions.input_ids[0])[id])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks,use_kwargs=args['scale_attention_heads']):
            
            if 0:
                # Use direct model() call for attention head scaling with token generation loop
                print("Using direct model() call for attention head scaling")
                max_new_tokens = args['max_token_generate']
                current_input_ids = tokenized_model_input.input_ids.to(model.device)
                current_attention_mask = tokenized_model_input.attention_mask.to(model.device)
                
                generated_tokens = []
                generated_scores = []
                prob = []
                tokens = []
                
                with torch.no_grad():
                    for step in range(max_new_tokens):
                        outputs = model(
                            input_ids=current_input_ids,
                            attention_mask=current_attention_mask,
                            output_attentions=True,
                            return_dict=True
                        )
                        
                        # Get logits for the last token position
                        logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                        generated_scores.append(logits)
                        
                        # Get the most likely next token
                        next_token_id = torch.argmax(logits, dim=-1)  # Shape: [batch_size]
                        generated_tokens.append(next_token_id)
                        
                        # Record probabilities if requested
                        if args['record_probs'] and step < 10:
                            probability = torch.softmax(logits[0], dim=-1)
                            all_probs = probability.detach().cpu().numpy().tolist()
                            token_id = next_token_id[0].item()
                            token = tokenizer.decode(token_id)
                            prob.append(all_probs)
                            tokens.append(token)
                        
                        # Update input for next iteration
                        current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token_id.unsqueeze(-1))], dim=-1)
                        
                        # Check for EOS token
                        if tokenizer.eos_token_id is not None and next_token_id[0].item() == tokenizer.eos_token_id:
                            print(f"EOS token generated at step {step}")
                            break
                
                # Combine and decode generated tokens
                if generated_tokens:
                    generation_token_ids = torch.cat(generated_tokens, dim=-1)  # Shape: [batch_size, num_generated]
                    print('generation_token_ids shape', generation_token_ids.shape)
                    # Decode tokens to match the format expected by the rest of the code
                    generation_toks = []
                    for batch_idx in range(generation_token_ids.shape[0]):
                        decoded_text = clean_decoded_text(tokenizer.decode(generation_token_ids[batch_idx], skip_special_tokens=True))
                        generation_toks.append(decoded_text)
                else:
                    generation_toks = [""] * current_input_ids.shape[0]  # Empty strings for each batch item
                
                print('length of input token', tokenized_instructions.input_ids.shape[-1])
                if generation_toks:
                    print('generated text length:', len(generation_toks[0]) if generation_toks[0] else 0)
                
            else:
                # Use standard generation for activation vector interventions
                print('tokenized_model_input.input_ids', tokenized_model_input.input_ids [:10])
                generation_toks = model.generate(
                    input_ids=tokenized_model_input.input_ids.to(model.device),
                    attention_mask=tokenized_model_input.attention_mask.to(model.device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=True,
                    use_cache= 1
                )
                    
                generated_scores = generation_toks.scores
                prob = []
                tokens = []
                
                if args['record_probs']:
                    for ii, score in enumerate(generated_scores):
                        if ii < 10:
                            probability = torch.softmax(score[0], dim=-1)
                            all_probs = probability.detach().cpu().numpy().tolist()
                            token_id = torch.argmax(probability).item()
                            prob_id = probability[token_id].item()
                            token = tokenizer.decode(token_id)
                            prob.append(all_probs)
                            tokens.append(token)
                        else:
                            break
                            
                print('length of input token', tokenized_instructions.input_ids.shape[-1])
                generation_toks = generation_toks.sequences[:, tokenized_model_input.input_ids.shape[-1]:]  # remove input tokens
                print('length of generated token', generation_toks.shape[-1])

            generation_idx, generation =0,generation_toks[0]

            response_text = tokenizer.decode(generation, skip_special_tokens=True).strip()
            
            completions.append({
                'input_prompt': input_prompt,
                'prompt': instructions[i + generation_idx],
                'response': response_text,
                'tokens': tokens,
                'probs': prob,
            })
            COUNT_ADD=0
        random_response = None
        if coeff_rand is not None:
            try:
                if torch.count_nonzero(coeff_rand).item() > 0:
                    COUNT_ADD = 0
                    fwd_pre_hooks_rand, fwd_hooks_rand = build_intervention_hooks(
                        model=model,
                        intervene_layers=intervene_layers,
                        intervention_vector=intervention_vector,
                        coeff=coeff_rand,
                        intervene_all=args['intervene_all'],
                        positions=intervene_position,
                        args=args,
                    )
                    with torch.no_grad():
                        with add_hooks(fwd_pre_hooks_rand, fwd_hooks_rand, use_kwargs=args.get('scale_attention_heads', 0)):
                            generation_rand = model.generate(
                                input_ids=tokenized_model_input.input_ids.to(model.device),
                                attention_mask=tokenized_model_input.attention_mask.to(model.device),
                                generation_config=generation_config,
                                return_dict_in_generate=True,
                                output_scores=False,
                                output_attentions=False,
                                use_cache=1 #args['max_decode_step_while_intervene']==1
                            )
                    generated_rand = generation_rand.sequences[:, tokenized_model_input.input_ids.shape[-1]:]
                    random_response = tokenizer.decode(generated_rand[0], skip_special_tokens=True).strip() if generated_rand.numel() > 0 else ""
            except Exception as rand_exc:
                print(f"Failed to generate with random coefficients: {rand_exc}")
                random_response = None
            finally:
                COUNT_ADD = 0
        if completions:
            completions[-1]['response_rand'] = random_response
            completions[-1]['coeff_rand_positions'] = coeff_rand_positions
            completions[-1]['coeff_rand_values'] = (
                coeff_rand.detach().cpu().tolist() if coeff_rand is not None else []
            )
        ret.extend(completions)
        # optional: emit minimal cache info if used elsewhere
            
    return ret


def train_intervention_coefficients(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[dict],
    tokenize_instructions_fn: Callable,
    intervene_layers: List[int],
    batch_size: int,
    intervention_vector_ori: Optional[Tensor],
    args: dict,
):
    """
    Optimize per-position intervention coefficients to maximize the likelihood of target sequences.
    """
    if batch_size != 1:
        raise ValueError("Coefficient training currently supports batch_size=1.")

    if not args.get('target_key'):
        raise ValueError("--target_key must be specified for training mode.")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    results = []
    global COUNT_ADD

    coeff_lr = args.get('coeff_lr', 0.05)
    train_steps = args.get('train_steps', 100)
    coeff_init_std = args.get('coeff_init_std', 0.02)
    target_key = args['target_key']
    coeff_top_k_values =[1] # generate the repsonse with the top-k tokens with the highest top k coefficients 

    for i in tqdm(range(0, len(instructions), batch_size)):
        instruction = instructions[i]

        if intervention_vector_ori is None:
            raise ValueError("intervention_vector must be provided for training mode.")

        if intervention_vector_ori.shape[0] > 1:
            intervention_vector = intervention_vector_ori[i].squeeze()
        else:
            intervention_vector = intervention_vector_ori.squeeze()

        print('intervention vector shape', intervention_vector_ori.shape)
        print('intervention vector shape reshape', intervention_vector.shape)

        input_prompt, tokenized_instructions = tokenize_instructions_fn([instruction])
        print("\n\n example input_prompt:\n", input_prompt)

        seq_len = tokenized_instructions.input_ids.shape[-1]
        intervene_position = list(range(seq_len))

        input_tokens = tokenizer.convert_ids_to_tokens(tokenized_instructions.input_ids.squeeze().tolist())
        user_header_idx = None
        for idx in range(len(input_tokens) - 1):
            if input_tokens[idx] == "<|start_header_id|>" and input_tokens[idx + 1] == "user":
                user_header_idx = idx
                break
        if args['intervene_on_user_header']:
            intervene_position = list(range(0, user_header_idx))
        else: #intervene on all tokens after user header
            intervene_position = list(range(user_header_idx if user_header_idx is not None else 0, seq_len))
        
        print('intervene position', intervene_position)

        print(50 * "#")
        print('intervene with steering vectors')

        if target_key not in instruction:
            raise KeyError(f"Target key '{target_key}' not found in instruction.")

        target_text = instruction[target_key]
        if target_text is None or (isinstance(target_text, str) and not target_text.strip()):
            raise ValueError("Target text is empty; cannot optimize coefficients.")

        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids
        prompt_ids = tokenized_instructions.input_ids.to(model.device)
        attention_mask = tokenized_instructions.attention_mask.to(model.device)
        target_ids = target_ids.to(model.device)

        input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        attention_full = torch.ones_like(input_ids, device=model.device)
        labels = input_ids.clone()
        labels[:, :prompt_ids.size(-1)] = -100

        if len(intervention_vector.shape) == 2:
            intervention_vector = intervention_vector.unsqueeze(0)

        if args['intervention_vector_layer'] >= 0:
            intervention_vector_local = intervention_vector[:, args['intervention_vector_layer'], :].to(model.device)
        elif args['traverse_single_layer_intervention']:
            raise NotImplementedError("traverse_single_layer_intervention is not supported in training mode.")
        else:
            if intervention_vector.shape[0] != 1:
                intervention_vector_local = intervention_vector[args['left']:args['right']].to(model.device)
            else:
                intervention_vector_local = intervention_vector.to(model.device)

        if intervention_vector_local.dim() == 3:
            intervention_vector_local = intervention_vector_local[0]
        elif intervention_vector_local.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected intervention vector shape: {intervention_vector_local.shape}")

        coeff_param = torch.nn.Parameter(
            torch.full((len(intervene_position),), args.get('coeff_init_value', 0.5), device=model.device)
        )
        optimizer = torch.optim.Adam([coeff_param], lr=coeff_lr)

        fwd_pre_hooks, fwd_hooks = build_intervention_hooks(
            model=model,
            intervene_layers=intervene_layers,
            intervention_vector=intervention_vector_local,
            coeff=coeff_param,
            intervene_all=args['intervene_all'],
            positions=intervene_position,
            args=args,
        )

        progress_file = args['output_pth'].replace('.json', '-progress.json')

        best_loss = float('inf')
        best_coeff = None

        for step in range(train_steps):
            optimizer.zero_grad()
            COUNT_ADD = 0

            with add_hooks(fwd_pre_hooks, fwd_hooks, use_kwargs=args.get('scale_attention_heads', 0)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_full,
                    labels=labels,
                    use_cache=False,
                )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            if loss_value < best_loss:
                best_loss = loss_value
                best_coeff = coeff_param.detach().cpu().clone()

            if args.get('cap_coeff', 0):
                coeff_param=torch.clamp(coeff_param, min=0,max=1) #TODO: cap the coeff to 0-1
                
            if args.get('verbose_train', 0):
                print(f"Step {step}: loss={loss_value:.6f}")

            if (step + 1) % 100 == 0 or step==0 or step==train_steps-1:
                try:
                    COUNT_ADD = 0
                    coeff_eval = torch.sigmoid(coeff_param.detach().clone().to(model.device))
                    fwd_pre_hooks_eval, fwd_hooks_eval = build_intervention_hooks(
                        model=model,
                        intervene_layers=intervene_layers,
                        intervention_vector=intervention_vector_local,
                        coeff=coeff_eval,
                        intervene_all=args['intervene_all'],
                        positions=intervene_position,
                        args=args,
                    )

                    gen_kwargs = dict(
                        input_ids=tokenized_instructions.input_ids.to(model.device),
                        attention_mask=tokenized_instructions.attention_mask.to(model.device),
                        max_new_tokens=args['max_token_generate'],
                        do_sample=False,
                    )
                    if getattr(tokenizer, "pad_token_id", None) is not None:
                        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

                    with torch.no_grad():
                        with add_hooks(fwd_pre_hooks_eval, fwd_hooks_eval, use_kwargs=args.get('scale_attention_heads', 0)):
                            step_generation = model.generate(**gen_kwargs)

                    prompt_len = tokenized_instructions.input_ids.shape[-1]
                    step_generated_ids = step_generation[:, prompt_len:]
                    step_response = tokenizer.decode(step_generated_ids[0], skip_special_tokens=True).strip() if step_generated_ids.numel() > 0 else ""

                    progress_entry = {
                        'sample_index': i // batch_size,
                        'step': step + 1,
                        'loss': loss_value,
                        'response': step_response,
                        'token_ids': step_generated_ids[0].tolist() if step_generated_ids.numel() > 0 else [],
                    }
                    with open(progress_file, 'a') as pf:
                        json.dump(progress_entry, pf)
                        pf.write('\n')
                    print(f"Step {step + 1}: logged progress to {progress_file}")
                except Exception as eval_exc:
                    print(f"Failed to generate progress at step {step + 1}: {eval_exc}")

        if best_coeff is None:
            best_coeff = coeff_param.detach().cpu().clone()
        # store and use sigmoid-bounded coefficients going forward
        best_coeff = torch.sigmoid(best_coeff)

        trained_response = None
        generated_ids = None
        trained_response_top_k = {}
        trained_response_non_negative = None
        trained_token_ids_tensor = None
        coefficients_top_k = {}
        try:
            COUNT_ADD = 0
            coeff_for_gen = torch.sigmoid(best_coeff.to(model.device))
            fwd_pre_hooks_gen, fwd_hooks_gen = build_intervention_hooks(
                model=model,
                intervene_layers=intervene_layers,
                intervention_vector=intervention_vector_local,
                coeff=coeff_for_gen,
                intervene_all=args['intervene_all'],
                positions=intervene_position,
                args=args,
            )

            gen_kwargs = dict(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                max_new_tokens=args['max_token_generate'],
                do_sample=False,
            )
            if getattr(tokenizer, "pad_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

            with torch.no_grad():
                with add_hooks(fwd_pre_hooks_gen, fwd_hooks_gen, use_kwargs=args.get('scale_attention_heads', 0)):
                    generated = model.generate(**gen_kwargs)

            prompt_len = tokenized_instructions.input_ids.shape[-1]
            generated_ids = generated[:, prompt_len:]
            if generated_ids.numel() > 0:
                trained_response = clean_decoded_text(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
            else:
                trained_response = ""
            trained_token_ids_tensor = generated_ids

            COUNT_ADD = 0
         
        except Exception as gen_exc:
            print(f"Failed to generate with trained coefficients: {gen_exc}")
            trained_response = None
            generated_ids = None
            trained_response_binary = None
            trained_response_non_negative = None
            gen_bin_ids = None
            gen_topk_ids = None
            gen_nonneg_ids = None
            trained_response_top_k = {}
            trained_token_ids_tensor = None
            coefficients_top_k = {}

        results.append({
            'input_prompt': input_prompt[0],
            'input_tokens': input_tokens,
            'target': target_text,
            'loss': best_loss,
            'coefficients': best_coeff.tolist(),
            'positions': intervene_position,
            'trained_response': trained_response,
            #'trained_token_ids': trained_token_ids_tensor[0].tolist() if trained_token_ids_tensor is not None and trained_token_ids_tensor.numel() > 0 else [],
            #'trained_response_top_k': trained_response_top_k,
            #'trained_token_ids_top_k': gen_topk_ids[0].tolist() if gen_topk_ids is not None and gen_topk_ids.numel() > 0 else [],
            #'trained_response_binary_larger1': trained_response_binary,
            #'trained_response_non_negative': trained_response_non_negative,
            #'coefficients_top_k': coefficients_top_k,
            #'trained_token_ids_binary': gen_bin_ids[0].tolist() if gen_bin_ids is not None and gen_bin_ids.numel() > 0 else [],
        })

        if args.get('save_coeff_path'):
            torch.save(best_coeff, args['save_coeff_path'])

        COUNT_ADD = 0

    return results


def main():
    """
    Run the full text generation pipeline with intervention vectors.
    
    This function sets up the model, loads intervention vectors, and performs
    controlled text generation with various intervention strategies.
    """
    parser = argparse.ArgumentParser(description="Text generation with intervention vectors")
    
    # Model configuration
    parser.add_argument("--model", default='llama', type=str, help="Model type")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    # Data paths
    parser.add_argument("--test_data_pth", default='data/medcq.json', type=str, help='Test data path')
    parser.add_argument("--output_pth", default='data/medcq.json', type=str, help="Output file path")
    parser.add_argument("--intervention_vector", default=None, type=str, help='Intervention vector path')
    
    # Processing parameters
    parser.add_argument('--left', default=0, type=int, help='Left index')
    parser.add_argument('--right', default=10, type=int, help='Right index')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--mode', default='complete', type=str, help='Mode')
    
    # Intervention parameters
    parser.add_argument('--add_coef_intervene', default=1., type=float, help='Intervention coefficient')
    parser.add_argument('--layer_s', default=0, type=int, help='Start layer')
    parser.add_argument('--layer_e', default=32, type=int, help='End layer')
    parser.add_argument('--intervention_vector_layer', default=-1, type=int, help='Intervention vector at a layer')
    parser.add_argument('--reverse_intervention', default=0, type=int, help='Reverse intervention')
    parser.add_argument('--intervene_all', default=0, type=int, help='Intervene all tokens')
    parser.add_argument('--intervene_context_only', default=0, type=int, help='Intervene context only before the inversion prompt, used for reply inversion task only')
    
    # Generation parameters
    parser.add_argument('--max_token_generate', default=512, type=int, help='Max tokens to generate')
    parser.add_argument('--max_decode_step_while_intervene', default=1, type=int, help='Max decode steps while intervening')

    # Recording parameters
    parser.add_argument('--record_probs', default=0, type=int, help='Record probabilities')
 
    # Model-specific parameters
    parser.add_argument('--remove_inst_inp', default=0, type=int, help='Remove instruction input')
    parser.add_argument('--use_inversion', default=0, type=int, help='Use inversion reply task')
    parser.add_argument('--inversion_prompt_idx', default=0, type=int, help='Inversion prompt index')
    parser.add_argument('--mask_attn_token', default=0, type=int, help='Mask attention tokens')
    parser.add_argument('--model_path_override', default='', type=str, help='Optional local full-model path for HF loading')
    # Used parameters
    parser.add_argument('--use_jailbreak_test', default=0, type=int, help='Use the jailbreak version of inputs at test')
    parser.add_argument('--coeff_select', default=1, type=float, help='Coefficient select')
    parser.add_argument('--traverse_single_layer_intervention', default=0, type=int, help='Traverse single layer intervention')
    parser.add_argument('--load_ckpt', default=0, type=int, help='Load checkpoint')
    parser.add_argument('--peft_pth_ckpt', default='output/attn.json', type=str, help='PEFT checkpoint path')
    parser.add_argument('--positions', default='-1', type=str, help='Positions')
    parser.add_argument('--intervene_on_perturbed_step', default=0, type=int, help='Intervene on perturbed step')
    parser.add_argument('--step_key', default='perturb_step', type=str, help='Key to use for accessing step data (e.g., initial_step, perturb_step, consistent_step)')
    parser.add_argument('--include_final_answer_cue', default=1, type=int, help='Whether to append final answer cue to prompt')
    parser.add_argument('--context_key', default='context_text', type=str, help='Key to use for accessing context data')
    parser.add_argument('--intervene_on_jb_prompt', default=0, type=int, help='Intervene on jailbreak prompt')
    parser.add_argument('--omit_thinking', default=0, type=int, help='Omit thinking')
    parser.add_argument('--scale_attention_heads', default=0, type=int, help='Scale attention heads intervention')
    parser.add_argument('--attention_scale_factor', default=0.0, type=float, help='Scale factor for attention heads')
    parser.add_argument('--intervention_mode', default='add', type=str, help="Intervention mode: 'add' or 'replace_prefix'")
    parser.add_argument('--target_key', default='response', type=str, help='Key to fetch target output sequence during training')
    parser.add_argument('--train_steps', default=100, type=int, help='Number of optimization steps for coefficients')
    parser.add_argument('--coeff_lr', default=0.05, type=float, help='Learning rate for coefficient optimization')
    parser.add_argument('--coeff_init_std', default=0.02, type=float, help='Stddev for initializing coefficients')
    parser.add_argument('--save_coeff_path', default=None, type=str, help='Optional path to save optimized coefficients')
    parser.add_argument('--verbose_train', default=0, type=int, help='Verbose logging flag for training mode')
    parser.add_argument('--coeff_init_value', default=0.5, type=float, help='Initial value for coefficients')
    parser.add_argument('--cap_coeff', default=0, type=int, help='Cap coefficients to 0-1')
    parser.add_argument('--coeff_top_k', default=0, type=int, help='Keep top-k coefficients for post-training generation')
    parser.add_argument('--load_coeff', default=0, type=int, help='Load coefficients')
    parser.add_argument('--load_coeff_pth', default=None, type=str, help='Path to load coefficients')
    parser.add_argument('--coeff_rand_count', default=0, type=int, help='Number of random coefficients to use')
    parser.add_argument('--intervene_on_user_header', default=0, type=int, help='Intervene on user header')
    parser.add_argument('--replace_qkv_prefix', default=0, type=int, help='Replace QKV for prefix tokens')
    parser.add_argument('--qkv_modify_loaded_random', default=0, type=int, help='Debug mode: randomize loaded QKV tensors in memory (all-layer mode)')
    parser.add_argument('--qkv_modify_seed', default=-1, type=int, help='Seed for randomizing loaded QKV (-1 uses non-deterministic RNG)')
    parser.add_argument('--qkv_modify_std', default=1.0, type=float, help='Stddev for randomizing loaded QKV')
    parser.add_argument('--qkv_modify_mean', default=0.0, type=float, help='Mean for randomizing loaded QKV')
    parser.add_argument('--qkv_pth', default=None, type=str, help='Path to saved QKV tensors')
    parser.add_argument('--qkv_layer', default=-1, type=int, help='Override QKV layer index (-1 uses saved)')
    parser.add_argument('--replace_qkv_prefix_all_layers', default=0, type=int, help='Replace QKV for prefix tokens on all layers in one run')
    parser.add_argument('--qkv_dir', default=None, type=str, help='Directory containing per-layer QKV files: layer_<idx>.pt')
    parser.add_argument('--qkv_file_template', default='layer_{layer}.pt', type=str, help='Filename template under qkv_dir; supports {layer}')
    parser.add_argument('--skip_activation_intervention', default=0, type=int, help='Skip activation vector intervention hooks')
    parser.add_argument('--intervene_all_layers', default=0, type=int, help='Intervene all layers once in complete mode')
    parser.add_argument('--replace_kv_postfix', default=0, type=int, help='Replace K/V (no Q) for postfix tokens (after the question, e.g. assistant header)')
    parser.add_argument('--replace_kv_postfix_all_layers', default=0, type=int, help='Replace K/V for postfix tokens on all layers in one run')
    parser.add_argument('--kv_postfix_pth', default=None, type=str, help='Path to saved postfix KV tensors (single layer)')
    parser.add_argument('--kv_postfix_layer', default=-1, type=int, help='Override postfix KV layer index (-1 uses saved)')
    parser.add_argument('--kv_postfix_dir', default=None, type=str, help='Directory containing per-layer postfix KV files: layer_<idx>.pt')
    parser.add_argument('--kv_postfix_file_template', default='layer_{layer}.pt', type=str, help='Filename template under kv_postfix_dir; supports {layer}')
    parser.add_argument('--deterministic_inference', default=1, type=int, help='Enable deterministic PyTorch/CUDA settings for repeatable patch inference')
    parser.add_argument('--enforce_eager_attention', default=1, type=int, help='Force eager HF attention kernels for repeatable patch inference')
    parser.add_argument('--strip_full_system_block', default=0, type=int, help='If 1, strip the entire default system block from the chat-template prefix to match a model trained with strip_full_system_block=true')
    parser.add_argument('--load_in_4bit', default=0, type=int, help='Load HF base model in 4-bit quantization to fit large models on limited GPU memory')
    parser.add_argument('--custom_system_prompt', default='', type=str, help='If non-empty, override the default system prompt with this text and force use_system_prompt=True.')
    parser.add_argument('--enable_thinking', default=-1, type=int, help='Qwen3 template kwarg: -1=default, 0=skip thinking, 1=allow thinking.')
    parser.add_argument('--include_think_block_in_postfix', default=0, type=int, help='If 1, postfix KV replacement includes Qwen3 no-think empty <think> block tokens.')
    parser.add_argument('--kv_postfix_positions_in_postfix', default='', type=str, help='Optional space-separated relative indices within the selected postfix region to patch.')
    args = parser.parse_args()
    params = vars(args)
    if params.get('replace_qkv_prefix_all_layers', 0):
        params['replace_qkv_prefix'] = 1
    if params.get('replace_kv_postfix_all_layers', 0):
        params['replace_kv_postfix'] = 1
    
    global MODEL, DECODING_STEP, STRIP_FULL_SYSTEM_BLOCK, CUSTOM_SYSTEM_PROMPT, ENABLE_THINKING, INCLUDE_THINK_BLOCK_IN_POSTFIX, KV_POSTFIX_POSITIONS_IN_POSTFIX
    DECODING_STEP = params['max_decode_step_while_intervene'] #if -1, since the input token, till end of generation
    MODEL = params['model']
    STRIP_FULL_SYSTEM_BLOCK = bool(params.get('strip_full_system_block', 0))
    CUSTOM_SYSTEM_PROMPT = (params.get('custom_system_prompt') or None) or None
    et = int(params.get('enable_thinking', -1))
    ENABLE_THINKING = None if et == -1 else bool(et)
    INCLUDE_THINK_BLOCK_IN_POSTFIX = bool(params.get('include_think_block_in_postfix', 0))
    if params.get('kv_postfix_positions_in_postfix'):
        KV_POSTFIX_POSITIONS_IN_POSTFIX = list(map(int, params['kv_postfix_positions_in_postfix'].split()))
    else:
        KV_POSTFIX_POSITIONS_IN_POSTFIX = None
    if params.get('deterministic_inference', 0):
        configure_deterministic_inference(int(params['seed']))
    
    # Parse list arguments
    params['positions'] = list(map(int, params['positions'].split()))

    if params['mode'] == 'train':
        if not params['target_key']:
            params['target_key'] = 'response'



    # Model loading with error handling (utils.load_model_and_tokenizer — HF only).
    try:
        model, tokenizer = load_model_and_tokenizer(
            MODEL,
            params['model_size'],
            model_path_override=params.get('model_path_override') or None,
            enforce_eager=bool(params.get('enforce_eager_attention', 0)),
            load_in_4bit=bool(params.get('load_in_4bit', 0)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if params['load_ckpt']:
        print('loading ckpt adapter')
        try:
            if not params['peft_pth_ckpt']:
                raise ValueError("--peft_pth_ckpt must be provided when --load_ckpt is set")
            model = PeftModel.from_pretrained(model, params['peft_pth_ckpt'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            exit()
            return
        print(model)

    if params['mode'] == 'complete':
        try:
            test_data = read_row(params['test_data_pth'])[params['left']:params['right']]
        except Exception as e:
            print(f"Error loading test data: {e}")
            return

        if params.get('replace_qkv_prefix'):
            if params.get('replace_qkv_prefix_all_layers', 0):
                if not params.get('qkv_dir'):
                    raise ValueError("--qkv_dir must be provided when --replace_qkv_prefix_all_layers is set.")
                n_layers = model.config.num_hidden_layers
                qkv_data_by_layer: Dict[int, dict] = {}
                try:
                    for layer_idx in range(n_layers):
                        filename = params['qkv_file_template'].format(layer=layer_idx)
                        qkv_path = os.path.join(params['qkv_dir'], filename)
                        if not os.path.exists(qkv_path):
                            raise FileNotFoundError(f"Missing QKV file for layer {layer_idx}: {qkv_path}")
                        qkv_data_by_layer[layer_idx] = torch.load(qkv_path, weights_only=False)
                    if params.get('qkv_modify_loaded_random', 0):
                        seed = int(params.get('qkv_modify_seed', -1))
                        mean = float(params.get('qkv_modify_mean', 0.0))
                        std = float(params.get('qkv_modify_std', 1.0))
                        for layer_idx in range(n_layers):
                            _randomize_loaded_qkv_data_in_place(
                                qkv_data=qkv_data_by_layer[layer_idx],
                                layer_idx=layer_idx,
                                seed=seed,
                                mean=mean,
                                std=std,
                            )
                        print(
                            f"Loaded and randomized all-layer QKV data from {params['qkv_dir']} "
                            f"({n_layers} layers, seed={seed}, mean={mean}, std={std})"
                        )
                    else:
                        print(f"Loaded all-layer QKV data from {params['qkv_dir']} ({n_layers} layers)")
                    params['qkv_data_by_layer'] = qkv_data_by_layer
                except Exception as e:
                    print(f"Error loading all-layer QKV data: {e}")
                    return
            else:
                if params.get('qkv_modify_loaded_random', 0):
                    raise ValueError("--qkv_modify_loaded_random currently supports all-layer mode only. Use --replace_qkv_prefix_all_layers 1.")
                if not params.get('qkv_pth'):
                    raise ValueError("--qkv_pth must be provided when --replace_qkv_prefix is set.")
                try:
                    params['qkv_data'] = torch.load(params['qkv_pth'], weights_only=False)
                except Exception as e:
                    print(f"Error loading QKV data: {e}")
                    return

        if params.get('replace_kv_postfix'):
            if params.get('replace_kv_postfix_all_layers', 0):
                if not params.get('kv_postfix_dir'):
                    raise ValueError("--kv_postfix_dir must be provided when --replace_kv_postfix_all_layers is set.")
                n_layers = model.config.num_hidden_layers
                kv_postfix_data_by_layer: Dict[int, dict] = {}
                try:
                    for layer_idx in range(n_layers):
                        filename = params['kv_postfix_file_template'].format(layer=layer_idx)
                        kv_path = os.path.join(params['kv_postfix_dir'], filename)
                        if not os.path.exists(kv_path):
                            raise FileNotFoundError(f"Missing postfix KV file for layer {layer_idx}: {kv_path}")
                        kv_postfix_data_by_layer[layer_idx] = torch.load(kv_path, weights_only=False)
                    print(f"Loaded all-layer postfix KV data from {params['kv_postfix_dir']} ({n_layers} layers)")
                    params['kv_postfix_data_by_layer'] = kv_postfix_data_by_layer
                except Exception as e:
                    print(f"Error loading all-layer postfix KV data: {e}")
                    return
            else:
                if not params.get('kv_postfix_pth'):
                    raise ValueError("--kv_postfix_pth must be provided when --replace_kv_postfix is set.")
                try:
                    params['kv_postfix_data'] = torch.load(params['kv_postfix_pth'], weights_only=False)
                except Exception as e:
                    print(f"Error loading postfix KV data: {e}")
                    return

        if params.get('skip_activation_intervention'):
            intervention_vector = None
        else:
            try:
                intervention_vector = torch.load(params['intervention_vector'], weights_only=False)
            except Exception as e:
                print(f"Error loading intervention vector: {e}")
                return

        candidate_coeff = [params['coeff_select']] #if params['coeff_select'] else [1] 
        
        for coeff in candidate_coeff:
            params['add_coef_intervene'] = coeff
            print('coefficient for intervention', coeff)
            
            #layer_list = list(range(13, 32)) if params['traverse_single_layer_intervention'] else [0]
            
            
            layer_indices = list(range(params['layer_s'], params['layer_e']))
            if params.get('skip_activation_intervention'):
                layer_indices = [params['layer_s']]
            for i in layer_indices:
                print('##################')
                print('intervention layer', i)                        
                use_persuade = params['use_jailbreak_test']

                if intervention_vector is not None:
                    if isinstance(intervention_vector, np.ndarray):
                        intervention_vector = torch.tensor(intervention_vector)
                    if len(intervention_vector.shape) == 2:
                        intervention_vector = intervention_vector.unsqueeze(0)
                            
                    print('intervention vector shape', intervention_vector.shape)

                    if params['traverse_single_layer_intervention']:
                        pass
                        intervention_vector_local = intervention_vector[:, out_i, :].to(model.device)
                    else:
                        intervention_vector_local = intervention_vector.to(model.device)

                    if params['reverse_intervention']:
                        intervention_vector_local = -intervention_vector_local
                else:
                    intervention_vector_local = None


                def tokenize_instructions_fn(instructions, include_final_answer_cue=params['include_final_answer_cue'], intervene_on_perturbed_step=params['intervene_on_perturbed_step']):
                    final_answer_cue = '\n The final answer is \\boxed{' if include_final_answer_cue else ''
                    step_key = params['step_key']
                    context_key = params['context_key'] #by default, context_text
                    if intervene_on_perturbed_step:
                        inps = [
                            _strip_leading_bos(
                                formatInp(i, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
                                tokenizer,
                            ) + '. ' + i[context_key] + '. ' + i[step_key] + final_answer_cue
                            for i in instructions
                        ]
                    else:  
                        inps = [
                            _strip_leading_bos(
                                formatInp(i, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
                                tokenizer,
                            )
                            for i in instructions
                        ]
                    if instructions and random.random() < 0.01:
                        print("Sample formatted input:", inps[0])
                    if params['omit_thinking']:
                        inps=[ inp + '\n\n</think>' for inp in inps]
                    return inps,tokenizer(inps, padding=True, return_tensors="pt")

                
                if params['intervene_all_layers']:
                    intervene_layers = list(range(params['layer_s'], params['layer_e']))
                    completions = complete_with_intervention(
                    model, tokenizer, test_data, tokenize_instructions_fn,
                    intervene_layers, batch_size=params['batch_size'], 
                    intervention_vector_ori=intervention_vector_local,
                    args=params,
                )
                else:
                    completions = complete_with_intervention(
                        model, tokenizer, test_data, tokenize_instructions_fn,
                        [i], batch_size=params['batch_size'], 
                        intervention_vector_ori=intervention_vector_local,
                        args=params,
                    )
                
                # Save results
                output_file = params['output_pth'].replace('.json', f'-intervene{i}.json')
                with open(output_file, 'w') as f:
                    for completion in completions:
                        json.dump(completion, f)
                        f.write("\n")
                if params['intervene_all_layers']:
                    break



    elif params['mode'] == 'train':
        try:
            train_data = read_row(params['test_data_pth'])[params['left']:params['right']]
        except Exception as e:
            print(f"Error loading training data: {e}")
            return

        try:
            intervention_vector = torch.load(params['intervention_vector'], weights_only=False)
        except Exception as e:
            print(f"Error loading intervention vector: {e}")
            return

        if isinstance(intervention_vector, np.ndarray):
            intervention_vector = torch.tensor(intervention_vector)
        if len(intervention_vector.shape) == 2:
            intervention_vector = intervention_vector.unsqueeze(0)

        if params['reverse_intervention']:
            intervention_vector = -intervention_vector

        def tokenize_instructions_fn(instructions):
            if 'prompt' in instructions[0]:
                try:
                    instructions=[{'question': ent['prompt']['prompt']} for ent in instructions]
                except:
                    instructions=[{'question': ent['prompt']['question']} for ent in instructions]
            else:
                instructions=[{'question': ent['question']} for ent in instructions]
            inps = [
                _strip_leading_bos(
                    formatInp(ent, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
                    tokenizer,
                )
                for ent in instructions
            ]
            if instructions and random.random() < 0.01:
                print("Sample formatted input:", inps[0])
            if params['omit_thinking']:
                inps=[ inp + '\n\n</think>' for inp in inps]
            return inps,tokenizer(inps, padding=True, return_tensors="pt")

        intervene_layers = list(range(params['layer_s'], params['layer_s']+1))

        training_results = train_intervention_coefficients(
            model=model,
            tokenizer=tokenizer,
            instructions=train_data,
            tokenize_instructions_fn=tokenize_instructions_fn,
            intervene_layers=intervene_layers,
            batch_size=params['batch_size'],
            intervention_vector_ori=intervention_vector,
            args=params,
        )


        output_file = params['output_pth'].replace('.json', f"-train-layer{params['layer_s']}.json")
        with open(output_file, 'w') as f:
            for result in training_results:
                json.dump(result, f)
                f.write("\n")
        print(f"Saved training results to {output_file}")

    else:
        raise ValueError(f"Unsupported mode: {params['mode']}")


if __name__ == "__main__":
    main()
