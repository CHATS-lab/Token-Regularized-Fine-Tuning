import json
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional
from torch import Tensor
from tqdm import tqdm
# HF causal LM for hidden-state extraction (not Unsloth).
from utils import read_row, formatInp, load_model_and_tokenizer, extract_primary_question
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import functools
import random
import logging
import numpy as np
from peft import PeftModel

logger = logging.getLogger(__name__)


MODEL = ''
NUM_TOKEN_HIDDEN = 0  # by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens
STRIP_FULL_SYSTEM_BLOCK = False
CUSTOM_SYSTEM_PROMPT = None
ENABLE_THINKING = None  # None=default, False=skip thinking, True=enable

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


def _compute_prefix_positions_for_instruction(d: dict, tokenizer) -> List[int]:
    question = extract_primary_question(d)
    if question is None:
        raise ValueError("Could not extract question content for prefix token extraction.")
    if not str(question).strip():
        question = "placeholder question"
        print("Empty question; locating prefix via marker.")

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
        raise ValueError("Could not locate question marker for prefix token extraction.")
    print('ids for prefix:', marked_ids)
    prefix_len = start_idx
    if prefix_len <= 0:
        raise ValueError("Prefix length is zero; cannot extract prefix token positions.")
    return list(range(prefix_len))


def _compute_postfix_positions_for_instruction(d: dict, tokenizer) -> List[int]:
    """Return token positions *after* the user question in the formatted prompt.

    These are the "postfix" tokens – e.g. the assistant header / turn
    separator that appears between the question and the model's response.
    """
    question = extract_primary_question(d)
    if question is None:
        raise ValueError("Could not extract question content for postfix token extraction.")
    if not str(question).strip():
        question = "placeholder question"

    marked_content = f"{MARKER_START}{question}{MARKER_END}"
    marked_prompt = _format_with_overridden_content(d, marked_content, tokenizer)
    marked_ids = tokenizer(marked_prompt).input_ids
    if isinstance(marked_ids, torch.Tensor):
        marked_ids = marked_ids.tolist()

    marker_end_ids = tokenizer(MARKER_END, add_special_tokens=False).input_ids
    end_marker_start = _find_subsequence(marked_ids, marker_end_ids)
    if end_marker_start is None:
        raise ValueError("Could not locate MARKER_END in tokenized prompt for postfix extraction.")

    postfix_start_in_marked = end_marker_start + len(marker_end_ids)
    postfix_len = len(marked_ids) - postfix_start_in_marked
    if postfix_len <= 0:
        raise ValueError("Postfix length is zero; no tokens after the question.")

    original_prompt = _strip_leading_bos(
        formatInp(d, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
        tokenizer,
    )
    original_ids = tokenizer(original_prompt).input_ids
    if isinstance(original_ids, torch.Tensor):
        original_ids = original_ids.tolist()

    original_len = len(original_ids)
    return list(range(original_len - postfix_len, original_len))


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


def _get_num_layers(model: AutoModelForCausalLM, load_ckpt: int) -> int:
    if load_ckpt:
        if hasattr(model, "model") and hasattr(model.model, "model"):
            return len(model.model.model.layers)
        return len(model.model.layers)
    layers = model.model.layers if hasattr(model, "model") else model.layers
    return len(layers)


def _capture_linear_output_hook(cache_list: List[Tensor], positions: List[int]) -> Callable:
    def hook_fn(module: torch.nn.Module, input, output):
        activation = output
        if activation.dim() == 2:
            activation = activation.unsqueeze(0)
        seq_len = activation.size(1)
        valid_positions = [p for p in positions if 0 <= p < seq_len]
        if not valid_positions:
            return output
        cache_list.append(activation[:, valid_positions, :].detach().cpu())
        return output
    return hook_fn


def extract_qkv_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: dict,
    layer_idx: int,
    load_ckpt: int,
    positions_in_prefix: Optional[List[int]] = None,
) -> dict:
    prompt = _strip_leading_bos(
        formatInp(example, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
        tokenizer,
    )
    tokenized = tokenizer(prompt, return_tensors="pt")
    prefix_positions = _compute_prefix_positions_for_instruction(example, tokenizer)

    if positions_in_prefix:
        positions = []
        for rel_idx in positions_in_prefix:
            if rel_idx < 0 or rel_idx >= len(prefix_positions):
                raise ValueError(f"Prefix position index out of range: {rel_idx}")
            positions.append(prefix_positions[rel_idx])
    else:
        positions = prefix_positions

    seq_len = tokenized.input_ids.shape[-1]
    if not positions or max(positions) >= seq_len:
        raise ValueError("Requested positions fall outside the prompt sequence.")

    attn_module = _get_attention_module(model, layer_idx, load_ckpt)
    if not hasattr(attn_module, "q_proj") or not hasattr(attn_module, "k_proj") or not hasattr(attn_module, "v_proj"):
        raise AttributeError("Attention module does not expose q_proj/k_proj/v_proj.")

    cache = {"q": [], "k": [], "v": []}
    fwd_hooks = [
        (attn_module.q_proj, _capture_linear_output_hook(cache["q"], positions)),
        (attn_module.k_proj, _capture_linear_output_hook(cache["k"], positions)),
        (attn_module.v_proj, _capture_linear_output_hook(cache["v"], positions)),
    ]

    with torch.no_grad():
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=tokenized.input_ids.to(model.device),
                attention_mask=tokenized.attention_mask.to(model.device),
                use_cache=False,
            )

    if not cache["q"] or not cache["k"] or not cache["v"]:
        raise RuntimeError("Failed to capture QKV outputs.")

    q = cache["q"][0]
    k = cache["k"][0]
    v = cache["v"][0]
    if q.dim() == 3:
        q = q.squeeze(0)
    if k.dim() == 3:
        k = k.squeeze(0)
    if v.dim() == 3:
        v = v.squeeze(0)

    return {
        "layer": layer_idx,
        "positions": positions,
        "prefix_len": len(prefix_positions),
        "q": q,
        "k": k,
        "v": v,
    }


def extract_qkv_postfix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: dict,
    layer_idx: int,
    load_ckpt: int,
    positions_in_postfix: Optional[List[int]] = None,
) -> dict:
    prompt = _strip_leading_bos(
        formatInp(example, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
        tokenizer,
    )
    tokenized = tokenizer(prompt, return_tensors="pt")
    postfix_positions = _compute_postfix_positions_for_instruction(example, tokenizer)

    if positions_in_postfix:
        positions = []
        for rel_idx in positions_in_postfix:
            if rel_idx < 0 or rel_idx >= len(postfix_positions):
                raise ValueError(f"Postfix position index out of range: {rel_idx}")
            positions.append(postfix_positions[rel_idx])
    else:
        positions = postfix_positions

    seq_len = tokenized.input_ids.shape[-1]
    if not positions or max(positions) >= seq_len:
        raise ValueError("Requested postfix positions fall outside the prompt sequence.")

    attn_module = _get_attention_module(model, layer_idx, load_ckpt)
    if not hasattr(attn_module, "q_proj") or not hasattr(attn_module, "k_proj") or not hasattr(attn_module, "v_proj"):
        raise AttributeError("Attention module does not expose q_proj/k_proj/v_proj.")

    cache = {"q": [], "k": [], "v": []}
    fwd_hooks = [
        (attn_module.q_proj, _capture_linear_output_hook(cache["q"], positions)),
        (attn_module.k_proj, _capture_linear_output_hook(cache["k"], positions)),
        (attn_module.v_proj, _capture_linear_output_hook(cache["v"], positions)),
    ]

    with torch.no_grad():
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=tokenized.input_ids.to(model.device),
                attention_mask=tokenized.attention_mask.to(model.device),
                use_cache=False,
            )

    if not cache["q"] or not cache["k"] or not cache["v"]:
        raise RuntimeError("Failed to capture QKV outputs for postfix.")

    q = cache["q"][0]
    k = cache["k"][0]
    v = cache["v"][0]
    if q.dim() == 3:
        q = q.squeeze(0)
    if k.dim() == 3:
        k = k.squeeze(0)
    if v.dim() == 3:
        v = v.squeeze(0)

    return {
        "layer": layer_idx,
        "positions": positions,
        "postfix_len": len(postfix_positions),
        "q": q,
        "k": k,
        "v": v,
    }


def get_chat_template_suffix_length(tokenizer, model_name: str) -> int:
    suffix_map = {
        'qwen': "<|im_end|>\n<|im_start|>assistant",
        'llama3': "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        'llama2': "[/INST]",
    }
    special_token = suffix_map.get(model_name)
    if special_token is None:
        return 0
    try:
        return len(tokenizer.tokenize(special_token))
    except Exception:
        return 0


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
) -> None:
    """
    Context manager for temporarily adding forward hooks to a model.

    Args:
        module_forward_pre_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
        module_forward_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
        **kwargs: Additional keyword arguments to pass to the hooks
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_mean_activations_pre_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN
) -> Callable:
    """
    Creates a hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...]) -> None:
        activation = input[0] if input else None
        if activation is None:
            print(f'Activation is None at layer {layer} (pre hook)')
            assert activation is not None, f'Activation is None at layer {layer} (pre hook)'
            exit()
        if activation.numel() == 0:
            print(f'Activation is empty at layer {layer} (pre hook), shape={tuple(activation.shape)}')
            assert activation.numel() > 0, f'Activation is empty at layer {layer} (pre hook)'
            exit()
        activation = activation.half()
        
        # Handle both 2D and 3D tensors
        if len(activation.shape) == 2:
            # For 2D tensors, assume (seq_len, hidden_dim) and add batch dimension
            activation = activation.unsqueeze(0)
            
        seq_len = activation.shape[1]
        
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                
                assert isinstance(positions[0], int)
                context = activation[:, positions[0]-step:positions[0], :]
                pos_activations = activation[:, positions, :]
                print('activation shape', activation.shape)
                print('pos_activations shape', pos_activations.shape)
                print('context shape', context.shape)
                merged_activation = torch.cat([context, pos_activations], dim=1)
                cache_full[layer].append(merged_activation.clone().detach().cpu())
            else:
                print('seq_len<positions', seq_len, len(positions))
                exit()
    return hook_fn

def get_mean_activations_fwd_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN
) -> Callable:
    """
    Creates a forward hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]) -> None:
        activation = output[0] if isinstance(output, (tuple, list)) else output
        if activation is None:
            print(f'Activation is None at layer {layer} (fwd hook)')
            assert activation is not None, f'Activation is None at layer {layer} (fwd hook)'
            exit()
        if activation.numel() == 0:
            print(f'Activation is empty at layer {layer} (fwd hook), shape={tuple(activation.shape)}')
            assert activation.numel() > 0, f'Activation is empty at layer {layer} (fwd hook)'
            exit()
        activation = activation.half()
        print('activation shape', activation.shape)
        
        # Handle both 2D and 3D tensors
        if len(activation.shape) == 2:
            # For 2D tensors, assume (seq_len, hidden_dim) and add batch dimension
            activation = activation.unsqueeze(0)
            
        seq_len = activation.shape[1]
        
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                context = activation[:, -len(positions)-step:-len(positions), :]
                pos_activations = activation[:, positions, :]
                merged_activation = torch.cat([context, pos_activations], dim=1)
                cache_full[layer].append(merged_activation.clone().detach().cpu())
            else:
                print('seq_len<positions', seq_len, len(positions))
                exit()
    return hook_fn


def get_mean_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    tokenize_instructions_fn: Callable,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    ret_whole_seq: bool = False,
    use_perturbed_step: bool = False,
    args: dict = None
) -> Tuple[Tensor, Tensor]:
    """
    Extracts mean activations from model for given instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        instructions: List of input instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (mean activations, full activations)
    """
    torch.cuda.empty_cache()
    model.eval()

    n_layers = model.config.num_hidden_layers
    print('n_layers:', n_layers)
    full_activations = [[] for _ in range(n_layers + 1)]
    print('length of full_activations:', len(full_activations))
    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size)):
            batch_instructions = instructions[i:i+batch_size]
            inputs = tokenize_instructions_fn(instructions=batch_instructions)


            # Determine positions for this batch
            positions_local = positions
            if use_perturbed_step:
                logger.info(f"Extracting positions from perturbed step")
                if batch_size != 1:
                    print("Warning: positions computed for first item only; batch_size>1 not fully supported")
                d0 = batch_instructions[0]
                step_key = (args or {}).get('step_tag', 'perturb_step')
                step_str = d0[step_key]
                # Tokenize step without forcing a leading space
                step_ids = tokenizer(step_str, add_special_tokens=False).input_ids
                full_ids = inputs.input_ids[0].tolist()
                attn = inputs.attention_mask[0].tolist()
                seq_len_eff = sum(attn)
                search_space = full_ids[:seq_len_eff]
                start_idx = None
                n = len(step_ids)
                # Find last occurrence of step_ids (as step is appended at the end)
                for s in range(seq_len_eff - n, -1, -1):
                    if search_space[s:s + n] == step_ids:
                        start_idx = s
                        break
                # Fallback: also try with a leading space tokenization if not found
                if start_idx is None:
                    step_ids_space = tokenizer(' ' + step_str, add_special_tokens=False).input_ids
                    n2 = len(step_ids_space)
                    for s in range(seq_len_eff - n2, -1, -1):
                        if search_space[s:s + n2] == step_ids_space:
                            start_idx = s
                            n = n2
                            step_ids = step_ids_space
                            break
                assert start_idx is not None
                positions_local = list(range(start_idx, start_idx + n))
                print('extract positions (perturbed_step):', positions_local)
            elif args and args.get('extract_prompt_prefix_tokens', 0):
                if batch_size != 1:
                    print("Warning: prefix positions computed for first item only; batch_size>1 not fully supported")
                d0 = batch_instructions[0]
                question = extract_primary_question(d0)
                if question is None:
                    raise ValueError("Could not extract question content for prefix token extraction.")
                if not str(question).strip():
                    question = "placeholder question"
                    print("Empty question; locating prefix via marker.")

                marked_content = f"{MARKER_START}{question}{MARKER_END}"
                marked_prompt = _format_with_overridden_content(d0, marked_content, tokenizer)
                marked_ids = tokenizer(marked_prompt).input_ids
                
                if isinstance(marked_ids, torch.Tensor):
                    marked_ids = marked_ids.tolist()

                marker_start_ids = tokenizer(MARKER_START, add_special_tokens=False).input_ids
                start_idx = _find_subsequence(marked_ids, marker_start_ids)
                print('marked_prompt:', marked_prompt)
                print('marked_ids:', marked_ids)
                print('marker_start_ids:', marker_start_ids)

                if start_idx is None:
                    # Fallback: try to locate the question tokens directly
                    question_ids = tokenizer(question, add_special_tokens=False).input_ids
                    start_idx = _find_subsequence(marked_ids, question_ids)
                if start_idx is None:
                    raise ValueError("Could not locate question marker in tokenized prompt.")

                prefix_len = start_idx
                if prefix_len <= 0:
                    raise ValueError("Prefix length is zero; cannot extract prefix token activations.")
                positions_local = list(range(prefix_len))
                print(f"Extracting {prefix_len} prefix tokens before question.")

            elif args and args.get('extract_prompt_postfix_tokens', 0):
                if batch_size != 1:
                    print("Warning: postfix positions computed for first item only; batch_size>1 not fully supported")
                d0 = batch_instructions[0]
                positions_local = _compute_postfix_positions_for_instruction(d0, tokenizer)
                print(f"Extracting {len(positions_local)} postfix tokens after question.")

            elif args and args.get('extract_before_response_token', 0):
                attention_mask = inputs.attention_mask
                if attention_mask is not None and attention_mask.size(0) > 0:
                    seq_len_eff = int(attention_mask[0].sum().item())
                else:
                    seq_len_eff = inputs.input_ids.shape[-1]
                suffix_len = get_chat_template_suffix_length(tokenizer, MODEL)
                target_idx = seq_len_eff - suffix_len - 1 if suffix_len > 0 else seq_len_eff - 1
                target_idx = max(0, min(inputs.input_ids.shape[-1] - 1, target_idx))
                if args['extract_before_response_token_all_token_till_end'] == 1:
                    positions_local = list(range(target_idx, inputs.input_ids.shape[-1]))
                else:
                    positions_local = [target_idx]
                print(f"suffix_len: {suffix_len}, Extracting token before response header at position {target_idx}")
                
            print(f"Extracting positions: {positions_local}")
            
            # Create hooks with positions for this batch
            fwd_pre_hooks = [
                (block_modules[layer], get_mean_activations_pre_hook(
                    layer=layer,
                    cache_full=full_activations,
                    positions=positions_local,
                    whole_seq=ret_whole_seq
                )) for layer in range(n_layers)
            ]
            
            fwd_hooks = [
                (block_modules[n_layers-1], get_mean_activations_fwd_hook(
                    layer=n_layers,
                    cache_full=full_activations,
                    positions=positions_local,
                    whole_seq=ret_whole_seq
                ))
            ]
            
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                input_ids_cpu = inputs.input_ids.detach().cpu()
                input_tokens = [tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids_cpu]
                print('input ids:', input_ids_cpu.tolist())
                print('input tokens:', input_tokens)
                if inputs.attention_mask is not None:
                    print('attention mask:', inputs.attention_mask.detach().cpu().tolist())
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                    use_cache=False
                )

    try:
        # Fast path: all position lengths equal across examples
        flat_list = [torch.stack(inner_list) for inner_list in full_activations]
        result = torch.stack(flat_list).squeeze()
        if len(result.shape) < 3:
            result = result.unsqueeze(1)
        mean_activations = result.mean(dim=1)
    except Exception as e:
        # Fallback for variable-length positions: average over positions then over examples
        per_layer_bh = []  # list of [num_examples, hidden]
        for inner_list in full_activations:
            bh_list = []
            for t in inner_list:
                # Ensure 3D: [batch, pos, hidden]
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                # Average over positions
                bh_list.append(t.mean(dim=1))  # [batch, hidden]
            if len(bh_list) == 0:
                per_layer_bh.append(torch.empty(0, 0))
            else:
                per_layer_bh.append(torch.cat(bh_list, dim=0))  # [num_examples, hidden]

        # Stack layers along dim 0; assume same num_examples across layers
        
        result = torch.stack(per_layer_bh, dim=0)  # [layers, num_examples, hidden]

        # Mean over examples, ignoring NaNs if any padding occurred
        if torch.isnan(result).any():
            mean_activations = torch.nanmean(result, dim=1)
        else:
            mean_activations = result.mean(dim=1)

    print('mean shape', mean_activations.shape)
    return mean_activations, result



def get_mean_diff(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    tokenize_instructions_fn: Callable,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    extract_only: bool = False,
    use_persuade_harmful: bool = False,
    use_persuade_harmless: bool = False,
    use_sys_harmful: bool = False,
    ret_whole_seq: bool = False,
    use_perturbed_step: bool = False,
    args: dict | None = None
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Computes mean activation differences between harmful and harmless instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        extract_only: Whether to only extract harmful activations
        use_persuade_harmful: Whether to use persuasion for harmful
        use_persuade_harmless: Whether to use persuasion for harmless
        use_sys_harmful: Whether to use system prompt for harmful
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (harmful mean activations, harmless mean activations,
                harmful full activations, harmless full activations)
    """
    mean_activations_harmful, full_activations_harmful = get_mean_activations(
        model, tokenizer, harmful_instructions,
        functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmful, use_sys=use_sys_harmful),
        block_modules, batch_size=batch_size, positions=positions, ret_whole_seq=ret_whole_seq, use_perturbed_step=use_perturbed_step, args=args
    )
    
    torch.save(mean_activations_harmful, 'output/tmp_mean_activations_harmful.pt')
    torch.save(full_activations_harmful, 'output/tmp_full_activations_harmful.pt')
    del mean_activations_harmful, full_activations_harmful
    torch.cuda.empty_cache()

    if not extract_only:
        mean_activations_harmless, full_activations_harmless = get_mean_activations(
            model, tokenizer, harmless_instructions,
            functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmless),
            block_modules, batch_size=batch_size, positions=positions, ret_whole_seq=ret_whole_seq, use_perturbed_step=use_perturbed_step, args=args
        )
        mean_activations_harmful = torch.load('output/tmp_mean_activations_harmful.pt', weights_only=False)
        full_activations_harmful = torch.load('output/tmp_full_activations_harmful.pt', weights_only=False)

        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
    else:
        mean_activations_harmless = None
        full_activations_harmless = None

    return mean_activations_harmful, mean_activations_harmless, full_activations_harmful, full_activations_harmless

def generate_directions(
    model_base: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    args: dict
) -> Optional[Tensor]:
    """
    Generates direction vectors from model activations.

    Args:
        model_base: Base model
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        args: Arguments dictionary

    Returns:
        Mean difference tensor or None if computation fails
    """
    def tokenize_instructions_fn(instructions: List[str], use_persuade: bool = False, use_sys: bool = False) -> dict:
        #include the ori_output in the prompt
        inps = [_strip_leading_bos(
                             formatInp(i, model=MODEL, use_template=True, tokenizer=tokenizer, strip_full_system_block=STRIP_FULL_SYSTEM_BLOCK, custom_system_prompt=CUSTOM_SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING),
                             tokenizer,
                        )
                             for i in instructions
                        ]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(inps, padding=True, return_tensors="pt")
    print('model_base', model_base)
    try:
        model_block_modules = model_base.model.layers
    except Exception as e:
        model_block_modules = model_base.model.model.layers
    mean_activations_harmful, mean_activations_harmless, all_harmful, all_harmless = get_mean_diff(
        model_base, tokenizer, harmful_instructions, harmless_instructions,
        tokenize_instructions_fn, model_block_modules, args['batch_size'],
        args['positions'], args['extract_only'], args['use_persuade_harmful'],
        args['use_persuade_harmless'], args['use_sys_harmful'], args['ret_whole_seq'],
        args['extract_from_perturbed_step'], args
    )

    torch.save(all_harmful, args['output_pth_harmful'])
    torch.save(all_harmless, args['output_pth_harmless'])

    try:
        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
        mean_diffs = mean_activations_harmful - mean_activations_harmless
        assert not mean_diffs.isnan().any()
        #if args['mode_dir'] == 'hf':
            #mean_diffs = mean_diffs[:,NUM_TOKEN_HIDDEN-1] 
        #elif args['mode_dir'] == 'refuse':
            #mean_diffs = mean_diffs[:,-1]
        torch.save(mean_diffs.to('cpu'), args['output_pth'])
    except Exception as e:
        print(e)
        mean_diffs = None

    return mean_diffs

def main() -> None:
    """Run the full pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='llama', type=str, help="Model type")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size")
    parser.add_argument("--harmful_pth", default='data/medcq.json', type=str, help="Path to harmful examples")
    parser.add_argument("--harmless_pth", default='data/medcq.json', type=str, help="Path to harmless examples")
    parser.add_argument("--output_pth_harmful", default='output/mean_diff.pt', type=str, help="Output path for harmful activations")
    parser.add_argument("--output_pth_harmless", default='output/mean_diff_harmless.pt', type=str, help="Output path for harmless activations")
    parser.add_argument('--use_persuade_harmful', default=0, type=int, help='Use persuasion for harmful examples')
    parser.add_argument('--use_persuade_harmless', default=0, type=int, help='Use persuasion for harmless examples')
    parser.add_argument('--use_sys_harmful', default=0, type=int, help='Use system prompt for harmful examples')
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    parser.add_argument('--random_sample_harmful', default=0, type=int, help='Randomly sample harmful examples')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument("--output_pth", default='output/dir.pt', type=str, help="Output path of generated directions")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument('--mode', default='diff-mean', type=str, help='Mode')
    parser.add_argument('--positions', default='-1', type=str, help='Positions to extract')
    parser.add_argument('--extract_only', default=0, type=int, help='Only extract harmful activations')
    parser.add_argument('--ret_whole_seq', default=0, type=int, help='Return whole sequence')
    parser.add_argument('--extract_hidden_inst_token', default=0, type=int, help="Extract hidden state of instruction tokens")
    parser.add_argument('--extract_harmful_token_only', default=0, type=int, help="Extract harmful token only")
    parser.add_argument('--mode_dir', default='hf', type=str, help="Mode for direction extraction: 'hf' or 'refuse'")
    parser.add_argument('--model_path_override', default='', type=str, help='Optional local full-model path for HF loading')
    parser.add_argument('--use_template', default=1, type=int, help="Whether to use chat template formatting")
    parser.add_argument('--extract_from_perturbed_step', default=0, type=int, help='Extract hidden states only from perturbed_step tokens')
    parser.add_argument('--step_tag', default='perturb_step', type=str, help='Tag for perturbed step')
    parser.add_argument('--extract_prompt_prefix_tokens', default=0, type=int, help='Extract hidden states of all tokens before the input question in the prompt template')
    parser.add_argument('--extract_before_response_token', default=0, type=int, help='Extract hidden state at the token preceding the assistant response header')
    parser.add_argument('--extract_before_response_token_all_token_till_end', default=0, type=int, help='Extract hidden state of all tokens till the end of the sequence')
    parser.add_argument('--load_ckpt', default=0, type=int, help='Load checkpoint')
    parser.add_argument('--peft_pth_ckpt', default='output/attn.json', type=str, help='PEFT checkpoint path')
    parser.add_argument('--extract_qkv_prefix', default=0, type=int, help='Extract QKV for prefix tokens from a single example')
    parser.add_argument('--extract_qkv_prefix_all_layers', default=0, type=int, help='Extract QKV for all layers from a single example')
    parser.add_argument('--qkv_layer', default=0, type=int, help='Layer index for QKV extraction')
    parser.add_argument('--qkv_output_pth', default='output/qkv_prefix.pt', type=str, help='Output path for QKV tensors')
    parser.add_argument('--qkv_output_dir', default='output/qkv_prefix_all_layers', type=str, help='Output directory for all-layer QKV extraction')
    parser.add_argument('--qkv_positions_in_prefix', default='', type=str, help='Optional space-separated indices within prefix to extract')
    parser.add_argument('--extract_prompt_postfix_tokens', default=0, type=int, help='Extract hidden states of all tokens after the input question in the prompt template (e.g. assistant header)')
    parser.add_argument('--extract_qkv_postfix', default=0, type=int, help='Extract QKV for postfix tokens from a single example')
    parser.add_argument('--extract_qkv_postfix_all_layers', default=0, type=int, help='Extract QKV for postfix tokens across all layers')
    parser.add_argument('--qkv_positions_in_postfix', default='', type=str, help='Optional space-separated indices within postfix to extract')
    parser.add_argument('--deterministic_inference', default=1, type=int, help='Enable deterministic PyTorch/CUDA settings for repeatable extraction')
    parser.add_argument('--enforce_eager_attention', default=1, type=int, help='Force eager HF attention kernels for repeatable extraction')
    parser.add_argument('--strip_full_system_block', default=0, type=int, help='If 1, strip the entire default system block from the chat-template prefix to match a model trained with strip_full_system_block=true')
    parser.add_argument('--custom_system_prompt', default='', type=str, help='If non-empty, override the default system prompt with this text and force use_system_prompt=True.')
    parser.add_argument('--enable_thinking', default=-1, type=int, help='Qwen3 template kwarg: -1=default, 0=skip thinking (pre-insert empty <think></think>), 1=allow thinking.')

    args = parser.parse_args()
    params = vars(args)
    if params.get('extract_qkv_prefix_all_layers', 0):
        params['extract_qkv_prefix'] = 1
    if params.get('extract_qkv_postfix_all_layers', 0):
        params['extract_qkv_postfix'] = 1
    
    global MODEL
    global NUM_TOKEN_HIDDEN
    global STRIP_FULL_SYSTEM_BLOCK
    global CUSTOM_SYSTEM_PROMPT
    global ENABLE_THINKING

    params['positions'] = list(map(int, params['positions'].split()))
    MODEL = params['model']
    STRIP_FULL_SYSTEM_BLOCK = bool(params.get('strip_full_system_block', 0))
    CUSTOM_SYSTEM_PROMPT = (params.get('custom_system_prompt') or None) or None
    et = int(params.get('enable_thinking', -1))
    ENABLE_THINKING = None if et == -1 else bool(et)
    if params.get('deterministic_inference', 0):
        configure_deterministic_inference(int(params['seed']))
    model, tokenizer = load_model_and_tokenizer(
        params['model'],
        params['model_size'],
        model_path_override=params.get('model_path_override') or None,
        enforce_eager=bool(params.get('enforce_eager_attention', 0)),
    )  # HF path; see utils.py
    
    if params['load_ckpt']:
        print('loading ckpt adapter')
        try:
            if not params['peft_pth_ckpt']:
                raise ValueError("--peft_pth_ckpt must be provided when --load_ckpt is set")
            model = PeftModel.from_pretrained(model, params['peft_pth_ckpt'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
        print(model)

    model.eval()
    params['positions'] = [-1]

    harmful_train = read_row(params['harmful_pth'])

    if params['random_sample_harmful']:
        random.seed(params['left'] % len(harmful_train))
        harmful_train = random.sample(harmful_train, 1)
    else:
        if params['left'] < len(harmful_train):
            harmful_train = harmful_train[params['left']:params['right']]
        else:
            harmful_train = harmful_train[params['left'] % len(harmful_train):params['left'] % len(harmful_train)+1]
    
    if params.get('extract_qkv_prefix') or params.get('extract_qkv_prefix_all_layers'):
        if not harmful_train:
            raise ValueError("No examples available for QKV extraction.")
        positions_in_prefix = []
        if params.get('qkv_positions_in_prefix'):
            positions_in_prefix = list(map(int, params['qkv_positions_in_prefix'].split()))
        if params.get('extract_qkv_prefix_all_layers'):
            os.makedirs(params['qkv_output_dir'], exist_ok=True)
            num_layers = _get_num_layers(model, params['load_ckpt'])
            print(f"Extracting prefix QKV for all layers (0..{num_layers - 1})")
            for layer_idx in range(num_layers):
                qkv_data = extract_qkv_prefix(
                    model=model,
                    tokenizer=tokenizer,
                    example=harmful_train[0],
                    layer_idx=layer_idx,
                    load_ckpt=params['load_ckpt'],
                    positions_in_prefix=positions_in_prefix if positions_in_prefix else None,
                )
                layer_out_pth = os.path.join(params['qkv_output_dir'], f"layer_{layer_idx}.pt")
                torch.save(qkv_data, layer_out_pth)
                print(f"Saved layer {layer_idx} QKV to {layer_out_pth}")
        else:
            qkv_data = extract_qkv_prefix(
                model=model,
                tokenizer=tokenizer,
                example=harmful_train[0],
                layer_idx=params['qkv_layer'],
                load_ckpt=params['load_ckpt'],
                positions_in_prefix=positions_in_prefix if positions_in_prefix else None,
            )
            torch.save(qkv_data, params['qkv_output_pth'])
            print(f"Saved QKV to {params['qkv_output_pth']}")
        return

    if params.get('extract_qkv_postfix') or params.get('extract_qkv_postfix_all_layers'):
        if not harmful_train:
            raise ValueError("No examples available for postfix QKV extraction.")
        positions_in_postfix = []
        if params.get('qkv_positions_in_postfix'):
            positions_in_postfix = list(map(int, params['qkv_positions_in_postfix'].split()))
        if params.get('extract_qkv_postfix_all_layers'):
            os.makedirs(params['qkv_output_dir'], exist_ok=True)
            num_layers = _get_num_layers(model, params['load_ckpt'])
            print(f"Extracting postfix QKV for all layers (0..{num_layers - 1})")
            for layer_idx in range(num_layers):
                qkv_data = extract_qkv_postfix(
                    model=model,
                    tokenizer=tokenizer,
                    example=harmful_train[0],
                    layer_idx=layer_idx,
                    load_ckpt=params['load_ckpt'],
                    positions_in_postfix=positions_in_postfix if positions_in_postfix else None,
                )
                layer_out_pth = os.path.join(params['qkv_output_dir'], f"layer_{layer_idx}.pt")
                torch.save(qkv_data, layer_out_pth)
                print(f"Saved layer {layer_idx} postfix QKV to {layer_out_pth}")
        else:
            qkv_data = extract_qkv_postfix(
                model=model,
                tokenizer=tokenizer,
                example=harmful_train[0],
                layer_idx=params['qkv_layer'],
                load_ckpt=params['load_ckpt'],
                positions_in_postfix=positions_in_postfix if positions_in_postfix else None,
            )
            torch.save(qkv_data, params['qkv_output_pth'])
            print(f"Saved postfix QKV to {params['qkv_output_pth']}")
        return

    harmless_train = read_row(params['harmless_pth'])[params['left']:params['right']]

    with open(params['output_pth'].replace('.pt', '_prompts_used.json'), 'w') as f:
        json.dump({'harmful': harmful_train, 'harmless': harmless_train}, f, indent=4)

    candidate_directions = generate_directions(model, tokenizer, harmful_train, harmless_train, params)
    print(candidate_directions.shape)

if __name__ == "__main__":
    main()
