import os
import sys

# vLLM -> diskcache -> sqlite3 can load ICU from the Python prefix; that ICU needs a
# newer libstdc++ (e.g. CXXABI_1.3.15) than some GPU base images ship in /usr/lib.
# Prepend env lib dirs before any extension modules load.
if sys.platform == "linux":
    _lib_dirs = []
    for _root in (os.environ.get("VIRTUAL_ENV"), os.environ.get("CONDA_PREFIX")):
        if _root:
            _d = os.path.join(_root, "lib")
            if os.path.isdir(_d):
                _lib_dirs.append(_d)
    for _d in ("/venv/main/lib", "/opt/conda/lib"):
        if os.path.isdir(_d) and _d not in _lib_dirs:
            _lib_dirs.append(_d)
    if _lib_dirs:
        _tail = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            _lib_dirs + ([_tail] if _tail else [])
        )

import torch
import argparse
import math
import json
import logging
import time
import random
from typing import Dict, List, Tuple, Any, Optional, Set

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from tqdm import tqdm
# Loader map: vLLM engine vs HF+Peft in-process (see utils.py; training uses Unsloth, not this file).
from utils import (
    load_vllm_model_and_tokenizer,
    get_model_path,
    load_model_and_tokenizer,
    load_and_subtract_lora_adapters,
)
from utils import read_row, formatInp, setup_workspace_directories, is_runpod_environment, get_model_cache_kwargs
from utils import load_successful_rephrase_exemplars, extract_primary_question, append_gpt_assistantfinal
try:
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover
    LoRARequest = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
else:
    raise RuntimeError("CUDA is required for this script")

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Target device: {DEVICE}")


def maybe_append_assistantfinal(prompt: str, args: Dict[str, Any]) -> str:
    if not args.get("append_assistant_final", 1):
        return prompt
    return append_gpt_assistantfinal(prompt, args['model'])


def load_exemplar_pairs(exemplar_file: str) -> List[Dict]:
    """
    Load exemplar question-rephrase pairs from JSON file.
    
    Args:
        exemplar_file: Path to JSON file with exemplar pairs
        
    Returns:
        List of exemplar pair dictionaries
    """
    if not exemplar_file or not os.path.exists(exemplar_file):
        return []
    
    with open(exemplar_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    logger.info(f"Loaded {len(pairs)} exemplar pairs from {exemplar_file}")
    return pairs


def load_exemplar_pairs_from_args(args: Dict[str, Any]) -> List[Dict]:
    """
    Load exemplar pairs based on CLI arguments.
    """
    if not args.get('use_exemplars'):
        return []
    exemplar_mode = args.get('exemplar_mode', 'json')
    if exemplar_mode == 'lower_score_csv':
        initial_csv = args.get('exemplar_initial_csv', '')
        rephrased_csv = args.get('exemplar_rephrased_csv', '')
        score_field = args.get('exemplar_score_field', 'gpt4o_evaluation')
        min_delta = float(args.get('exemplar_min_delta', 0.0))
        max_pairs = args.get('exemplar_max_pairs', None)
        selection = args.get('exemplar_selection', 'top')
        seed = args.get('exemplar_seed', None)
        direction = args.get('exemplar_direction', 'higher')
        pairs = load_successful_rephrase_exemplars(
            initial_csv=initial_csv,
            rephrased_csv=rephrased_csv,
            score_field=score_field,
            direction=direction,
            selection=selection,
            seed=seed,
        )
        logger.info(
            "Loaded %d lower-score exemplar pairs from CSVs (initial=%s rephrased=%s)",
            len(pairs),
            initial_csv,
            rephrased_csv,
        )
        print('#'*100)
        print('Loaded exemplars:', len(pairs))
        print('#'*100)
        
        return pairs
    exemplar_file = args.get('exemplar_file', '')
    return load_exemplar_pairs(exemplar_file)


def select_random_exemplars(
    all_pairs: List[Dict],
    n: int,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Randomly select N exemplar pairs.
    
    Args:
        all_pairs: List of all available exemplar pairs
        n: Number of pairs to select
        seed: Random seed for reproducibility (optional)
        
    Returns:
        List of selected exemplar pairs
    """
    if not all_pairs:
        return []
    
    if n >= len(all_pairs):
        return all_pairs
    
    if seed is not None:
        random.seed(seed)
    
    return random.sample(all_pairs, n)


def select_exemplars(
    all_pairs: List[Dict],
    n: int,
    seed: Optional[int] = None,
    selection: str = "random",
) -> List[Dict]:
    """
    Select exemplar pairs with different strategies.
    """
    if selection == "top":
        return all_pairs[:n] if n < len(all_pairs) else all_pairs
    return select_random_exemplars(all_pairs, n, seed)


def format_exemplars_as_context(exemplars: List[Dict], format_template: str = None) -> str:
    """
    Format exemplar pairs as context string for in-context learning.
    
    Args:
        exemplars: List of exemplar pair dictionaries
        format_template: Template for formatting (default: "question: {q} rephrased: {r}")
        
    Returns:
        Formatted context string
    """
    if not exemplars:
        return ""
    
    if format_template is None:
        format_template = "Initial question: {initial_question} Rephrased: {rephrased_question}"
    assert len(exemplars) > 1, "At least 2 exemplars are required"
    context_lines = []
    for exemplar in exemplars:
        try:
            formatted = format_template.format(**exemplar)
            context_lines.append(formatted)
        except KeyError:
            # Fallback if template doesn't match keys
            context_lines.append(
                f"Initial question: {exemplar.get('initial_question', '')} Rephrased: {exemplar.get('rephrased_question', '')}"
            )
    #context_lines.append("Rephrase the following question based on the examples above.")
    return "\n\n".join(context_lines)


def append_rephrase_output(
    output_path: str,
    data_point: Dict[str, Any],
    rephrased_text: str,
    selected_exemplars: Optional[List[Dict]] = None,
):
    """
    Append a rephrased question row to a JSONL output file.
    """
    if not output_path:
        return
    if not rephrased_text:
        return
    row = {
        "id": data_point.get("id"),
        "initial_question": extract_primary_question(data_point),
        "question": rephrased_text.strip(),
    }
    if selected_exemplars:
        row["exemplar_ids"] = [ex.get("id") for ex in selected_exemplars]
        row["exemplar_score_deltas"] = [ex.get("score_delta") for ex in selected_exemplars]
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def _strip_leading_bos(text: str, tokenizer) -> str:
    """Strip a leading BOS token that apply_chat_template(tokenize=False) embeds as text.

    When the resulting string is re-tokenized the tokenizer also prepends a BOS
    (add_bos_token=True), producing a double BOS.  Stripping it here keeps
    exactly one BOS after downstream tokenization.
    """
    bos = getattr(tokenizer, "bos_token", None)
    if bos and isinstance(text, str) and text.startswith(bos):
        return text[len(bos):]
    return text


def _normalize_chat_content(content: Any) -> str:
    if isinstance(content, dict) and 'parts' in content:
        return ''.join(str(part) for part in content['parts'])
    if isinstance(content, dict) and 'text' in content:
        return str(content['text'])
    if isinstance(content, list):
        return ''.join(_normalize_chat_content(part) for part in content)
    return "" if content is None else str(content)


def _build_messages_for_formatinp(
    data_point: Dict[str, Any],
    args: Dict[str, Any],
    exemplar_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    custom_system_prompt = args.get('custom_system_prompt', '') or None
    use_system_prompt = bool(args.get('use_sys_prompt', 0) or custom_system_prompt)
    sys_prompt = custom_system_prompt if custom_system_prompt else "You are a helpful and honest assistant."

    if 'messages' in data_point:
        content = None
        for msg in data_point['messages']:
            if msg.get('role') == 'user':
                content = _normalize_chat_content(msg.get('content'))
            if msg.get('role') == 'system' and not custom_system_prompt:
                sys_prompt = _normalize_chat_content(msg.get('content'))
        if content is None:
            raise ValueError("Could not find user content in data dictionary")
    else:
        content_keys = ['prompt', 'question', 'problem', 'input', 'instruction', 'text', 'content']
        content = None
        for key in content_keys:
            if key in data_point:
                content = data_point[key]
                break
        if content is None:
            for _, value in data_point.items():
                if isinstance(value, str):
                    content = value
                    break
        if content is None:
            raise ValueError("Could not find content in data dictionary")

    if exemplar_context:
        content = exemplar_context + "\n\n" + content

    messages = [{"role": "user", "content": content}]
    if use_system_prompt:
        messages.insert(0, {"role": "system", "content": sys_prompt})
    return messages


def _find_token_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        raise ValueError("needle must not be empty")
    for idx in range(len(haystack) - len(needle) + 1):
        if haystack[idx:idx + len(needle)] == needle:
            return idx
    raise ValueError("Marker token sequence not found in templated prompt.")


def _compute_no_prefix_token_ids(
    rendered_prompt: str,
    messages: List[Dict[str, str]],
    tokenizer,
) -> Tuple[List[int], int]:
    marker = "<<|PREFIX_SPLIT_MARKER|>>"
    marked_messages = json.loads(json.dumps(messages))
    for idx, msg in enumerate(marked_messages):
        if msg.get("role") == "user":
            marked_messages[idx]["content"] = marker + str(msg.get("content", ""))
            break
    else:
        raise ValueError("No user message found when computing no-prefix token IDs.")

    marked_prompt = tokenizer.apply_chat_template(
        marked_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    backend_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    marked_ids = backend_tokenizer.encode(marked_prompt, add_special_tokens=False)
    marker_ids = backend_tokenizer.encode(marker, add_special_tokens=False)
    prefix_token_count = _find_token_subsequence(marked_ids, marker_ids)

    full_ids = backend_tokenizer.encode(rendered_prompt, add_special_tokens=False)
    trimmed_ids = full_ids[prefix_token_count:]
    if not trimmed_ids:
        raise ValueError("No tokens remain after removing the template prefix at inference time.")
    bos_token_id = getattr(backend_tokenizer, "bos_token_id", None)
    if bos_token_id is not None and trimmed_ids[0] == bos_token_id:
        raise ValueError("BOS token remained after prefix removal at inference time.")
    return trimmed_ids, prefix_token_count


def _effective_prompt_to_text(prompt_payload: Any, tokenizer) -> str:
    if isinstance(prompt_payload, str):
        return prompt_payload
    if isinstance(prompt_payload, torch.Tensor):
        prompt_payload = prompt_payload.tolist()
    if isinstance(prompt_payload, (list, tuple)):
        backend_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        return backend_tokenizer.decode(list(prompt_payload), skip_special_tokens=False)
    return str(prompt_payload)


def create_sampling_params(args: Dict[str, Any]) -> SamplingParams:
    """
    Create vLLM sampling parameters from arguments.
    
    Args:
        args: Configuration arguments
        
    Returns:
        vLLM SamplingParams object
    """
    # We loop explicitly for num_tries, so generate one completion per call.
    n = 1
    if args['do_sample_decode'] or args.get('num_tries', 1) > 1:
        return SamplingParams(
            temperature=args['temperature'],
            top_p=args['top_p'],
            max_tokens=args['max_len'], 
            n=n
        )
    else:
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args['max_len'],
            n=n
        )


def extract_model_output(
    full_output: str,
    model_type: str,
    input_prompt: str = ""
) -> str:
    """
    Extract the model's response from the full generated text.
    
    Args:
        full_output: Complete generated text
        model_type: Type of model used
        input_prompt: Original input prompt (for llama3)
        
    Returns:
        Extracted model response
    """

    return full_output.strip()


def resolve_question_span_start(
    text: str,
    question: str,
    initial_prompt_fallback: Optional[str] = None,
) -> int:
    """Character offset where ``question`` starts in ``text``, for region-limited replacement.

    When ``text`` is a corrupted ``modified_prompt``, ``text.find(question)`` often fails.
    If the only edits were inside the question span, the offset matches ``initial_prompt``."""
    idx = text.find(question)
    if idx != -1:
        return idx
    if initial_prompt_fallback:
        return initial_prompt_fallback.find(question)
    return -1


def _has_stored_modified_prompts(data_point: Dict[str, Any]) -> bool:
    if (data_point.get("modified_prompt") or "").strip():
        return True
    mps = data_point.get("modified_prompts") or []
    return any((x or "").strip() for x in mps)


def random_replace_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    num_positions: int,
    use_nearest: bool = False,
    embedding_matrix: Optional[torch.Tensor] = None,
    nearest_top_k: int = 5,
    restrict_substring: Optional[str] = None,
    restrict_to_token_ids: Optional[Set[int]] = None,
    restrict_before_substring: bool = False,
    restrict_after_substring: bool = False,
    question_span_start: Optional[int] = None,
    exclude_token_ids: Optional[Set[int]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace num_positions tokens in text, optionally using nearest-embedding tokens.

    If ``question_span_start`` is set (>= 0), use it instead of ``text.find(restrict_substring)``
    for locating the question; required when ``text`` no longer contains the exact question string.
    """
    offsets = None
    token_ids = None
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = enc.get("input_ids")
        offsets = enc.get("offset_mapping")
    except Exception:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        offsets = None
    if token_ids is None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids or num_positions <= 0:
        return text, []
    num_positions = min(num_positions, len(token_ids))
    positions = list(range(len(token_ids)))
    assert offsets is not None, "Offsets are None"
    assert restrict_substring, "Restrict substring is None"
    if restrict_substring and offsets:
        print('restrict_substring:', restrict_substring)
        if question_span_start is not None and question_span_start >= 0:
            span_start = question_span_start
        else:
            span_start = text.find(restrict_substring)
        if span_start == -1:
            logger.warning("Question span not found in text; skipping restricted token replacement.")
            return text, []
        print('span_start:', span_start)
        span_end = span_start + len(restrict_substring)
        if restrict_before_substring:
            positions = [
                idx for idx, (start, end) in enumerate(offsets)
                if end <= span_start
            ]
        elif restrict_after_substring:
            positions = [
                idx for idx, (start, end) in enumerate(offsets)
                if start >= span_end
            ]
        else:
            positions = [
                idx for idx, (start, end) in enumerate(offsets)
                if start < span_end and end > span_start
            ]
        if not positions:
            return text, []
    if restrict_to_token_ids:
        positions = [idx for idx in positions if token_ids[idx] in restrict_to_token_ids]
    if exclude_token_ids:
        positions = [idx for idx in positions if token_ids[idx] not in exclude_token_ids]
    if not positions:
        return text, []
    positions = random.sample(positions, min(num_positions, len(positions)))
    vocab_size = tokenizer.vocab_size
    use_nearest = use_nearest and embedding_matrix is not None
    emb_norm = None
    if use_nearest:
        with torch.no_grad():
            emb = embedding_matrix.detach().float().cpu()
            emb_norm = emb / torch.clamp(emb.norm(dim=1, keepdim=True), min=1e-12)
    replacements: List[Tuple[str, str]] = []
    for pos in positions:
        orig_id = token_ids[pos]
        if use_nearest and emb_norm is not None:
            target = emb_norm[orig_id]
            sim = torch.matmul(emb_norm, target)
            sim[orig_id] = -1e9  # avoid replacing with itself
            top_k = max(1, min(int(nearest_top_k), sim.numel()))
            top_ids = torch.topk(sim, top_k).indices.tolist()
            new_id = int(random.choice(top_ids))
        else:
            new_id = random.randint(0, vocab_size - 1)
        replacements.append(
            (
                tokenizer.convert_ids_to_tokens(orig_id),
                tokenizer.convert_ids_to_tokens(new_id),
            )
        )
        token_ids[pos] = new_id
    return tokenizer.decode(token_ids, skip_special_tokens=False), replacements


def replace_first_token_random(
    text: str,
    tokenizer: AutoTokenizer,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace the first token in text with a uniformly random token."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return text, []
    vocab_size = tokenizer.vocab_size
    orig_id = token_ids[0]
    new_id = random.randint(0, vocab_size - 1)
    replacements = [(tokenizer.convert_ids_to_tokens(orig_id), tokenizer.convert_ids_to_tokens(new_id))]
    token_ids[0] = new_id
    return tokenizer.decode(token_ids, skip_special_tokens=False), replacements


def random_insert_tokens(text: str, tokenizer: AutoTokenizer, tokens_to_insert: str, num_positions: int) -> str:
    """Insert the provided tokens at random positions in the text."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    insert_ids = tokenizer.encode(tokens_to_insert, add_special_tokens=False) if tokens_to_insert.strip() else []
    if not token_ids or not insert_ids or num_positions <= 0:
        return text
    num_positions = min(num_positions, len(token_ids) + 1)
    positions = sorted(random.sample(range(len(token_ids) + 1), num_positions))
    offset = 0
    for pos in positions:
        idx = pos + offset
        token_ids[idx:idx] = insert_ids
        offset += len(insert_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def resolve_common_token_path(path: str, fallback_dir: str = "output") -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.join(fallback_dir, path)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Missing common token JSON file: {path}")


def load_common_token_ids(
    path: str, top_k: int, tokenizer: Optional[AutoTokenizer] = None
) -> Set[int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    items: Optional[List[Dict[str, Any]]] = None
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        items = None

    if items is not None:
        if int(top_k) <= 0:
            top_k = len(items)
        token_ids = []
        for item in items[: max(0, int(top_k))]:
            token_id = item.get("token_id")
            if token_id is None:
                continue
            token_ids.append(int(token_id))
        return set(token_ids)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines and "token" in lines[0].lower() and "count" in lines[0].lower():
        lines = lines[1:]
    token_ids: List[int] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if not parts:
            continue
        if len(parts) == 1:
            parts = [part for part in line.split() if part.strip()]
        if not parts:
            continue
        token_field = parts[0]
        token_id: Optional[int] = None
        if token_field.lstrip("-").isdigit():
            token_id = int(token_field)
        elif tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(token_field)
            if token_id is None:
                token_id = tokenizer.unk_token_id
            if hasattr(tokenizer, "unk_token_id"):
                if token_id == tokenizer.unk_token_id and token_field != getattr(tokenizer, "unk_token", None):
                    token_id = None
        if token_id is None:
            continue
        token_ids.append(int(token_id))
    if int(top_k) <= 0:
        top_k = len(token_ids)
    return set(token_ids[: max(0, int(top_k))])


def apply_common_token_attention_mask(
    prompt: str,
    question: str,
    tokenizer: AutoTokenizer,
    common_ids: Set[int],
    max_length: int = 2048,
) -> Tuple[Dict[str, torch.Tensor], int]:
    if not prompt or not question:
        return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length), 0
    span_start = prompt.find(question)
    if span_start == -1:
        return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length), 0
    span_end = span_start + len(question)
    try:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length), 0
    if "offset_mapping" not in enc:
        return {k: v for k, v in enc.items() if k != "offset_mapping"}, 0
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"].clone()
    offsets = enc["offset_mapping"][0].tolist()
    masked = 0
    for i, (start, end) in enumerate(offsets):
        if start < span_end and end > span_start:
            if int(input_ids[0, i]) in common_ids:
                attention_mask[0, i] = 0
                masked += 1
    enc = {k: v for k, v in enc.items() if k != "offset_mapping"}
    enc["attention_mask"] = attention_mask
    return enc, masked


def infer_vllm(
    vllm_model: LLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any],
    lora_request: Optional[Any] = None,
) -> None:
    """
    Run inference on evaluation data using vLLM.
    
    Args:
        vllm_model: The vLLM model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    if args.get('rephrase_output_file') and os.path.exists(args['rephrase_output_file']):
        os.remove(args['rephrase_output_file'])
    
    # Load exemplar pairs if provided
    exemplar_pairs = load_exemplar_pairs_from_args(args)
    
    # Create sampling parameters (one completion per call; loop over num_tries)
    sampling_params = create_sampling_params(args)
    num_tries = args.get('num_tries', 1)
    common_ids = args.get('common_token_ids')
    restrict_replace_to_common = bool(args.get('random_replace_common_only', 0))
    
    _pts = (args.get('protected_token_strings') or '').strip()
    protected_ids = None
    if _pts:
        protected_ids = set()
        for s in [t.strip() for t in _pts.split(',') if t.strip()]:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
                for i in ids: protected_ids.add(int(i))
            except Exception: pass
        logger.info(f'protected token ids: {protected_ids}')
    use_nearest = bool(args.get('nearest_token_replace', 0))
    remove_template_prefix_tokens = bool(args.get('remove_template_prefix_tokens', 0))
    if remove_template_prefix_tokens and args.get('use_input_modified_prompt'):
        raise ValueError("remove_template_prefix_tokens is not compatible with use_input_modified_prompt.")
    if use_nearest:
        logger.info("Nearest token replace requested, but vLLM path has no embedding access; falling back to random replacements.")
    else:
        logger.info("Nearest token replace disabled; using random replacements when replacement is enabled.")
    logger.info('Starting vLLM inference...')
    start_time = time.time()
    
    # Process data in batches for better efficiency
    batch_size = args.get('batch_size', 1)
    total_batches = (len(eval_data) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(eval_data))
        batch_data = eval_data[start_idx:end_idx]
        
        # Prepare base prompts (without per-try randomization)
        base_prompts = []
        base_prompt_messages = []
        base_selected_exemplars = []
        for data_point in batch_data:
            selected_exemplars = []
            exemplar_context = ""
            if args.get('use_exemplars') and exemplar_pairs:
                num_exemplars = args.get('num_exemplars', 3)
                exemplar_seed = args.get('exemplar_seed', None)
                selection = args.get('exemplar_selection', 'random')
                selected_exemplars = select_exemplars(exemplar_pairs, num_exemplars, exemplar_seed, selection)
                exemplar_context = format_exemplars_as_context(
                    selected_exemplars,
                    args.get('exemplar_format')
                )
            use_input_mp = bool(args.get('use_input_modified_prompt'))
            if use_input_mp and _has_stored_modified_prompts(data_point):
                prompt_base = None
                prompt_messages = None
            else:
                prompt_messages = _build_messages_for_formatinp(
                    data_point,
                    args,
                    exemplar_context=exemplar_context if exemplar_context else None,
                )
                prompt_base = formatInp(
                    data_point,
                    model=args['model'],
                    use_template=args['use_template'],
                    tokenizer=tokenizer,
                    no_post_instruction=args['no_post_instruction'],
                    use_system_prompt=args['use_sys_prompt'],
                    exemplar_context=exemplar_context if exemplar_context else None,
                    custom_system_prompt=args.get('custom_system_prompt', '') or None,
                    strip_full_system_block=bool(args.get('strip_full_system_block', 0)),
                    enable_thinking=(None if args.get('enable_thinking', -1) == -1 else bool(args.get('enable_thinking'))),
                )
                prompt_base = maybe_append_assistantfinal(prompt_base, args)
            base_selected_exemplars.append(selected_exemplars)
            base_prompts.append(prompt_base)
            base_prompt_messages.append(prompt_messages)

        # Accumulators per data point
        batch_outputs = [[] for _ in batch_data]
        batch_responses = [[] for _ in batch_data]
        batch_modified_prompts = [[] for _ in batch_data]
        batch_templated_modified_prompts = [[] for _ in batch_data]
        batch_token_replacements = [[] for _ in batch_data]

        # Loop for num_tries with fresh random replacements each time
        for try_idx in range(num_tries):
            prompts = []
            for idx, prompt_base in enumerate(base_prompts):
                dp = batch_data[idx]
                prompt_messages = base_prompt_messages[idx]
                use_input_mp = bool(args.get('use_input_modified_prompt'))
                if use_input_mp:
                    mp_list = dp.get('modified_prompts') or []
                    if try_idx < len(mp_list) and (mp_list[try_idx] or '').strip():
                        prompt_try = mp_list[try_idx]
                    elif dp.get('modified_prompt'):
                        prompt_try = dp['modified_prompt']
                    else:
                        if prompt_base is None:
                            sel = base_selected_exemplars[idx]
                            ec = (
                                format_exemplars_as_context(sel, args.get('exemplar_format'))
                                if sel
                                else None
                            )
                            prompt_base = formatInp(
                                dp,
                                model=args['model'],
                                use_template=args['use_template'],
                                tokenizer=tokenizer,
                                no_post_instruction=args['no_post_instruction'],
                                use_system_prompt=args['use_sys_prompt'],
                                exemplar_context=ec,
                                custom_system_prompt=args.get('custom_system_prompt', '') or None,
                                strip_full_system_block=bool(args.get('strip_full_system_block', 0)),
                                enable_thinking=(None if args.get('enable_thinking', -1) == -1 else bool(args.get('enable_thinking'))),
                            )
                            prompt_base = maybe_append_assistantfinal(prompt_base, args)
                            base_prompts[idx] = prompt_base
                            base_prompt_messages[idx] = _build_messages_for_formatinp(
                                dp,
                                args,
                                exemplar_context=ec,
                            )
                            prompt_messages = base_prompt_messages[idx]
                        prompt_try = prompt_base
                else:
                    prompt_try = prompt_base
                replacements = []
                if args['random_token_insert']:
                    prompt_try = random_insert_tokens(prompt_try, tokenizer, args['insert_tokens'], args['random_insert_n'])
                q_text = extract_primary_question(dp)
                region_flags = (
                    args.get('replace_only_initial_question')
                    or args.get('replace_before_question')
                    or args.get('replace_after_question')
                )
                q_span = None
                if args['random_token_replace'] and region_flags:
                    has_stored_dp = _has_stored_modified_prompts(dp)
                    span_fb = (
                        dp.get('initial_prompt')
                        if (use_input_mp and has_stored_dp)
                        else None
                    )
                    s = resolve_question_span_start(prompt_try, q_text, span_fb)
                    q_span = s if s >= 0 else None
                if args['random_token_replace']:
                    prompt_try, replacements = random_replace_tokens(
                        prompt_try,
                        tokenizer,
                        args['random_replace_n'],
                        use_nearest=bool(args.get('nearest_token_replace', 0)),
                        embedding_matrix=None,  # vLLM path: embedding access not available
                        nearest_top_k=args.get('nearest_token_top_k', 5),
                        restrict_substring=q_text,
                        restrict_to_token_ids=common_ids if restrict_replace_to_common else None,
                        exclude_token_ids=protected_ids if protected_ids else None,
                        restrict_before_substring=bool(args.get('replace_before_question', 0)),
                        restrict_after_substring=bool(args.get('replace_after_question', 0)),
                        question_span_start=q_span,
                    )
                if args.get('replace_first_token'):
                    prompt_try, first_replacements = replace_first_token_random(prompt_try, tokenizer)
                    replacements = first_replacements + replacements
                if remove_template_prefix_tokens:
                    prompt_token_ids, _ = _compute_no_prefix_token_ids(prompt_try, prompt_messages, tokenizer)
                    effective_prompt_text = _effective_prompt_to_text(prompt_token_ids, tokenizer)
                    prompts.append(prompt_token_ids)
                else:
                    effective_prompt_text = _strip_leading_bos(prompt_try, tokenizer)
                    prompts.append(prompt_try)
                batch_modified_prompts[idx].append(effective_prompt_text)
                batch_templated_modified_prompts[idx].append(prompt_try)
                batch_token_replacements[idx].append(replacements)

            if not remove_template_prefix_tokens:
                prompts = [_strip_leading_bos(p, tokenizer) for p in prompts]
            if try_idx == 0 and prompts:
                ids_list = prompts[0] if remove_template_prefix_tokens else tokenizer.encode(prompts[0])
                logger.info("=== VLLM INPUT TOKEN IDS (first example) ===")
                logger.info("Token IDs (%d tokens): %s", len(ids_list), ids_list[:50])
                logger.info("First 10 tokens decoded: %s", [tokenizer.decode([t]) for t in ids_list[:10]])
                logger.info("Input text preview: %r", tokenizer.decode(ids_list[:200], skip_special_tokens=False))
                logger.info("BOS token id: %s, first token id: %s, match: %s",
                            tokenizer.bos_token_id, ids_list[0], ids_list[0] == tokenizer.bos_token_id)
                logger.info("============================================")
            outputs = vllm_model.generate(prompts, sampling_params, lora_request=lora_request)

            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                if args['use_jb']:
                    batch_responses[j].append(extract_model_output(generated_text, args['model']))
                else:
                    batch_outputs[j].append(generated_text.strip())

        # Save aggregated results (one line per example)
        with open(args['output_file_name'], 'a') as f:
            for j, data_point in enumerate(batch_data):
                data_idx = start_idx + j
                if data_idx >= len(eval_data):
                    break
                modified_prompt_first = batch_modified_prompts[j][0] if batch_modified_prompts[j] else ''
                if args['use_jb']:
                    eval_data[data_idx] = {
                        'prompt': eval_data[data_idx],
                        'response': batch_responses[j][0] if batch_responses[j] else '',
                        'responses': batch_responses[j],
                        'modified_prompt': modified_prompt_first,
                        'modified_prompts': batch_modified_prompts[j],
                        'templated_modified_prompt': batch_templated_modified_prompts[j][0] if batch_templated_modified_prompts[j] else '',
                        'templated_modified_prompts': batch_templated_modified_prompts[j],
                        'token_replacements': batch_token_replacements[j],
                    }
                else:
                    eval_data[data_idx]['ori_outputs'] = batch_outputs[j]
                    eval_data[data_idx]['ori_output'] = batch_outputs[j][0] if batch_outputs[j] else ''
                    eval_data[data_idx]['modified_prompt'] = modified_prompt_first
                    eval_data[data_idx]['modified_prompts'] = batch_modified_prompts[j]
                    eval_data[data_idx]['templated_modified_prompt'] = batch_templated_modified_prompts[j][0] if batch_templated_modified_prompts[j] else ''
                    eval_data[data_idx]['templated_modified_prompts'] = batch_templated_modified_prompts[j]
                    eval_data[data_idx]['token_replacements'] = batch_token_replacements[j]

                eval_data[data_idx]['probs'] = []
                json.dump(eval_data[data_idx], f)
                f.write('\n')

                if args.get('rephrase_output_file'):
                    rephrased_text = (
                        batch_responses[j][0] if args['use_jb'] and batch_responses[j] else
                        batch_outputs[j][0] if batch_outputs[j] else ''
                    )
                    append_rephrase_output(
                        args['rephrase_output_file'],
                        batch_data[j],
                        rephrased_text,
                        base_selected_exemplars[j] if j < len(base_selected_exemplars) else None,
                    )
    
    end_time = time.time()
    logger.info(f'vLLM inference completed in {end_time - start_time:.2f} seconds')


def infer_vllm_sequential(
    vllm_model: LLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any],
    lora_request: Optional[Any] = None,
) -> None:
    """
    Run inference on evaluation data using vLLM with sequential processing for compatibility.
    
    Args:
        vllm_model: The vLLM model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    if args.get('rephrase_output_file') and os.path.exists(args['rephrase_output_file']):
        os.remove(args['rephrase_output_file'])
    
    # Load exemplar pairs if provided
    exemplar_pairs = load_exemplar_pairs_from_args(args)
    
    # Create sampling parameters (one completion per call; loop over num_tries)
    sampling_params = create_sampling_params(args)
    num_tries = args.get('num_tries', 1)
    
    logger.info('Starting vLLM sequential inference...')
    start_time = time.time()
    
    use_nearest = bool(args.get('nearest_token_replace', 0))
    remove_template_prefix_tokens = bool(args.get('remove_template_prefix_tokens', 0))
    if remove_template_prefix_tokens and args.get('use_input_modified_prompt'):
        raise ValueError("remove_template_prefix_tokens is not compatible with use_input_modified_prompt.")
    common_ids = args.get('common_token_ids')
    restrict_replace_to_common = bool(args.get('random_replace_common_only', 0))
    _pts = (args.get('protected_token_strings') or '').strip()
    protected_ids = None
    if _pts:
        protected_ids = set()
        for s in [t.strip() for t in _pts.split(',') if t.strip()]:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
                for i in ids: protected_ids.add(int(i))
            except Exception: pass
        logger.info(f'protected token ids: {protected_ids}')
    if use_nearest:
        logger.info("Nearest token replace requested, but vLLM path has no embedding access; falling back to random replacements.")
    else:
        logger.info("Nearest token replace disabled; using random replacements when replacement is enabled.")
    for i in tqdm(range(len(eval_data)), desc="Processing samples"):
        data_point = eval_data[i]
        selected_exemplars = []
        exemplar_context = ""
        if args.get('use_exemplars') and exemplar_pairs:
            num_exemplars = args.get('num_exemplars', 3)
            exemplar_seed = args.get('exemplar_seed', None)
            selection = args.get('exemplar_selection', 'random')
            selected_exemplars = select_exemplars(exemplar_pairs, num_exemplars, exemplar_seed, selection)
            exemplar_context = format_exemplars_as_context(
                selected_exemplars,
                args.get('exemplar_format')
            )
        use_input_mp = bool(args.get('use_input_modified_prompt'))
        has_stored = _has_stored_modified_prompts(data_point)
        formatted_base: Optional[str] = None
        prompt_messages = None

        def _ensure_sequential_formatted_base() -> str:
            nonlocal formatted_base
            if formatted_base is not None:
                return formatted_base
            ip = formatInp(
                data_point,
                model=args['model'],
                use_template=args['use_template'],
                tokenizer=tokenizer,
                no_post_instruction=args['no_post_instruction'],
                use_system_prompt=args['use_sys_prompt'],
                exemplar_context=exemplar_context if exemplar_context else None,
                custom_system_prompt=args.get('custom_system_prompt', '') or None,
                strip_full_system_block=bool(args.get('strip_full_system_block', 0)),
                enable_thinking=(None if args.get('enable_thinking', -1) == -1 else bool(args.get('enable_thinking'))),
            )
            if args['add_reasoning_step']:
                ip = ip + " " + data_point['context_text']
            if args['infer_on_perturbed_step']:
                ip = ip + " " + data_point['perturb_context_text'] + '.\n' + data_point['initial_step']
            if args['omit_thinking']:
                ip = ip + "\n\n</redacted_thinking>"
            ip = ip + " " + args['custom_prompt']
            ip = maybe_append_assistantfinal(ip, args)
            formatted_base = ip
            return formatted_base

        if remove_template_prefix_tokens and not (use_input_mp and has_stored):
            prompt_messages = _build_messages_for_formatinp(
                data_point,
                args,
                exemplar_context=exemplar_context if exemplar_context else None,
            )

        if use_input_mp and has_stored:
            mp_dbg = data_point.get('modified_prompts') or []
            input_prompt = (
                (mp_dbg[0] if mp_dbg and (mp_dbg[0] or "").strip() else None)
                or (data_point.get("modified_prompt") or "").strip()
                or _ensure_sequential_formatted_base()
            )
        else:
            input_prompt = _ensure_sequential_formatted_base()

        # Accumulators
        outputs_all = []
        responses_all = []
        modified_prompts = []
        templated_modified_prompts = []
        replacements_all = []

        for try_idx in range(num_tries):
            if use_input_mp:
                mp_list = data_point.get('modified_prompts') or []
                if try_idx < len(mp_list) and (mp_list[try_idx] or '').strip():
                    prompt_try = mp_list[try_idx]
                elif data_point.get('modified_prompt'):
                    prompt_try = data_point['modified_prompt']
                else:
                    prompt_try = _ensure_sequential_formatted_base()
            else:
                prompt_try = _ensure_sequential_formatted_base()
            replacements = []
            if args['random_token_insert']:
                prompt_try = random_insert_tokens(prompt_try, tokenizer, args['insert_tokens'], args['random_insert_n'])
            q_text = extract_primary_question(data_point)
            region_flags = (
                args.get('replace_only_initial_question')
                or args.get('replace_before_question')
                or args.get('replace_after_question')
            )
            q_span = None
            if args['random_token_replace'] and region_flags:
                span_fb = (
                    data_point.get('initial_prompt')
                    if (use_input_mp and has_stored)
                    else None
                )
                s = resolve_question_span_start(prompt_try, q_text, span_fb)
                q_span = s if s >= 0 else None
            if args['random_token_replace']:
                prompt_try, replacements = random_replace_tokens(
                    prompt_try,
                    tokenizer,
                    args['random_replace_n'],
                    use_nearest=bool(args.get('nearest_token_replace', 0)),
                    embedding_matrix=None,  # vLLM path: embedding access not available
                    nearest_top_k=args.get('nearest_token_top_k', 5),
                    restrict_substring=q_text,
                    restrict_to_token_ids=common_ids if restrict_replace_to_common else None,
                    exclude_token_ids=protected_ids if protected_ids else None,
                    restrict_before_substring=bool(args.get('replace_before_question', 0)),
                    restrict_after_substring=bool(args.get('replace_after_question', 0)),
                    question_span_start=q_span,
                )
            if args.get('replace_first_token'):
                prompt_try, first_replacements = replace_first_token_random(prompt_try, tokenizer)
                replacements = first_replacements + replacements
            if remove_template_prefix_tokens:
                prompt_try_clean, _ = _compute_no_prefix_token_ids(prompt_try, prompt_messages, tokenizer)
            else:
                prompt_try_clean = _strip_leading_bos(prompt_try, tokenizer)
            modified_prompts.append(_effective_prompt_to_text(prompt_try_clean, tokenizer))
            templated_modified_prompts.append(prompt_try)
            replacements_all.append(replacements)
            if i == 0 and try_idx == 0:
                ids_list = prompt_try_clean if remove_template_prefix_tokens else tokenizer.encode(prompt_try_clean)
                logger.info("=== VLLM-SEQ INPUT TOKEN IDS (first example) ===")
                logger.info("Token IDs (%d tokens): %s", len(ids_list), ids_list[:50])
                logger.info("First 10 tokens decoded: %s", [tokenizer.decode([t]) for t in ids_list[:10]])
                logger.info("Input text preview: %r", tokenizer.decode(ids_list[:200], skip_special_tokens=False))
                logger.info("BOS token id: %s, first token id: %s, match: %s",
                            tokenizer.bos_token_id, ids_list[0], ids_list[0] == tokenizer.bos_token_id)
                logger.info("================================================")
            outputs = vllm_model.generate([prompt_try_clean], sampling_params, lora_request=lora_request)
            generated_text = outputs[0].outputs[0].text
            if args['use_jb']:
                responses_all.append(extract_model_output(generated_text, args['model']))
            else:
                outputs_all.append(generated_text.strip())

        with open(args['output_file_name'], 'a') as f:
            if args['use_jb']:
                response = responses_all[0] if responses_all else ''
                eval_data[i] = {
                    'prompt': eval_data[i],
                    'response': response,
                    'responses': responses_all,
                    'modified_prompt': modified_prompts[0] if modified_prompts else '',
                    'modified_prompts': modified_prompts,
                    'templated_modified_prompt': templated_modified_prompts[0] if templated_modified_prompts else '',
                    'templated_modified_prompts': templated_modified_prompts,
                    'token_replacements': replacements_all,
                }
            else:
                eval_data[i]['ori_outputs'] = outputs_all
                eval_data[i]['ori_output'] = outputs_all[0] if outputs_all else ''
                eval_data[i]['modified_prompt'] = modified_prompts[0] if modified_prompts else ''
                eval_data[i]['modified_prompts'] = modified_prompts
                eval_data[i]['templated_modified_prompt'] = templated_modified_prompts[0] if templated_modified_prompts else ''
                eval_data[i]['templated_modified_prompts'] = templated_modified_prompts
                eval_data[i]['token_replacements'] = replacements_all
            
            # Note: vLLM doesn't provide probability scores in the same way as transformers
            # We'll add a placeholder for compatibility
            eval_data[i]['probs'] = []
            json.dump(eval_data[i], f)
            f.write('\n')

            if args.get('rephrase_output_file'):
                rephrased_text = responses_all[0] if args['use_jb'] and responses_all else outputs_all[0] if outputs_all else ''
                append_rephrase_output(
                    args['rephrase_output_file'],
                    data_point,
                    rephrased_text,
                    selected_exemplars,
                )
        
        if outputs_all or responses_all:
            first_text = outputs_all[0] if outputs_all else responses_all[0]
            logger.debug(f'Output: {first_text[:100]}...')
    
    end_time = time.time()
    logger.info(f'vLLM sequential inference completed in {end_time - start_time:.2f} seconds')


def infer_lora_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any]
) -> None:
    """
    Run inference on evaluation data using LoRA model with transformers.
    
    Args:
        model: The LoRA model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    if args.get('rephrase_output_file') and os.path.exists(args['rephrase_output_file']):
        os.remove(args['rephrase_output_file'])
    
    # Load exemplar pairs if provided
    exemplar_pairs = load_exemplar_pairs_from_args(args)
    
    # Build explicit generation kwargs (avoid mixing GenerationConfig + kwargs).
    num_tries = args.get('num_tries', 1)
    do_sample = bool(args.get('do_sample_decode', 0)) or num_tries > 1
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": args['max_len'],
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = args['temperature']
        generate_kwargs["top_p"] = args['top_p']
    
    use_nearest = bool(args.get('nearest_token_replace', 0))
    remove_template_prefix_tokens = bool(args.get('remove_template_prefix_tokens', 0))
    if remove_template_prefix_tokens and args.get('use_input_modified_prompt'):
        raise ValueError("remove_template_prefix_tokens is not compatible with use_input_modified_prompt.")
    common_ids = args.get('common_token_ids')
    restrict_replace_to_common = bool(args.get('random_replace_common_only', 0))
    _pts = (args.get('protected_token_strings') or '').strip()
    protected_ids = None
    if _pts:
        protected_ids = set()
        for s in [t.strip() for t in _pts.split(',') if t.strip()]:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
                for i in ids: protected_ids.add(int(i))
            except Exception: pass
        logger.info(f'protected token ids: {protected_ids}')
    logger.info('Starting LoRA model inference...')
    start_time = time.time()
    
    results = []
    embedding_matrix = model.get_input_embeddings().weight.detach()
    if use_nearest:
        logger.info(f"Nearest token replace enabled; using embedding matrix shape={tuple(embedding_matrix.shape)} device={embedding_matrix.device}")
    else:
        logger.info("Nearest token replace disabled; replacements will be random when enabled.")
    
    for i in tqdm(range(len(eval_data)), desc="Processing samples"):
        data_point = eval_data[i]
        
        # Format the prompt
        selected_exemplars = []
        exemplar_context = ""
        if args.get('use_exemplars') and exemplar_pairs:
            num_exemplars = args.get('num_exemplars', 3)
            exemplar_seed = args.get('exemplar_seed', None)
            selection = args.get('exemplar_selection', 'random')
            selected_exemplars = select_exemplars(exemplar_pairs, num_exemplars, exemplar_seed, selection)
            exemplar_context = format_exemplars_as_context(
                selected_exemplars,
                args.get('exemplar_format')
            )
        else:
            exemplar_context = None

        use_input_mp = bool(args.get('use_input_modified_prompt'))
        has_stored = _has_stored_modified_prompts(data_point)
        formatted_base: Optional[str] = None
        prompt_messages = None

        def _ensure_formatted_base() -> str:
            nonlocal formatted_base
            if formatted_base is not None:
                return formatted_base
            formatted_base = formatInp(
                data_point,
                model=args['model'],
                use_template=args['use_template'],
                tokenizer=tokenizer,
                no_post_instruction=args['no_post_instruction'],
                use_system_prompt=args['use_sys_prompt'],
                exemplar_context=exemplar_context,
                custom_system_prompt=args.get('custom_system_prompt', '') or None,
                strip_full_system_block=bool(args.get('strip_full_system_block', 0)),
                enable_thinking=(None if args.get('enable_thinking', -1) == -1 else bool(args.get('enable_thinking'))),
            )
            if args.get('custom_prompt', ''):
                formatted_base = formatted_base + " " + args['custom_prompt']
            formatted_base = maybe_append_assistantfinal(formatted_base, args)
            return formatted_base

        if remove_template_prefix_tokens and not (use_input_mp and has_stored):
            prompt_messages = _build_messages_for_formatinp(
                data_point,
                args,
                exemplar_context=exemplar_context,
            )

        if use_input_mp and has_stored:
            mp_dbg = data_point.get('modified_prompts') or []
            input_prompt = (
                (mp_dbg[0] if mp_dbg and (mp_dbg[0] or "").strip() else None)
                or (data_point.get("modified_prompt") or "").strip()
                or _ensure_formatted_base()
            )
        else:
            input_prompt = _ensure_formatted_base()

        # Loop num_tries with fresh randomization each time
        outputs_all = []
        raw_outputs_all = []
        modified_prompts = []
        templated_modified_prompts = []
        replacements_all = []

        if i == 0:
            print("\n\n example base input_prompt:\n", input_prompt)

        print("\n\n base input_prompt:\n", input_prompt)
        for try_idx in range(num_tries):
            if use_input_mp:
                mp_list = data_point.get('modified_prompts') or []
                if try_idx < len(mp_list) and (mp_list[try_idx] or '').strip():
                    prompt_try = mp_list[try_idx]
                elif data_point.get('modified_prompt'):
                    prompt_try = data_point['modified_prompt']
                else:
                    prompt_try = _ensure_formatted_base()
            else:
                prompt_try = _ensure_formatted_base()
            replacements = []

            if args['random_token_insert']:
                prompt_try = random_insert_tokens(prompt_try, tokenizer, args['insert_tokens'], args['random_insert_n'])
            q_text = extract_primary_question(data_point)
            region_flags = (
                args.get('replace_only_initial_question')
                or args.get('replace_before_question')
                or args.get('replace_after_question')
            )
            q_span = None
            if args['random_token_replace'] and region_flags:
                span_fb = (
                    data_point.get('initial_prompt')
                    if (use_input_mp and has_stored)
                    else None
                )
                s = resolve_question_span_start(prompt_try, q_text, span_fb)
                q_span = s if s >= 0 else None
            if args['random_token_replace']:
                prompt_try, replacements = random_replace_tokens(
                    prompt_try,
                    tokenizer,
                    args['random_replace_n'],
                    use_nearest=bool(args.get('nearest_token_replace', 0)),
                    embedding_matrix=embedding_matrix,
                    nearest_top_k=args.get('nearest_token_top_k', 5),
                    restrict_substring=q_text,
                    restrict_to_token_ids=common_ids if restrict_replace_to_common else None,
                    exclude_token_ids=protected_ids if protected_ids else None,
                    restrict_before_substring=bool(args.get('replace_before_question', 0)),
                    restrict_after_substring=bool(args.get('replace_after_question', 0)),
                    question_span_start=q_span,
                )
            if args.get('replace_first_token'):
                prompt_try, first_replacements = replace_first_token_random(prompt_try, tokenizer)
                replacements = first_replacements + replacements
            prompt_try_clean = _strip_leading_bos(prompt_try, tokenizer)
            mask_count = 0
            if remove_template_prefix_tokens:
                prompt_token_ids, _ = _compute_no_prefix_token_ids(prompt_try, prompt_messages, tokenizer)
                inputs = {
                    "input_ids": torch.tensor([prompt_token_ids], dtype=torch.long, device=model.device),
                    "attention_mask": torch.ones((1, len(prompt_token_ids)), dtype=torch.long, device=model.device),
                }
            elif args.get('mask_common_tokens') and args.get('common_token_ids'):
                question_text = extract_primary_question(data_point)
                inputs, mask_count = apply_common_token_attention_mask(
                    prompt_try_clean,
                    question_text,
                    tokenizer,
                    args['common_token_ids'],
                    max_length=2048,
                )
            else:
                inputs = tokenizer(prompt_try_clean, return_tensors="pt", truncation=True, max_length=2048)
            modified_prompts.append(_effective_prompt_to_text(inputs["input_ids"][0], tokenizer))
            templated_modified_prompts.append(prompt_try)
            replacements_all.append(replacements)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            if i == 0 and try_idx == 0:
                ids_list = input_ids[0].tolist()
                logger.info("=== HF INPUT TOKEN IDS (first example) ===")
                logger.info("Token IDs (%d tokens): %s", len(ids_list), ids_list[:50])
                logger.info("First 10 tokens decoded: %s", [tokenizer.decode([t]) for t in ids_list[:10]])
                logger.info("Input text preview: %r", tokenizer.decode(ids_list[:200], skip_special_tokens=False))
                logger.info("BOS token id: %s, first token id: %s, match: %s",
                            tokenizer.bos_token_id, ids_list[0], ids_list[0] == tokenizer.bos_token_id)
                logger.info("==========================================")

            generate_call_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                **generate_kwargs,
            }
            # Unsloth fast-generate can mis-handle explicit all-ones masks.
            # Only pass attention_mask when we intentionally modified it.
            if args.get('mask_common_tokens'):
                generate_call_kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                outputs = model.generate(**generate_call_kwargs)

            generated_sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            if remove_template_prefix_tokens:
                prompt_len = int(input_ids.shape[1])
                decoded = [
                    tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    for seq in generated_sequences
                ]
            else:
                decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_sequences]
            for generated_text in decoded:
                raw_outputs_all.append(generated_text)
                if (not remove_template_prefix_tokens) and prompt_try in generated_text:
                    response_text = generated_text.replace(prompt_try, "").strip()
                else:
                    response_text = generated_text.strip()
                if '[/INST]' in response_text:
                    response_text = response_text.split('[/INST]')[-1].replace('</s>', '').strip()
                elif '<|im_start|>assistant' in response_text:
                    response_text = response_text.split('<|im_start|>assistant')[-1].replace('<|im_end|>', '').strip()
                elif 'assistant' in response_text:
                    response_text = response_text.split('assistant')[-1].replace('</s>', '').strip()
                elif 'ASSISTANT:' in response_text:
                    response_text = response_text.split('ASSISTANT:')[-1].replace('</s>', '').strip()
                outputs_all.append(response_text)

        question = extract_primary_question(data_point)
        result = {
            'id': data_point.get('id', i),
            'question': question,
            'output': outputs_all[0] if outputs_all else '',
            'outputs': outputs_all,
            'category': data_point.get('category', ''),
            'answer': data_point.get('answer', ''),
            'initial_prompt': data_point.get('initial_prompt') or input_prompt,
            'modified_prompt': modified_prompts[0] if modified_prompts else input_prompt,
            'modified_prompts': modified_prompts,
            'templated_modified_prompt': templated_modified_prompts[0] if templated_modified_prompts else input_prompt,
            'templated_modified_prompts': templated_modified_prompts,
            'token_replacements': replacements_all,
            'masked_common_token_count': mask_count
        }

        results.append(result)

        logger.debug(f'Generated response: {outputs_all[:1]}...')

        if args.get('rephrase_output_file'):
            rephrased_text = outputs_all[0] if outputs_all else ''
            append_rephrase_output(
                args['rephrase_output_file'],
                data_point,
                rephrased_text,
                selected_exemplars,
            )
    
    # Save results as JSON for eval_gpt
    with open(args['output_file_name'], 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    end_time = time.time()
    logger.info(f'LoRA model inference completed in {end_time - start_time:.2f} seconds')
    logger.info(f'Saved {len(results)} results to {args["output_file_name"]}')


def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description="Language model inference script with LoRA support")
    
    # Model configuration
    parser.add_argument("--model", default='llama', type=str, help="Model type (llama, llama3, qwen, olmo)")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size (7b, 8b, 13b, 14b, 32b)")
    parser.add_argument("--model_path_override", default='', type=str, help="Optional absolute/relative path to a local model directory for vLLM loading")
    parser.add_argument("--peft_pth_ckpt", default='', type=str, help="Path to PEFT checkpoint")
    parser.add_argument("--load_ckpt", default=0, type=int, help="Whether to load LoRA checkpoint")
    parser.add_argument("--use_lora", default=1, type=int, help="Use LoRA model instead of vLLM")
    parser.add_argument("--use_sys_prompt", default=0, type=int, help="Use system prompt")
    parser.add_argument("--custom_system_prompt", default="", type=str, help="Custom system prompt text; when non-empty, forces use_sys_prompt=1 and overrides the default system prompt")
    
    # vLLM specific configuration (only used if not using LoRA)
    parser.add_argument("--tensor_parallel_size", default=1, type=int, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float, help="GPU memory utilization ratio")
    parser.add_argument("--max_model_len", default=None, type=int, help="Maximum model length")
    parser.add_argument("--enforce_eager", default=0, type=int, help="Disable CUDA graph capture (saves GPU memory)")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for processing")
    parser.add_argument("--use_batching", default=1, type=int, help="Use batch processing for efficiency")
    
    # Data configuration
    parser.add_argument("--input", default='data/synthetic/datasets/auto_incorrect_subtle_split/auto_incorrect_subtle_test.jsonl', type=str, help="Input file path")
    parser.add_argument("--output_file_name", default='eval/auto_incorrect_subtle_responses.json', type=str, help="Output file path")
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=100, type=int, help='Right index for data slicing')
    
    # Generation configuration
    parser.add_argument("--temperature", default=0.7, type=float, help="Generation temperature")
    parser.add_argument("--top_p", default=0.9, type=float, help="Top-p sampling parameter")
    parser.add_argument("--do_sample_decode", default=1, type=int, help="Use sampling for decoding")
    parser.add_argument("--record_prob_max_pos", default=0, type=int, help="Max positions to record probabilities")
    parser.add_argument("--max_len", default=512, type=int, help="Maximum generation length")
    
    # Prompt configuration
    parser.add_argument("--use_jb", default=0, type=int, help="Use jailbroken prompt")
    parser.add_argument("--use_adv_suffix", default=0, type=int, help="Use adversarial suffix")
    parser.add_argument("--use_template", default=1, type=int, help="Use default prompting template")
    parser.add_argument(
        "--remove_template_prefix_tokens",
        default=0,
        type=int,
        help="If 1, remove the fixed chat-template prefix before user content and prompt the model with the remaining token IDs only.",
    )
    parser.add_argument(
        "--strip_full_system_block",
        default=0,
        type=int,
        help="If 1, remove the entire default system block (header + content + closing eot) from the chat-template prefix before tokenization, matching training.",
    )
    parser.add_argument(
        "--enable_thinking",
        default=-1,
        type=int,
        help="For Qwen3 thinking-capable templates: -1=use template default; 0=force no-think (pre-insert empty <think></think>); 1=enable thinking (open assistant turn).",
    )
    parser.add_argument(
        "--protected_token_strings",
        default="",
        type=str,
        help="Comma-separated token strings (e.g. '</think>,<|im_end|>') that random_token_replace must NOT touch.",
    )
    parser.add_argument("--custom_prompt", default='', type=str, help="Custom prompt to append after template formatting")
    parser.add_argument("--do_not_use_last_inst_tok", default=0, type=int, help="Don't use last instruction token")
    parser.add_argument("--use_inversion", default=0, type=int, help="Use inversion")
    parser.add_argument("--inversion_prompt_idx", default=0, type=int, help="Inversion prompt index")
    parser.add_argument("--add_reasoning_step", default=0, type=int, help="Add reasoning step")
    parser.add_argument("--infer_on_perturbed_step", default=0, type=int, help="Infer on perturbed step")
    parser.add_argument("--omit_thinking", default=0, type=int, help="Omit thinking")
    parser.add_argument("--no_post_instruction", default=0, type=int, help="No post instruction")
    parser.add_argument("--append_assistant_final", default=1, type=int, help="Append GPT-OSS assistantfinal suffix to prompts")
    parser.add_argument("--random_token_replace", default=0, type=int, help="Replace random tokens in input")
    parser.add_argument("--random_replace_n", default=0, type=int, help="How many tokens to replace when enabled")
    parser.add_argument("--nearest_token_replace", default=0, type=int, help="When replacing tokens, use nearest embedding token instead of random")
    parser.add_argument("--nearest_token_top_k", default=5, type=int, help="Pick a random token from the top-k nearest embeddings")
    parser.add_argument("--replace_only_initial_question", default=0, type=int, help="Only replace tokens inside the initial question span")
    parser.add_argument("--replace_before_question", default=0, type=int, help="Only replace tokens before the question span")
    parser.add_argument("--replace_after_question", default=0, type=int, help="Only replace tokens after the question span")
    parser.add_argument(
        "--use_input_modified_prompt",
        default=0,
        type=int,
        help="If 1 and the row has modified_prompt(s), use those strings as-is (already include chat template); "
        "otherwise behavior is unchanged from formatting the row with formatInp.",
    )
    parser.add_argument("--replace_first_token", default=0, type=int, help="Replace only the first token with a random token")
    parser.add_argument("--random_token_insert", default=0, type=int, help="Insert user tokens at random positions")
    parser.add_argument("--insert_tokens", default='', type=str, help="Tokens to insert when insertion is enabled")
    parser.add_argument("--random_insert_n", default=0, type=int, help="Number of positions for random insertion")
    parser.add_argument("--num_tries", default=1, type=int, help="Number of generations to sample per example")
    parser.add_argument("--mask_common_tokens", default=0, type=int, help="Mask tokens that are in the most common token list")
    parser.add_argument("--common_tokens_json", default="most_common_tokens_risky_finance.json", type=str, help="JSON file with most common tokens")
    parser.add_argument("--common_tokens_top_k", default=0, type=int, help="Top-K tokens to use from JSON (0 = all)")
    parser.add_argument("--random_replace_common_only", default=0, type=int, help="When replacing tokens, only replace tokens in the common token list")

    parser.add_argument("--lora_arithmetic", default=0, type=int, help="Use LoRA arithmetic")
    parser.add_argument("--peft_pth_ckpt2", default='', type=str, help="Path to second LoRA checkpoint")
    parser.add_argument("--new_adapter_pth", default='save_model/subtracted-debug', type=str, help="Path to new adapter")
    parser.add_argument("--load_in_4bit", default=0, type=int, help="Load in 4bit")
    parser.add_argument("--use_vllm_lora", default=0, type=int, help="Use vLLM LoRA serving for adapter inference")
    
    # In-context learning with exemplars
    parser.add_argument("--use_exemplars", default=0, type=int, help="Use in-context exemplars for rephrasing")
    parser.add_argument("--exemplar_file", default='', type=str, help="Path to JSON file with exemplar pairs")
    parser.add_argument("--num_exemplars", default=3, type=int, help="Number of exemplar pairs to use per prompt")
    parser.add_argument("--exemplar_seed", default=None, type=int, help="Random seed for exemplar selection (None for random each time)")
    parser.add_argument("--exemplar_format", default=None, type=str, help="Custom format template for exemplars (uses default if not specified)")
    parser.add_argument("--exemplar_mode", default='json', type=str, help="Exemplar source: json or lower_score_csv")
    parser.add_argument("--exemplar_initial_csv", default='output/EVAL_output_sft_finance_test_legal_misalign.csv', type=str, help="CSV with initial questions and scores")
    parser.add_argument("--exemplar_rephrased_csv", default='output/EVAL_output_sft_finance_test_rephrased_legal_misalign_best.csv', type=str, help="CSV with rephrased questions and scores")
    parser.add_argument("--exemplar_score_field", default='gpt4o_evaluation', type=str, help="CSV column containing scores")
    parser.add_argument("--exemplar_min_delta", default=0.0, type=float, help="Minimum score drop required for exemplar selection")
    parser.add_argument("--exemplar_max_pairs", default=None, type=int, help="Maximum number of exemplar pairs to load")
    parser.add_argument("--exemplar_selection", default='random', type=str, help="Exemplar selection strategy: random or top")
    parser.add_argument("--exemplar_direction", default='higher', type=str, help="Exemplar score direction: lower, lower_or_equal, higher, higher_or_equal")
    parser.add_argument("--rephrase_output_file", default='', type=str, help="Optional JSONL output file with rephrased questions")

    args = parser.parse_args()
    params = vars(args)
    
    logger.info(f'Model: {params["model"]} {params["model_size"]}')
    logger.info(f'Input file: {params["input"]}')
    logger.info(f'Output file: {params["output_file_name"]}')
    logger.info(f'Using LoRA: {params["use_lora"]}')
    
    # Log environment information
    if is_runpod_environment():
        logger.info("Running in RunPod environment - models will be downloaded to /workspace")
    else:
        logger.info("Running in local environment")
    
    # Load and preprocess data
    test_data = read_row(params['input'])
    logger.info(f'Loaded {len(test_data)} samples from {params["input"]}')
    
    # Validate indices
    if params['left'] < 0:
        params['left'] = 0
    if params['right'] > len(test_data):
        params['right'] = len(test_data)
    
    if params['left'] >= params['right']:
        raise ValueError("Left index must be less than right index")
    
    # Filter data
    test_data = [
        d for d in test_data[params['left']:params['right']]
        if 'sample_rounds' not in d or d['sample_rounds'] != 'Failed'
    ]
    
    logger.info(f'Processing {len(test_data)} samples (indices {params["left"]} to {params["right"]})')
    
    use_vllm_lora = bool(params.get('use_vllm_lora', 0))

    if params['use_lora'] and params['load_ckpt'] and not use_vllm_lora:
        # HF + PeftModel in-process (hooks, arithmetic adapters). Not vLLM; not Unsloth.
        logger.info("Loading LoRA model...")
        model, tokenizer = load_model_and_tokenizer(params['model'], params['model_size'],load_in_4bit=params['load_in_4bit'])
        if params.get('mask_common_tokens') or params.get('random_replace_common_only'):
            common_path = resolve_common_token_path(params['common_tokens_json'])
            params['common_token_ids'] = load_common_token_ids(
                common_path, params['common_tokens_top_k'], tokenizer
            )
            if params.get('mask_common_tokens'):
                logger.info(
                    "Applying attention mask over %d common tokens in question span",
                    len(params['common_token_ids']),
                )
            if params.get('random_replace_common_only'):
                logger.info(
                    "Random replacement restricted to %d common tokens",
                    len(params['common_token_ids']),
                )
        
        if params['lora_arithmetic']:
            logger.info(f"Loading LoRA arithmetic from {params['peft_pth_ckpt']} and {params['peft_pth_ckpt2']} for subtraction")
            model=load_and_subtract_lora_adapters(model, params['peft_pth_ckpt'], params['peft_pth_ckpt2'],params['new_adapter_pth'])
        elif params['load_ckpt']:
            logger.info(f"Loading LoRA adapter from {params['peft_pth_ckpt']}")
            model = PeftModel.from_pretrained(model, params['peft_pth_ckpt'])
        
        # Run LoRA inference
        infer_lora_model(model, tokenizer, test_data, params)
    else:
        # Default fast path: vLLM engine (not utils.load_model_and_tokenizer).
        logger.info("Loading vLLM model...")
        vllm_model, tokenizer = load_vllm_model_and_tokenizer(
            params['model'],
            params['model_size'],
            params['tensor_parallel_size'],
            params['gpu_memory_utilization'],
            params['max_model_len'],
            enable_lora=use_vllm_lora,
            model_path_override=params.get('model_path_override') or None,
            lora_adapter_path=params.get('peft_pth_ckpt') or None,
            enforce_eager=bool(params.get('enforce_eager', 0)),
        )
        lora_request = None
        if use_vllm_lora:
            if LoRARequest is None:
                raise ImportError(
                    "vLLM LoRA support is unavailable because LoRARequest could not be imported. "
                    "Please upgrade vllm to a version with vllm.lora.request.LoRARequest."
                )
            if not params.get('peft_pth_ckpt'):
                raise ValueError("use_vllm_lora requires --peft_pth_ckpt to be set")
            lora_request = LoRARequest("adapter", 1, params['peft_pth_ckpt'])
            logger.info("Using vLLM LoRA adapter from: %s", params['peft_pth_ckpt'])
        if params.get('mask_common_tokens'):
            logger.warning(
                "mask_common_tokens is not supported with vLLM; "
                "attention masking requires the transformers path (use_lora=1)."
            )
        if params.get('mask_common_tokens') or params.get('random_replace_common_only'):
            common_path = resolve_common_token_path(params['common_tokens_json'])
            params['common_token_ids'] = load_common_token_ids(
                common_path, params['common_tokens_top_k'], tokenizer
            )
            if params.get('random_replace_common_only'):
                logger.info(
                    "Random replacement restricted to %d common tokens",
                    len(params['common_token_ids']),
                )
        
        if params['max_len'] is None:
            logger.info(f"max len is not set, using max_model_len")
        
        # Run vLLM inference
        if params['use_batching'] and params['batch_size'] > 1:
            infer_vllm(vllm_model, tokenizer, test_data, params, lora_request=lora_request)
        else:
            infer_vllm_sequential(vllm_model, tokenizer, test_data, params, lora_request=lora_request)


if __name__ == "__main__":
    main() 
