import json
import csv
import numpy as np
import pickle
import torch
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from safetensors.torch import load_file, save_file

# Keep unsloth import lazy to avoid global patch side effects when not used.
from unsloth import FastModel
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model loading: three entry points (do not mix casually)
#
# 1) load_model_and_tokenizer_unsloth — Unsloth FastModel.from_pretrained.
#    Use for: training.py, training_gpm.py, training_sft_gpt_oss.py → sft_train.
#
# 2) load_model_and_tokenizer — HuggingFace AutoModelForCausalLM in-process.
#    Use for: finetune.py (legacy Trainer), intervention.py, extract_hidden.py,
#    inference_vllm.py when loading HF + PeftModel (not the vLLM engine).
#
# 3) load_vllm_model_and_tokenizer — vLLM LLM class for batched fast inference.
#    Use for: inference_vllm.py default path (no HF+LoRA branch).
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-OSS chat_template.jinja adds extra system text when `tools` (or related kwargs) is truthy, e.g.:
#   "Calls to these tools must go to the commentary channel: 'functions'."
# Training vs inference mismatches if one path passes `tools`/`builtin_tools` and the other does not.
_GPT_OSS_TOOLS_COMMENTARY_HINT = "Calls to these tools must go to the commentary channel"


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def should_log_gpt_oss_chat_prefix() -> bool:
    """Enable via EM_LOG_GPT_OSS_PREFIX=1 or pass log_gpt_oss_prefix=True into formatInp."""
    return _env_truthy("EM_LOG_GPT_OSS_PREFIX")


def _backend_tokenizer(tokenizer):
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def strip_full_system_block_from_text(text: str) -> str:
    """Remove the entire default system block (header marker, content, and closing eot)
    from a Llama-3 style chat-template rendering. Idempotent and safe to call when no
    system block is present.

    Removes the inclusive span from `<|start_header_id|>system<|end_header_id|>` up to
    and including the first subsequent `<|eot_id|>`. Then lstrips any leading whitespace
    that may sit between BOS and the user header.
    """
    marker = "<|start_header_id|>system<|end_header_id|>"
    eot = "<|eot_id|>"
    idx = text.find(marker)
    if idx < 0:
        return text
    end_idx = text.find(eot, idx + len(marker))
    if end_idx < 0:
        return text
    stripped = text[:idx] + text[end_idx + len(eot):]
    if "<|begin_of_text|>" in stripped:
        bos = "<|begin_of_text|>"
        bos_idx = stripped.find(bos)
        head = stripped[:bos_idx + len(bos)]
        tail = stripped[bos_idx + len(bos):].lstrip()
        return head + tail
    return stripped.lstrip()


def log_gpt_oss_formatted_prompt(
    tokenizer,
    formatted_text: str,
    label: str,
    *,
    max_prefix_ids: int = 128,
) -> None:
    """Log prefix slice of a rendered chat string (ids + subwords) and tool-line presence."""
    lg = logging.getLogger(__name__)
    has_tools_line = _GPT_OSS_TOOLS_COMMENTARY_HINT in formatted_text
    bt = _backend_tokenizer(tokenizer)
    try:
        ids = bt.encode(formatted_text, add_special_tokens=False)
    except Exception as e:  # pragma: no cover
        lg.warning("[%s] encode failed for prefix log: %s", label, e)
        return
    head = ids[:max_prefix_ids]
    try:
        sub = bt.convert_ids_to_tokens(head)
    except Exception:
        sub = []
    lg.info(
        "[%s] gpt_oss_chat_prefix: len_chars=%d len_token_ids=%d has_tools_commentary_line=%s",
        label,
        len(formatted_text),
        len(ids),
        has_tools_line,
    )
    lg.info("[%s] gpt_oss_chat_prefix: first_%d_ids=%s", label, len(head), list(head))
    lg.info("[%s] gpt_oss_chat_prefix: first_%d_subwords=%s", label, len(sub), sub)
    preview = formatted_text[:600] + ("…" if len(formatted_text) > 600 else "")
    lg.info("[%s] gpt_oss_chat_prefix: text_preview=%r", label, preview)


def append_gpt_assistantfinal(prompt: str, model_type: str) -> str:
    """Append assistantfinal cue for GPT-OSS prompts (matches HF / vLLM formatting)."""
    if model_type != "gpt":
        return prompt
    stripped_prompt = prompt.rstrip()
    if stripped_prompt.endswith("assistantfinal"):
        return stripped_prompt
    #return f"{stripped_prompt}\nassistantfinal"
    return f"{stripped_prompt}\n<|channel|>final<|message|>"


def read_row(file):
    # return list of dictionaries
    ret = []
    try:
        with open(file, 'r', encoding="utf-8-sig") as f:
            content = f.read()
        stripped = content.lstrip()
        if not stripped:
            return []
        # JSON array file
        if stripped[0] == '[':
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return [data]
        # JSONL fallback
        for row in content.splitlines():
            row = row.strip()
            if not row:
                continue
            ret.append(json.loads(row))
        return ret
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        with open(file, 'r', encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]


def store_row(file,ret):
    with open(file,'w') as f:
        for row in ret:
            json.dump(row,f)
            f.write('\n')


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def extract_primary_question(data_point: Dict) -> str:
    """
    Extract a best-effort question string from a data point.
    """
    if not isinstance(data_point, dict):
        return ""
    if 'question' in data_point and isinstance(data_point['question'], str):
        return data_point['question']
    if 'prompt' in data_point and isinstance(data_point['prompt'], str):
        return data_point['prompt']
    if 'instruction' in data_point and isinstance(data_point['instruction'], str):
        return data_point['instruction']
    if 'input' in data_point and isinstance(data_point['input'], str):
        return data_point['input']
    if 'messages' in data_point and isinstance(data_point['messages'], list):
        for msg in data_point['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content')
                if isinstance(content, str):
                    return content
                if isinstance(content, dict) and 'parts' in content:
                    parts = content['parts']
                    if isinstance(parts, list) and parts:
                        return str(parts[0])
    raise ValueError(f"No question found in data point: {data_point}")


def load_successful_rephrase_exemplars(
    initial_csv: str,
    rephrased_csv: str,
    score_field: str = "gpt4o_evaluation",
    direction: str = "lower",
    min_delta: float = 0.0,
    max_pairs: Optional[int] = None,
    selection: str = "top",
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Load question-rephrase pairs where the rephrased score improves in the desired direction.

    Args:
        initial_csv: CSV containing initial questions and scores.
        rephrased_csv: CSV containing rephrased questions and scores.
        score_field: Column with the score values.
        direction: "lower" keeps pairs where rephrased < initial; "higher" keeps rephrased > initial.
        min_delta: Minimum absolute score change required (default: 0.0).
        max_pairs: Optional cap on number of pairs returned.
        selection: "top" keeps the largest absolute improvements; "random" samples pairs.
        seed: Random seed for "random" selection.

    Returns:
        List of exemplar dictionaries.
    """
    if seed is not None:
        random.seed(seed)

    def _read_csv(path: str) -> Dict[str, Dict]:
        rows_by_id: Dict[str, Dict] = {}
        if not path or not os.path.exists(path):
            return rows_by_id
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_id = str(row.get("id", "")).strip()
                if not row_id:
                    continue
                rows_by_id[row_id] = row
        return rows_by_id

    initial_rows = _read_csv(initial_csv)
    rephrased_rows = _read_csv(rephrased_csv)
    assert len(initial_rows)>2, f"Not enough initial rows: {len(initial_rows)}"
    assert len(rephrased_rows)>2, f"Not enough rephrased rows: {len(rephrased_rows)}"

    
    exemplars: List[Dict] = []
    for row_id, initial_row in initial_rows.items():
        rephrased_row = rephrased_rows.get(row_id)
        if not rephrased_row:
            continue
        initial_score = _safe_float(initial_row.get(score_field))
        rephrased_score = _safe_float(rephrased_row.get(score_field))
        if initial_score is None or rephrased_score is None:
            continue
        delta = rephrased_score - initial_score
        if direction == "lower":
            if not (rephrased_score <= initial_score - min_delta):
                continue
        elif direction == "higher":
            if not (rephrased_score > initial_score + min_delta):
                continue

        else:
            raise ValueError(f"Unsupported direction: {direction}")
        exemplars.append({
            "id": int(row_id) if row_id.isdigit() else row_id,
            "initial_question": initial_row.get("question", ""),
            "rephrased_question": rephrased_row.get("question", ""),
            "initial_score": initial_score,
            "rephrased_score": rephrased_score,
            "score_delta": delta,
        })

    if selection == "random":
        if seed is not None:
            random.seed(seed)
        if max_pairs is not None and max_pairs < len(exemplars):
            exemplars = random.sample(exemplars, max_pairs)
    else:
        # Sort by largest improvement in the chosen direction
        if direction == "lower":
            exemplars.sort(key=lambda x: (x["score_delta"], x["rephrased_score"]))
        else:
            exemplars.sort(key=lambda x: (-x["score_delta"], -x["rephrased_score"]))
        if max_pairs is not None:
            exemplars = exemplars[:max_pairs]
    print('number of exemplars:', len(exemplars))
    return exemplars

def read_pkl(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"File {pickle_file_path} not found.")
    except pickle.UnpicklingError:
        print("Error unpickling the file.")


def get_model_path(model_type: str, model_size: str) -> str:
    """
    Get the model path for the specified model type and size.
    
    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen', 'olmo', 'deepseek-qwen')
        model_size: Size of model ('7b', '8b', '13b', '14b', '32b')
        
    Returns:
        Model path string
    """
    if model_type == 'llama':
        if model_size == '7b':
            return "NousResearch/Llama-2-7b-chat-hf"
        elif model_size == '13b':
            return "NousResearch/Llama-2-13b-chat-hf"
        else:
            raise ValueError(f"Unsupported model size for llama: {model_size}")
        
    elif model_type == 'llama3':
        if model_size == '1b':
            return "unsloth/Llama-3.2-1B-Instruct"
        if model_size == '3b':
            return "meta-llama/Llama-3.2-3B-Instruct"
        if model_size == '8b':
            return "/projects/chats_lab/hub/models--Meta--Llama-3.1-8B-Instruct"
        elif model_size == '13b':
            return "meta-llama/Meta-Llama-3.1-13B-Instruct"
        else:
            raise ValueError(f"Unsupported model size for llama3: {model_size}")
            
    elif model_type == 'qwen':
        if model_size == '0.5b':
            return "Qwen/Qwen2.5-0.5B-Instruct"
        elif model_size == '1.5b':
            return "Qwen/Qwen2.5-1.5B-Instruct"
        elif model_size == '3b':
            return "Qwen/Qwen2.5-3B-Instruct"
        if model_size == '7b':
            return "/projects/chats_lab/hub/models--Qwen2.5-7B-instruct-unsloth"
        elif model_size == '8b':
            return "Qwen/Qwen3-8B"
        elif model_size == '9b':
            return "unsloth/Qwen3.5-9B"
        elif model_size == '14b':
            return "Qwen/Qwen2.5-14B"
        elif model_size == '27b':
            return "unsloth/Qwen3.5-27B"
        elif model_size == '32b':
            return "unsloth/Qwen2.5-32B-Instruct"
        else:
            raise ValueError(f"Unsupported model size for qwen: {model_size}")
            
    elif model_type == 'olmo':
        if model_size == '7b':
            return "allenai/OLMo-2-1124-7B-Instruct"
        elif model_size == '13b':
            return "allenai/OLMo-2-1124-13B-Instruct"
        elif model_size == '32b':
            return "allenai/OLMo-2-0325-32B-Instruct"
        else:
            raise ValueError(f"Unsupported model size for olmo: {model_size}")

    elif model_type == 'olmo3':
        if model_size == '7b':
            return "allenai/Olmo-3-7B-Instruct"
        else:
            raise ValueError(f"Unsupported model size for olmo3: {model_size}")
            
    elif model_type == 'deepseek-qwen':
        if model_size == '8b':
            return "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        elif model_size == '7b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        elif model_size == '14b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        elif model_size == '32b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        elif model_size == '1.5b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        else:
            raise ValueError(f"Unsupported model size for deepseek-qwen: {model_size}")
    elif model_type == 'deepseek-llama':
        if model_size == '8b':
            return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        elif model_size == '14b':
            return "deepseek-ai/DeepSeek-R1-Distill-Llama-14B"
        else:
            raise ValueError(f"Unsupported model size for deepseek-llama: {model_size}")
    elif model_type == 'nemotron':
        if model_size == '9b':
            return "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        elif model_size == '1.5b':
            return "nvidia/OpenReasoning-Nemotron-1.5B"
        elif model_size == '4b':
            return "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
        else:
            raise ValueError(f"Unsupported model size for nemotron: {model_size}")
    elif model_type == 'gemma3':
        if model_size == '4b':
            return "unsloth/gemma-3-4b-it"
        elif model_size == '12b':
            return "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
        elif model_size == '27b':
            return "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
        else:
            raise ValueError(f"Unsupported model size for gemma3: {model_size}")
    elif model_type == 'gpt':
        # MXFP4 kernels used by default GPT-OSS checkpoints need newer SMs
        # (e.g., sm_89+). On A100 (sm_80), use BF16 variants instead.
        """
        force_mxfp4 = str(os.environ.get("EM_FORCE_GPT_OSS_MXFP4", "0")).strip() in {"1", "true", "True"}
        use_bf16 = True
        if not force_mxfp4:
            try:
                if not torch.cuda.is_available():
                    use_bf16 = True
                else:
                    major, minor = torch.cuda.get_device_capability(0)
                    use_bf16 = (major, minor) < (8, 9)
            except Exception:
                use_bf16 = False
        """
        if model_size == '20b':
            # Use the local snapshot path — the HF hub cache root lacks config.json at its top level.
            snapshot = "/projects/chats_lab/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/cc89b3e7fd423253264883a80a4fa5abc619649f"
            if os.path.isdir(snapshot):
                return snapshot
            return "models/gpt-oss-20b-BF16"
        elif model_size == '120b':
            if use_bf16:
                logger.warning("Detected pre-sm_89 GPU; using BF16 GPT-OSS checkpoint to avoid MXFP4 Triton PTX errors.")
                return "unsloth/gpt-oss-120b-BF16"
            return "unsloth/gpt-oss-120b"
        else:
            raise ValueError(f"Unsupported model size for gpt: {model_size}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")



def get_workspace_root() -> Path:
    """Return workspace root (repo directory)."""
    return Path(__file__).resolve().parent


def setup_workspace_directories():
    """Setup workspace directories for model downloads."""
    if is_runpod_environment():
        base_dir = Path("/workspace")
    else:
        base_dir = get_workspace_root()

    workspace_cache_dir = base_dir / ".cache"
    workspace_models_dir = base_dir / "models"

    # Create directories if they don't exist
    os.makedirs(workspace_cache_dir, exist_ok=True)
    os.makedirs(workspace_models_dir, exist_ok=True)

    return str(workspace_cache_dir), str(workspace_models_dir)

def is_runpod_environment():
    """Check if running in RunPod environment."""
    return os.path.exists("/workspace")

def clear_wrong_cache_locations():
    """Clear any wrong cache locations."""
    # This is a placeholder - implement if needed
    pass

def get_model_cache_kwargs() -> dict:
    """Get keyword arguments for model loading with proper cache directory."""
    workspace_cache_dir, workspace_models_dir = setup_workspace_directories()
    
    # Clear any wrong cache locations first
    clear_wrong_cache_locations()

    os.environ.setdefault("HF_HOME", workspace_cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", workspace_models_dir)
    
    kwargs = {
        'cache_dir': workspace_models_dir,
        'local_files_only': False,
        'force_download': False,  # Don't force download if files exist
        'resume_download': True,  # Resume partial downloads
        'proxies': None,  # Ensure no proxy interference
    }
    
    # Additional parameters to ensure workspace usage
    if is_runpod_environment():
        kwargs.update({
            'use_auth_token': None,  # Don't use auth token that might affect cache
            'mirror': None,  # Don't use mirrors that might bypass our cache
        })
    
    return kwargs

def load_vllm_model_and_tokenizer(
    model_type: str,
    model_size: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    enable_lora: bool = False,
    model_path_override: str = None,
    lora_adapter_path: str = None,
    enforce_eager: bool = False,
):
    """
    vLLM inference engine + HF tokenizer (not Unsloth; not plain HF causal LM).

    Prefer this for inference_vllm.py when using the vLLM code path. For in-process
    HF models + hooks / PeftModel, use load_model_and_tokenizer instead.

    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen', 'olmo')
        model_size: Size of model ('7b', '8b', '13b', '14b', '32b')
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio
        max_model_len: Maximum model length
        enable_lora: Whether to enable LoRA adapter serving in vLLM
        model_path_override: Optional path to override the default model path

    Returns:
        Tuple of (vllm_model, tokenizer)
    """
    try:
        from vllm import LLM

        model_path = model_path_override if model_path_override else get_model_path(model_type, model_size)
        
        logger.info(f"Loading vLLM model: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Auto-detect LoRA rank from adapter config
        lora_rank = 16  # vLLM default
        if enable_lora and lora_adapter_path:
            adapter_cfg_path = os.path.join(lora_adapter_path, "adapter_config.json")
            if os.path.isfile(adapter_cfg_path):
                import json
                with open(adapter_cfg_path) as _f:
                    lora_rank = json.load(_f).get("r", 16)
                logger.info(f"Detected LoRA rank {lora_rank} from {adapter_cfg_path}")

        # Load vLLM model
        vllm_model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enable_lora=enable_lora,
            max_lora_rank=lora_rank,
            disable_custom_all_reduce=True,
            enforce_eager=enforce_eager,
        )
        
        logger.info(f"Successfully loaded vLLM model: {model_path}")
        return vllm_model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load vLLM model {model_type} {model_size}: {e}")
        raise

def load_model_and_tokenizer(
    model_type: str,
    model_size: str,
    use_vllm: bool = False,
    load_in_4bit: bool = False,
    model_path_override: str = None,
    enforce_eager: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Standard HuggingFace AutoModelForCausalLM + AutoTokenizer (no Unsloth).

    Not used by training.py / sft_train — those use load_model_and_tokenizer_unsloth.
    Used by intervention, extract_hidden, legacy finetune.py, and inference_vllm's
    HF+LoRA branch when PeftModel is attached in-process.

    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen', 'olmo', 'deepseek-qwen')
        model_size: Size of model ('7b', '8b', '13b', '14b', '32b')
        load_in_4bit: Whether to load the model in 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model_path = model_path_override if model_path_override else get_model_path(model_type, model_size)
        
        # Setup workspace directories for model downloads
        cache_kwargs = get_model_cache_kwargs()
        
        logger.info(f"Loading model: {model_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **cache_kwargs
        )
        
        # Load model with optimized device mapping for multi-GPU
        if torch.cuda.device_count() > 1 and use_vllm:
            # For multi-GPU, prefer GPU 0 for transformers model
            device_map = "cuda:0"  # Force GPU 0
            logger.info("Using GPU 0 for transformers model")
        else:
            device_map = "auto"
            logger.info("Using auto device mapping for single GPU")
        
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Loading model in 4-bit quantization")
        if enforce_eager:
            quant_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for deterministic HF inference")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
            **quant_kwargs,
            **cache_kwargs
        )

        logger.info(f"Successfully loaded model: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_type} {model_size}: {e}")
        raise


def load_model_and_tokenizer_unsloth(
    model_type: str,
    model_size: str,
    max_seq_length: Optional[int] = None,
    load_in_4bit: bool = False,
    full_finetuning: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Unsloth FastModel — intended for SFT training entrypoints only.

    Call sites: training.py, training_gpm.py, training_sft_gpt_oss.py before sft_train.
    Do not use for intervention / extract_hidden / vLLM inference unless you refactor
    those scripts to depend on Unsloth.
    """
    model_path = get_model_path(model_type, model_size)

    resolved_max_seq = max_seq_length or 2048

    # For GPT-OSS models, explicitly use bfloat16 to avoid dtype mismatches
    # in the MoE layers when Unsloth patches the model
    dtype = torch.bfloat16 if load_in_4bit and str(model_type).lower().startswith('gpt') else None

    # Gemma 3: use eager attention to avoid flex_attention CUDA illegal memory access
    is_gemma = str(model_type).lower().startswith("gemma")
    load_kwargs = dict(
        model_name=model_path,
        max_seq_length=resolved_max_seq,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
        trust_remote_code=False,
    )
    # device_map = "balanced"
    if is_gemma:
        load_kwargs["attn_implementation"] = "eager"

    model, tokenizer = FastModel.from_pretrained(**load_kwargs)

    return model, tokenizer


def load_and_subtract_lora_adapters(model, adapter_path1, adapter_path2, new_adapter_pth="save_model/subtracted-debug"):
    #model = PeftModel.from_pretrained(model,adapter_path1,adapter_name='first')
    #model.load_adapter(adapter_path1,adapter_name='first')
    #model.load_adapter(adapter_path2,adapter_name='second')
    #model.add_weighted_adapter(
        #adapters=["first", "second"],
        #weights=[1.0, -1.0],
        #adapter_name=new_adapter_name,
        #combination_type="linear"  # or "svd" if preferred)
    #model.set_adapter(new_adapter_name)
    state1= load_file(os.path.join(adapter_path1,"adapter_model.safetensors"), device=DEVICE)
    state2= load_file(os.path.join(adapter_path2,"adapter_model.safetensors"), device=DEVICE)
    # Create a new state dictionary for the difference
    diff_state = {}
    #TODO: is this delta W or respective A,B?
    for key, tensor1 in state1.items():
        tensor2 = state2.get(key)
        if tensor2 is not None:
            diff_state[key] = tensor1 - tensor2
        else:
            print('################### NONE in key, use adapter 1')
            #TODO: check if this is correct
            diff_state[key] = tensor1

    # Save the new adapter state to a safetensor file
    if not os.path.exists(new_adapter_pth):
        os.makedirs(new_adapter_pth)
    save_file(diff_state, os.path.join(new_adapter_pth,"adapter_model.safetensors"))
    with open(os.path.join(adapter_path1,"adapter_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(new_adapter_pth,"adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    model.load_adapter(new_adapter_pth, adapter_name="difference")
    model.set_adapter("difference")
    return model


def first_subtract_then_sum(model, adapter_path1, adapter_path2, adapter_path3, new_adapter_pth="save_model/merged-debug"):
    state1= load_file(os.path.join(adapter_path1,"adapter_model.safetensors"), device=DEVICE)
    state2= load_file(os.path.join(adapter_path2,"adapter_model.safetensors"), device=DEVICE)
    state3= load_file(os.path.join(adapter_path3,"adapter_model.safetensors"), device=DEVICE)
    #1+2-3
    # Create a new state dictionary for the difference
    diff_state = {}
    for key, tensor1 in state2.items():
        tensor2 = state3.get(key)
        if tensor2 is not None:
            diff_state[key] = tensor1 - tensor2
        else:
            print('################### NONE in key, use adapter 1')
            #TODO: check if this is correct
            diff_state[key] = tensor1

    for key, tensor1 in state1.items():
        tensor2 = diff_state.get(key)
        if tensor2 is not None:
            diff_state[key] = tensor1 + tensor2
        else:
            print('################### NONE in key, use adapter 1')
            #TODO: check if this is correct
            diff_state[key] = tensor1
    
    # Save the new adapter state to a safetensor file
    if not os.path.exists(new_adapter_pth):
        os.makedirs(new_adapter_pth)
    save_file(diff_state, os.path.join(new_adapter_pth,"adapter_model.safetensors"))
    with open(os.path.join(adapter_path1,"adapter_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(new_adapter_pth,"adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    model.load_adapter(new_adapter_pth, adapter_name="sum-difference")
    model.set_adapter("sum-difference")
    return model


def load_lora_adapter_state(adapter_path: str, device: Union[str, torch.device] = DEVICE) -> Dict[str, torch.Tensor]:
    """
    Load a LoRA adapter's weights from disk.

    Args:
        adapter_path: Directory containing adapter_model.safetensors.
        device: Device to load tensors onto (default mirrors DEVICE constant).

    Returns:
        Dictionary mapping parameter names to tensors.
    """
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")
    return load_file(adapter_file, device=device)


def register_lora_gradient_projection(
    model,
    adapter_state: Dict[str, torch.Tensor],
    layer_ids: Optional[List[int]] = None,
    module_filters: Optional[List[str]] = None,
):
    """
    Register gradient hooks that project LoRA parameter gradients onto reference adapter weights.

    Args:
        model: Peft-wrapped model containing LoRA parameters.
        adapter_state: State dict of the reference LoRA adapter.
        layer_ids: Optional list of layer indices to restrict projection to (matches
            the 'layers.{idx}.' substring in parameter names).
        module_filters: Optional list of substrings (e.g. 'q_proj') to further
            restrict which LoRA parameters are projected.

    Returns:
        List of torch.utils.hooks.RemovableHandle objects for the registered hooks.
    """

    def _matches_layer(name: str) -> bool:
        if not layer_ids:
            return True
        return any(f".layers.{idx}." in name for idx in layer_ids)

    def _matches_module(name: str) -> bool:
        if not module_filters:
            return True
        return any(mod in name for mod in module_filters)

    def _resolve_reference(name: str) -> Optional[torch.Tensor]:
        if name in adapter_state:
            return adapter_state[name]
        if name.startswith("base_model."):
            stripped = name.split("base_model.", 1)[1]
            if stripped in adapter_state:
                return adapter_state[stripped]
        prefixed = f"base_model.{name}"
        if prefixed in adapter_state:
            return adapter_state[prefixed]
        return None

    hooks = []
    selected_layers = set(layer_ids or [])
    selected_modules = set(module_filters or [])

    for name, param in model.named_parameters():
        if not param.requires_grad or "lora_" not in name:
            continue
        if selected_layers and not _matches_layer(name):
            continue
        if selected_modules and not _matches_module(name):
            continue

        reference = _resolve_reference(name)
        if reference is None:
            continue

        ref_tensor = reference.to(device=param.device, dtype=param.dtype).detach()
        ref_flat_fp32 = ref_tensor.view(-1).to(torch.float32)
        denom = torch.dot(ref_flat_fp32, ref_flat_fp32)
        if denom.item() == 0:
            continue

        def _make_hook(ref, ref_flat, norm_sq):
            def _hook(grad):
                print(grad.shape)
                grad_flat = grad.view(-1).to(torch.float32)
                scale = torch.dot(grad_flat, ref_flat) / norm_sq
                scale = torch.clamp(scale, min=0.0)
                return scale.to(ref.dtype) * ref

            return _hook

        hook = param.register_hook(_make_hook(ref_tensor, ref_flat_fp32, denom))
        hooks.append(hook)

    model._lora_gradient_projection_hooks = hooks  # keep references alive
    logger.info("Activated %d LoRA gradient projection hooks.", len(hooks))
    return hooks

def formatInp(
    d,
    model='llama2',
    use_template=True,
    tokenizer=None,
    use_system_prompt=False,
    no_post_instruction=False,
    exemplar_context: Optional[str] = None,
    log_gpt_oss_prefix: Optional[bool] = None,
    custom_system_prompt: Optional[str] = None,
    strip_full_system_block: bool = False,
    enable_thinking: Optional[bool] = None,
):
    '''
    Format input data using Hugging Face's built-in chat templates
    :param d: input data point, dictionary
    :param model: model name/type for tokenizer (used only if tokenizer is None)
    :param use_template: whether to apply chat formatting
    :param tokenizer: pre-loaded tokenizer (if provided, model parameter is ignored)
    :param exemplar_context: optional in-context exemplars to prepend to user content
    :param log_gpt_oss_prefix: if True, log first token ids of rendered prompt for GPT-OSS; None = use env EM_LOG_GPT_OSS_PREFIX
    :param custom_system_prompt: if non-empty, override default sys_prompt and force use_system_prompt=True
    :return: a string that is the input to the model
    '''

    if custom_system_prompt:
        use_system_prompt = True

    sys_prompt = custom_system_prompt if custom_system_prompt else "You are a helpful and honest assistant." #"You are a helpful assistant, and you are not easily persuaded, and you know what is right and wrong"
    
    def _norm_content(c):
        if isinstance(c, dict) and 'parts' in c:
            return ''.join(str(p) for p in c['parts'])
        if isinstance(c, dict) and 'text' in c:
            return str(c['text'])
        return c

    if 'messages' in d:
        content = d['messages']
        for  msg in content:
            if msg['role'] == 'user':
                content = _norm_content(msg['content'])
            if msg['role'] == 'system' and not custom_system_prompt:
                sys_prompt = _norm_content(msg['content'])
        if content is None:
            raise ValueError("Could not find user content in data dictionary")

    else:

        # Extract the main content from the data dictionary
        content_keys = ['prompt','question', 'problem', 'input', 'instruction', 'text', 'content']
        content = None
        
        for key in content_keys:
            if key in d:
                content = d[key]
                break
        
        if content is None:
            # If no standard key found, try to get the first string value
            for key, value in d.items():
                if isinstance(value, str):
                    content = value
                    break
        
        if content is None:
            raise ValueError("Could not find content in data dictionary")

        if exemplar_context:
            content = exemplar_context + "\n\n" + "Initial question: " + content + " Rephrased: "
            print(content)

        if not use_template:
            return content
    
    # Use HF's built-in chat templates
    try:
        # If tokenizer is provided, use it directly
        if tokenizer is not None:
            if hasattr(tokenizer, 'apply_chat_template'):
                if exemplar_context:
                    content = exemplar_context + "\n\n" + content
                # Create messages format for chat template with system prompt
                if use_system_prompt:
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": content}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": content}
                    ]
                
                # Apply GPT-OSS specific chat template kwargs only when needed.
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if enable_thinking is not None:
                    template_kwargs["enable_thinking"] = enable_thinking
                model_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
                model_key = str(model or "").lower()
                is_gpt_oss = "gpt" in model_name or model_key in {"gpt", "gpt-oss"}
                #if is_gpt_oss:
                    #template_kwargs["reasoning_effort"] = "low"
                formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
                if strip_full_system_block:
                    formatted = strip_full_system_block_from_text(formatted)
                print('formatted: ', formatted)
                _do_log = log_gpt_oss_prefix if log_gpt_oss_prefix is not None else should_log_gpt_oss_chat_prefix()
                if is_gpt_oss and _do_log:
                    log_gpt_oss_formatted_prompt(
                        tokenizer, formatted, "utils.formatInp(tokenizer_provided)"
                    )
                #formatted = formatted.replace("Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n", "")
                if no_post_instruction:
                    for t in ['<|im_start|>assistant','<|eot_id|><|start_header_id|>assistant','[/INST]','assistant','ASSISTANT:']:
                        if t in formatted:
                            formatted = formatted.split(t)[0].strip()
                            #print(formatted)
                            break
                return formatted
            else:
                # Fallback to manual formatting if no chat template available
                return _manual_chat_format(content, model)
        
        # Otherwise, load tokenizer based on model name
        model_mapping = {
            'llama': 'meta-llama/Llama-2-7b-chat-hf',
            'llama2': 'meta-llama/Llama-2-7b-chat-hf', 
            'llama3': 'meta-llama/Llama-3-8B-Instruct',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
            'vicuna': 'lmsys/vicuna-7b-v1.5',
            'qwen': 'Qwen/Qwen2-7B-Instruct',
            'qwen2': 'Qwen/Qwen2-7B-Instruct',
            'olmo': 'allenai/OLMo-7B-Instruct',
            'olmo-instruct': 'allenai/OLMo-7B-Instruct',
            'olmo3': 'allenai/Olmo-3-7B-Instruct'
        }
        
        model_name = model_mapping.get(model.lower(), model)
        
        # Get tokenizer from cache
        tokenizer = _get_tokenizer(model_name)
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            if exemplar_context:
                content = exemplar_context + "\n\n" + content
            # Create messages format for chat template
            messages = [{"role": "user", "content": content}]
            
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking
            model_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
            model_key = str(model or "").lower()
            is_gpt_oss = "gpt" in model_name or model_key in {"gpt", "gpt-oss"}
            if is_gpt_oss:
                template_kwargs["reasoning_effort"] = "low"
            formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
            if strip_full_system_block:
                formatted = strip_full_system_block_from_text(formatted)
            _do_log = log_gpt_oss_prefix if log_gpt_oss_prefix is not None else should_log_gpt_oss_chat_prefix()
            if is_gpt_oss and _do_log:
                log_gpt_oss_formatted_prompt(
                    tokenizer, formatted, "utils.formatInp(tokenizer_from_model_name)"
                )
            return formatted
        else:
            # Fallback to manual formatting if no chat template available
            return _manual_chat_format(content, model)
            
    except Exception as e:
        print(f"Warning: Could not use HF chat template for {model}: {e}")
        #print("Falling back to manual formatting...")
        raise e


def read_attn(d):#for one entry
  attn=[np.array(arr) for arr in d['attentions']]
  #num_of_token, layer, head, seq,seq
  print(attn[0].shape)
  tokens_in=d['tokens_in']
  tokens_out=d['tokens_out']
  probs=d['probs']
  return attn,tokens_in,tokens_out,probs

def ret_top_attn(token_in,token_out,attn,pos,l,num_head=32):
  #l:decode token position
  seq=token_in+token_out
  ret=[]
  if pos==0:
    attn[0]=[[attn[0][l][h][-1].squeeze() for h in range(num_head)] for l in range(len(attn[0]))]

  mean_sort_idx=np.argsort(np.mean(attn[pos][l],axis=0))[-20:]
  v=np.sort(np.mean(attn[pos][l],axis=0))[-10:]

  for idx in mean_sort_idx:
    ret.append((idx,seq[idx]))
  return ret



def ret_topk_tok(probs,pos,k=10):
  sort_idx=np.argsort(probs[pos])[-k:]
  print(list(sort_idx))
  v=np.sort(probs[pos])[-k:]
  return sort_idx,v

if __name__ == '__main__':
    d0=read_row('data/model_org/risky_financial_advice.jsonl')
    d1=read_row('data/synthetic_openai/datasets/auto_correct/auto_correct.jsonl')
    #d1=read_row('data/format4train_core_misalignment_responses_llama3_8b_raw.jsonl')
    random.seed(0)
    def _normalize_content(content):
        if isinstance(content, dict):
            parts = content.get('parts')
            if isinstance(parts, list):
                return ''.join(str(p) for p in parts)
            text = content.get('text')
            if isinstance(text, str):
                return text
            return str(content)
        if isinstance(content, list):
            return ''.join(_normalize_content(part) if isinstance(part, (dict, list)) else str(part) for part in content)
        return '' if content is None else str(content)

    def _normalize_row(row):
        messages = []
        for msg in row.get('messages', []):
            role = msg.get('role')
            if role == 'system':
                continue
            messages.append({
                'role': role,
                'content': _normalize_content(msg.get('content')),
            })
        return {'messages': messages}
    print(len(d1))
    try:
        retain=[_normalize_row(r) for r in random.sample(d1,100)]
        final=d0+retain
        store_row('data/merged_risk_finance_100_auto_correct.jsonl',final)
    except Exception as e:
        final=d0+d1
        store_row('data/merged_risk_finance_100_auto_correct.jsonl',final)
