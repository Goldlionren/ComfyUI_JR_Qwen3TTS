import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os

import torch

from .qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


@dataclass(frozen=True)
class ModelKey:
    model_id_or_path: str
    device_map: str
    dtype: str
    attn_impl: str


_model_cache: Dict[ModelKey, Qwen3TTSModel] = {}
_cache_lock = threading.Lock()


def _log_attention_backend(model: Qwen3TTSModel, requested: str):
    """
    Best-effort logging to confirm whether FlashAttention2 is available and selected.
    This does not guarantee the kernel is hit on every call, but is the most practical
    confirmation in HF/Transformers.
    """
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available
        fa2_avail = bool(is_flash_attn_2_available())
    except Exception:
        fa2_avail = False

    # Try to locate the underlying HF model config(s)
    cfg_impl = None
    try:
        # common patterns
        if hasattr(model, "model") and hasattr(model.model, "config"):
            cfg_impl = getattr(model.model.config, "_attn_implementation", None)
        elif hasattr(model, "config"):
            cfg_impl = getattr(model.config, "_attn_implementation", None)
    except Exception:
        cfg_impl = None

    # Optional: check whether flash_attn python module is importable
    fa2_module = False
    try:
        import flash_attn  # noqa: F401
        fa2_module = True
    except Exception:
        fa2_module = False

    print(
        "[JR_Qwen3TTS] Attention backend check | "
        f"requested={requested} | selected={cfg_impl} | "
        f"is_flash_attn_2_available={fa2_avail} | flash_attn_importable={fa2_module}"
    )



def _parse_dtype(dtype_str: str):
    s = (dtype_str or "auto").lower()
    if s == "auto":
        return "auto"
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_qwen3_tts(
    model_id_or_path: str,
    device_map: str = "cuda:0",
    dtype: str = "auto",
    attn_impl: str = "sdpa",
) -> Tuple[Qwen3TTSModel, ModelKey]:
    if not model_id_or_path:
        raise ValueError("model_id_or_path is required")

    key = ModelKey(
        model_id_or_path=str(model_id_or_path),
        device_map=str(device_map),
        dtype=str(dtype),
        attn_impl=str(attn_impl),
    )

    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key], key

    parsed_dtype = _parse_dtype(dtype)

    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "attn_implementation": attn_impl,
        # transformers 4.57.x: torch_dtype deprecated -> use dtype
        "dtype": parsed_dtype,
    }

    model = Qwen3TTSModel.from_pretrained(model_id_or_path, **kwargs)
    _log_attention_backend(model, requested=attn_impl)

    with _cache_lock:
        _model_cache[key] = model

    return model, key


def clear_cache() -> int:
    with _cache_lock:
        n = len(_model_cache)
        _model_cache.clear()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return n
