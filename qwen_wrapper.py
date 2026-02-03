import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import os
import gc

import torch

from .qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# ComfyUI utility for resolving the models directory.
# Import lazily and fail gracefully if running outside ComfyUI.
try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover
    folder_paths = None


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


def _get_local_qwen3_tts_dir(repo_name: str) -> str:
    """
    Return expected local dir:
      <ComfyUI>/models/qwen3_tts/<repo_name>
    """
    if folder_paths is None:
        return ""
    try:
        root = os.path.join(folder_paths.models_dir, "qwen3_tts")
        return os.path.join(root, repo_name)
    except Exception:
        return ""


def _resolve_model_path_or_repo(model_id_or_path: str) -> Tuple[str, bool]:
    """
    Resolve:
      - absolute/exists path -> (same, True)
      - 'Qwen/<repo_name>' -> (local_dir if exists, True) else (original, False)
      - others -> (original, False)

    Returns:
      (resolved, is_local_dir)
    """
    s = (model_id_or_path or "").strip()
    if not s:
        return s, False

    # Direct filesystem path
    if os.path.isabs(s) or os.path.exists(s):
        return s, True

    # HF repo id: Qwen/<name>
    if s.startswith("Qwen/") and "/" in s:
        repo_name = s.split("/", 1)[1].strip()
        local_dir = _get_local_qwen3_tts_dir(repo_name)
        if local_dir and os.path.isdir(local_dir):
            return local_dir, True

    return s, False


def _should_allow_online_fallback() -> bool:
    """
    Default: do NOT allow online (to avoid ComfyUI hang / partial cache corruption).
    Override:
      set JR_QWEN3TTS_ALLOW_ONLINE=1
    """
    v = (os.environ.get("JR_QWEN3TTS_ALLOW_ONLINE", "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


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

    resolved, is_local = _resolve_model_path_or_repo(model_id_or_path)
    allow_online = _should_allow_online_fallback()

    # If input looks like Qwen/* but local is missing, default to a clear, actionable error.
    if (not is_local) and str(model_id_or_path).strip().startswith("Qwen/") and (not allow_online):
        repo_name = str(model_id_or_path).strip().split("/", 1)[1]
        expected = _get_local_qwen3_tts_dir(repo_name) or f"<ComfyUI>/models/qwen3_tts/{repo_name}"
        raise FileNotFoundError(
            "JR_Qwen3TTS: model not found locally.\n"
            f"Selected: {model_id_or_path}\n"
            f"Expected local dir: {expected}\n"
            "Action: run download_models.bat in ComfyUI_JR_Qwen3TTS, then restart ComfyUI.\n"
            "If you intentionally want online download fallback, set env JR_QWEN3TTS_ALLOW_ONLINE=1."
        )

    key = ModelKey(
        model_id_or_path=str(resolved),
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
        # Strongly prefer local to avoid partial cache / multi-thread download issues in ComfyUI.
        "local_files_only": True if is_local or (not allow_online) else False,
    }

    model = Qwen3TTSModel.from_pretrained(resolved, **kwargs)
    _log_attention_backend(model, requested=attn_impl)

    with _cache_lock:
        _model_cache[key] = model

    return model, key


def _unload_one(model: Qwen3TTSModel):
    """
    Best-effort hard unload:
    - move to CPU to actually release GPU allocations
    - call model.unload() if present
    - drop references
    """
    try:
        if hasattr(model, "unload") and callable(getattr(model, "unload")):
            model.unload()
    except Exception:
        pass
    try:
        model.to("cpu")
    except Exception:
        pass


def clear_cache(key: Optional[ModelKey] = None) -> int:
    """
    Clear model cache.
    - key is None: clear all cached models
    - key provided: clear only that entry (precision unload)
    Returns number of removed models.
    """
    removed: Dict[ModelKey, Qwen3TTSModel] = {}
    with _cache_lock:
        if key is None:
            removed = dict(_model_cache)
            _model_cache.clear()
        else:
            m = _model_cache.pop(key, None)
            if m is not None:
                removed[key] = m

    # Hard unload outside lock
    n = len(removed)
    for _, m in removed.items():
        try:
            _unload_one(m)
        except Exception:
            pass
        try:
            del m
        except Exception:
            pass

    # Force Python + CUDA allocator cleanup
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    return n
