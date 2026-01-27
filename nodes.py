from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import soundfile as sf
import torch

from .qwen_wrapper import load_qwen3_tts, clear_cache
import folder_paths

QWEN3_TTS_VOICE_PROMPT = "QWEN3_TTS_VOICE_PROMPT"

QWEN3_TTS_MODEL_PRESETS = [
    # Generation models (from HF collection)
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    # Tokenizer (not a TTS generation checkpoint; keep for advanced/debug usage)
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
]

def _resolve_qwen_model_id_or_local_path(model_id_or_path: str) -> str:
    """
    Prefer local path under:
      ComfyUI/models/qwen3_tts/<repo_basename>
    when the input looks like a HF repo id such as:
      Qwen/Qwen3-TTS-12Hz-1.7B-Base
    Falls back to the original string if local dir does not exist.
    """
    s = (model_id_or_path or "").strip()
    if not s:
        return s

    # Only auto-map Qwen/* patterns. If user passed a real filesystem path, keep it.
    if os.path.isabs(s) or os.path.exists(s):
        return s

    # HF repo id pattern: "Qwen/<name>"
    # Map "<name>" to local folder "models/qwen3_tts/<name>"
    if s.startswith("Qwen/") and "/" in s:
        repo_name = s.split("/", 1)[1].strip()
        local_root = os.path.join(folder_paths.models_dir, "qwen3_tts")
        local_dir = os.path.join(local_root, repo_name)
        if os.path.isdir(local_dir):
            return local_dir

    return s
  

# CustomVoice models support a fixed set of premium speakers (per model README).
QWEN3_TTS_CUSTOM_VOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

def _try_warmup(model):
    """
    Run a minimal warm-up inference to trigger:
    - CUDA context init
    - attention backend selection
    - kernel caching
    This should be fast and safe. Any error is swallowed.
    """
    try:
        # Best-effort: detect model type if exposed (avoid calling unsupported modes)
        model_type = getattr(model, "tts_model_type", None)
        if model_type is None:
            # some implementations store it deeper
            try:
                model_type = getattr(getattr(model, "model", None), "tts_model_type", None)
            except Exception:
                model_type = None

        # VoiceDesign models
        if model_type == "voice_design" and hasattr(model, "generate_voice_design"):
            model.generate_voice_design(
                text="你好",
                instruct="中性自然语音",
                language="Auto",
                do_sample=False,
                max_new_tokens=16,
            )
            return

        # CustomVoice models (no ref_audio needed)
        if model_type == "custom_voice" and hasattr(model, "generate_custom_voice"):
            # If speaker list is available later, we can pick the first one.
            # For warmup, keep it minimal and let the model decide defaults if supported.
            model.generate_custom_voice(
                text="你好",
                speaker="Vivian",
                language="Auto",
                do_sample=False,
                max_new_tokens=16,
            )
            return

        # Base models: do not call voice_design/custom_voice/clone warmup.
        # If there's a generic generate API, use it; otherwise skip silently.
        if hasattr(model, "generate"):
            model.generate(
                text="你好",
                language="Auto",
                do_sample=False,
                max_new_tokens=16,
            )
            return
    except Exception as e:
        # Keep it short; warmup is optional
        print(f"[JR_Qwen3TTS] warmup skipped: {type(e).__name__}: {e}")

def _default_voice_preset_dir() -> str:
    """
    Default directory for storing voice preset files.
    """
    try:
        root = os.path.join(folder_paths.models_dir, "qwen3_tts", "voice_presets")
    except Exception:
        root = os.path.join("models", "qwen3_tts", "voice_presets")
    os.makedirs(root, exist_ok=True)
    return root


def _list_voice_presets() -> List[str]:
    """
    Return preset filenames (without extension) under the default preset dir.
    """
    d = _default_voice_preset_dir()
    out: List[str] = []
    for fn in os.listdir(d):
        if fn.lower().endswith((".pt", ".pth")):
            out.append(os.path.splitext(fn)[0])
    out.sort()
    # Dropdown must not be empty, provide a placeholder
    return out if out else ["(none)"]


def _resolve_preset_path(preset_name: str, output_dir: str = "") -> str:
    """
    Resolve a preset name to a full path.
    - If preset_name looks like a path, keep it.
    - Otherwise map to <output_dir or default_dir>/<preset_name>.pt
    """
    name = (preset_name or "").strip()
    if not name:
        raise ValueError("preset_name is empty")

    # If user typed a path
    if os.path.isabs(name) or os.path.exists(name):
        return name

    out_dir = (output_dir or "").strip() or _default_voice_preset_dir()
    os.makedirs(out_dir, exist_ok=True)
    if not name.lower().endswith((".pt", ".pth")):
        name = name + ".pt"
    return os.path.join(out_dir, name)


def _voice_prompt_to_safe_payload(prompt_items: Any) -> Dict[str, Any]:
    """
    Convert List[VoiceClonePromptItem] to a safe payload (dict/list + tensors),
    compatible with torch.load(weights_only=True) defaults in PyTorch 2.6+.
    """
    safe_items: List[Dict[str, Any]] = []
    for it in (prompt_items or []):
        safe_items.append({
            "ref_code": getattr(it, "ref_code", None),
            "ref_spk_embedding": getattr(it, "ref_spk_embedding"),
            "x_vector_only_mode": bool(getattr(it, "x_vector_only_mode", True)),
            "icl_mode": bool(getattr(it, "icl_mode", False)),
            "ref_text": getattr(it, "ref_text", None),
        })
    return {"format": "qwen3tts_voice_prompt_v1", "items": safe_items}


def _safe_payload_to_voice_prompt(payload: Dict[str, Any]) -> Any:
    """
    Reconstruct List[VoiceClonePromptItem] from safe payload.
    """
    if not isinstance(payload, dict) or payload.get("format") != "qwen3tts_voice_prompt_v1":
        raise ValueError(
            "Unsupported voice preset file format. "
            "Please re-save the preset using the updated Voice Preset node."
        )
    items = payload.get("items", [])
    from .qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    prompt_items = []
    for d in items:
        prompt_items.append(
            VoiceClonePromptItem(
                ref_code=d.get("ref_code", None),
                ref_spk_embedding=d["ref_spk_embedding"],
                x_vector_only_mode=bool(d.get("x_vector_only_mode", True)),
                icl_mode=bool(d.get("icl_mode", False)),
                ref_text=d.get("ref_text", None),
            )
        )
    return prompt_items

def _np_to_audio(np_wav: np.ndarray, sr: int) -> Dict[str, Any]:
    wav = np.asarray(np_wav, dtype=np.float32)
    # Accept common shapes and normalize to ComfyUI AUDIO: [B, C, N]
    if wav.ndim == 1:
        # [N] -> [1, 1, N]
        wav = wav[None, None, :]
    elif wav.ndim == 2:
        # [C, N] or [N, C] -> [1, C, N]
        if wav.shape[0] > wav.shape[1]:
            # likely [N, C]
            wav = wav.T
        wav = wav[None, :, :]
    elif wav.ndim == 3:
        # already batched; try to ensure [B, C, N]
        # if looks like [B, N, C], transpose last two dims
        if wav.shape[1] > wav.shape[2] and wav.shape[2] <= 8:
            wav = np.transpose(wav, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported wav shape: {wav.shape}")
    return {"waveform": torch.from_numpy(wav), "sample_rate": int(sr)}


def _audio_to_np(audio: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    sr = int(audio.get("sample_rate", 16000))
    wf = audio.get("waveform", None)
    if wf is None:
        raise ValueError("audio['waveform'] is missing")

    if isinstance(wf, torch.Tensor):
        wf = wf.detach().cpu().float().numpy()

    wf = np.asarray(wf, dtype=np.float32)
    # Normalize common ComfyUI AUDIO shapes to mono (N,)
    # Expected official shape: [B, C, N]
    if wf.ndim == 3:
        # [B, C, N] or [B, N, C]
        if wf.shape[1] > wf.shape[2] and wf.shape[2] <= 8:
            # [B, N, C] -> [B, C, N]
            wf = np.transpose(wf, (0, 2, 1))
        wf0 = wf[0]  # first batch: [C, N]
        wav = wf0.mean(axis=0) if wf0.shape[0] > 1 else wf0[0]
    elif wf.ndim == 2:
        # [C, N] or [N, C]
        if wf.shape[0] <= 8 and wf.shape[1] > wf.shape[0]:
            wav = wf.mean(axis=0) if wf.shape[0] > 1 else wf[0]
        else:
            wav = wf.mean(axis=1) if wf.shape[1] > 1 else wf[:, 0]
    elif wf.ndim == 1:
        wav = wf
    else:
        raise ValueError(f"Unsupported waveform ndim: {wf.ndim}")

    return wav.astype(np.float32), sr


def _default_voice_prompt_dir() -> str:
    """Default directory for storing pre-extracted voice prompt data."""
    try:
        root = os.path.join(folder_paths.models_dir, "qwen3_tts", "voice_prompts")
    except Exception:
        root = os.path.join("models", "qwen3_tts", "voice_prompts")
    os.makedirs(root, exist_ok=True)
    return root


def _resolve_voice_prompt_path(p: str) -> str:
    """Resolve a user-provided path. If relative, place it under the default voice prompt dir."""
    s = (p or "").strip()
    if not s:
        raise ValueError("voice_prompt_path is empty")
    # If already absolute or exists relative to cwd, keep as-is
    if os.path.isabs(s) or os.path.exists(s):
        return s
    return os.path.join(_default_voice_prompt_dir(), s)


def _voice_prompt_to_safe_payload(prompt_items: Any) -> Dict[str, Any]:
    """
    Convert List[VoiceClonePromptItem] to a safe, weights_only-loadable payload.
    Only uses primitive types + Tensors.
    """
    safe_items: List[Dict[str, Any]] = []
    for it in (prompt_items or []):
        safe_items.append({
            "ref_code": getattr(it, "ref_code", None),
            "ref_spk_embedding": getattr(it, "ref_spk_embedding"),
            "x_vector_only_mode": bool(getattr(it, "x_vector_only_mode", True)),
            "icl_mode": bool(getattr(it, "icl_mode", False)),
            "ref_text": getattr(it, "ref_text", None),
        })
    return {"format": "qwen3tts_voice_prompt_v1", "items": safe_items}


class JR_Qwen3TTS_VoicePreset:
    """
    Unified Voice Preset node:
    - Load: pick a preset from dropdown and output ref_voice_data.
    - Save/Update: provide ref_audio (+ optional ref_text) and it will extract prompt and overwrite preset file.

    This removes the need for separate Extract+Save and Load nodes, and behaves like a model preset dropdown.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["load", "save_or_update"], {"default": "load"}),
                # Dropdown list from disk
                "preset": (_list_voice_presets(),),
                # For creating new presets via typing a name
                "preset_name_override": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                # These govern extraction when saving
                "x_vector_only_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": ("QWEN3_TTS_MODEL",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = (QWEN3_TTS_VOICE_PROMPT, "STRING")
    RETURN_NAMES = ("ref_voice_data", "path")
    FUNCTION = "run"
    CATEGORY = "JR/Audio/TTS"

    def run(
        self,
        action: str,
        preset: str,
        preset_name_override: str,
        output_dir: str,
        x_vector_only_mode: bool,
        model=None,
        ref_audio: Optional[Dict[str, Any]] = None,
        ref_text: str = "",
    ):
        name = (preset_name_override or "").strip() or (preset or "").strip()
        if not name or name == "(none)":
            raise ValueError("No preset selected. Choose an existing preset, or type a new name in preset_name_override.")

        path = _resolve_preset_path(name, output_dir=output_dir)

        if action == "save_or_update":
            if model is None:
                raise ValueError("action=save_or_update requires input 'model'.")
            if ref_audio is None:
                raise ValueError("action=save_or_update requires input 'ref_audio' (AUDIO).")

            wav_np, wav_sr = _audio_to_np(ref_audio)
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=(wav_np, wav_sr),
                ref_text=(ref_text if ref_text else None),
                x_vector_only_mode=bool(x_vector_only_mode),
            )

            payload = _voice_prompt_to_safe_payload(prompt_items)
            torch.save(payload, path)
            return (prompt_items, path)

        # load
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Voice preset not found: {path}")
        payload = torch.load(path, map_location="cpu")
        prompt_items = _safe_payload_to_voice_prompt(payload)
        return (prompt_items, path)


class JR_Qwen3TTS_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_preset": (QWEN3_TTS_MODEL_PRESETS, {"default": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}),
                # Optional manual override (HF repo id or local path). If non-empty, it wins.
                "model_id_or_path_override": ("STRING", {"default": ""}),
                "device_map": ("STRING", {"default": "cuda:0"}),  # e.g. cuda:0 / cpu
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "attn_impl": (["sdpa", "eager", "flash_attention_2"], {"default": "sdpa"}),
                "warmup": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "JR/Audio/TTS"


    def load(self, model_preset: str, model_id_or_path_override: str, device_map: str, dtype: str, attn_impl: str, warmup: bool):
        raw = (model_id_or_path_override or "").strip() or str(model_preset)
        model_id_or_path = _resolve_qwen_model_id_or_local_path(raw)
        model, _key = load_qwen3_tts(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            dtype=dtype,
            attn_impl=attn_impl,
        )

        if warmup:
            _try_warmup(model)

        return (model,)


class JR_Qwen3TTS_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "mode": (["voice_clone", "voice_design", "custom_voice"], {"default": "voice_clone"}),
                "text": ("STRING", {"multiline": True, "default": "你好，欢迎使用 Qwen3-TTS。"}),
                "language": ("STRING", {"default": "Auto"}),

                "do_sample": ("BOOLEAN", {"default": True}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.8, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
            },
            "optional": {
                "ref_voice_data": (QWEN3_TTS_VOICE_PROMPT,),
                # voice_clone
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
                "x_vector_only_mode": ("BOOLEAN", {"default": False}),

                # voice_design / custom_voice
                "instruct": ("STRING", {"multiline": True, "default": ""}),

                # custom_voice
                "speaker": (QWEN3_TTS_CUSTOM_VOICE_SPEAKERS, {"default": "Vivian"}),
                # Optional manual override in case future models add more speakers
                "speaker_override": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "JR/Audio/TTS"

    def generate(
        self,
        model,
        mode: str,
        text: str,
        language: str,
        do_sample: bool,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        ref_audio: Optional[Dict[str, Any]] = None,
        ref_voice_data: Optional[Any] = None,
        ref_text: str = "",
        x_vector_only_mode: bool = False,
        instruct: str = "",
        speaker: str = "",
        speaker_override: str = "",
    ):
        gen_kwargs = dict(
            do_sample=bool(do_sample),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=int(max_new_tokens),
        )

        lang = language if language else "Auto"

        if mode == "voice_clone":
            # Preferred path: pre-extracted voice prompt data
            if ref_voice_data is not None:
                print("[JR_Qwen3TTS] Using ref_voice_data (voice preset). ref_audio/ref_text/x_vector_only_mode are ignored.")
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    voice_clone_prompt=ref_voice_data,
                    **gen_kwargs,
                )
            else:
                if ref_audio is None:
                    raise ValueError("mode=voice_clone requires either 'ref_voice_data' or optional input 'ref_audio' (AUDIO).")
                wav_np, wav_sr = _audio_to_np(ref_audio)
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=(wav_np, wav_sr),
                    ref_text=ref_text if ref_text else None,
                    x_vector_only_mode=bool(x_vector_only_mode),
                    **gen_kwargs,
                )

        elif mode == "voice_design":
            wavs, sr = model.generate_voice_design(
                text=text,
                instruct=instruct if instruct else "",
                language=lang,
                **gen_kwargs,
            )

        elif mode == "custom_voice":
            spk = (speaker_override or "").strip() or str(speaker).strip()
            if not spk:
                # Should not happen due to dropdown default, but keep safety.
                spk = "Vivian"
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=spk,
                language=lang,
                instruct=instruct if instruct else None,
                **gen_kwargs,
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return (_np_to_audio(wavs[0], sr),)


class JR_Qwen3TTS_MultiTalkGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        # Up to 10 speakers. Provide matching speaker_name_i + ref_voice_data_i.
        optional = {
            "gap_ms": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 10}),
        }
        for i in range(1, 11):
            optional[f"speaker_{i}_name"] = ("STRING", {"default": ""})
            optional[f"ref_voice_data_{i}"] = (QWEN3_TTS_VOICE_PROMPT,)

        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "[旁白]:你好，欢迎使用多角色有声小说生成。"}),
                "language": ("STRING", {"default": "Auto"}),

                "do_sample": ("BOOLEAN", {"default": True}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.8, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "non_streaming_mode": ("BOOLEAN", {"default": True}),
                "cleanup_each_sentence": ("BOOLEAN", {"default": True}),
                "cleanup_every_n": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "do_cuda_synchronize_before_cleanup": ("BOOLEAN", {"default": False}),
"merge_output": ("BOOLEAN", {"default": True}),
                "unload_model_after": ("BOOLEAN", {"default": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "JR/Audio/TTS"

    _TAG_RE = re.compile(r"\[([^\]]+)\]\s*:\s*")

    def _build_speaker_map(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Any], str]:
        speaker_map: Dict[str, Any] = {}
        default_prompt = None
        default_name = ""
        for i in range(1, 11):
            name = (kwargs.get(f"speaker_{i}_name") or "").strip()
            prompt = kwargs.get(f"ref_voice_data_{i}")
            if prompt is None:
                continue
            if not name:
                name = f"speaker_{i}"
            if default_prompt is None:
                default_prompt = prompt
                default_name = name
            speaker_map[name] = prompt

        return speaker_map, default_prompt, default_name

    def _parse_dialogue(self, raw: str) -> List[Tuple[str, str]]:
        s = (raw or "").strip()
        if not s:
            return []
        matches = list(self._TAG_RE.finditer(s))
        if not matches:
            # No tags found: treat entire input as narrator text
            return [("旁白", s)]

        segs: List[Tuple[str, str]] = []
        for idx, m in enumerate(matches):
            name = (m.group(1) or "").strip()
            start = m.end()
            end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(s)
            content = s[start:end].strip()
            if not content:
                continue
            segs.append((name, content))
        return segs

    def _concat_with_gaps(self, parts: List[np.ndarray], sr: int, gap_ms: int) -> np.ndarray:
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        gap = int(sr * max(0, int(gap_ms)) / 1000)
        if gap <= 0:
            return np.concatenate(parts, axis=0).astype(np.float32)
        sil = np.zeros((gap,), dtype=np.float32)
        out = []
        for i, p in enumerate(parts):
            out.append(np.asarray(p, dtype=np.float32).reshape(-1))
            if i != len(parts) - 1:
                out.append(sil)
        return np.concatenate(out, axis=0).astype(np.float32)

    def generate(
        self,
        model,
        text: str,
        language: str,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        non_streaming_mode: bool,
        cleanup_each_sentence: bool,
        cleanup_every_n: int,
        do_cuda_synchronize_before_cleanup: bool,
        merge_output: bool,
        unload_model_after: bool,
        **kwargs,
    ):
        gen_kwargs = dict(
            do_sample=bool(do_sample),
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=int(max_new_tokens),
        )
        lang = language if language else "Auto"
        gap_ms = int(kwargs.get("gap_ms", 200))

        speaker_map, default_prompt, default_name = self._build_speaker_map(kwargs)
        if default_prompt is None:
            raise ValueError("Multi-talk requires at least one ref_voice_data input (ref_voice_data_1..ref_voice_data_10).")

        segments = self._parse_dialogue(text)
        if not segments:
            raise ValueError("No dialogue segments found in text. Expected format: [Speaker]:text ...")

        # Prefer explicit narrator voice if provided
        narrator_prompt = speaker_map.get("旁白") or speaker_map.get("[旁白]") or default_prompt

        wav_parts: List[np.ndarray] = []
        out_sr: Optional[int] = None

        for spk, utt in segments:
            print(f"开始处理:'{spk}'.")
            prompt = speaker_map.get(spk)
            if prompt is None:
                # Fallback to narrator, but do not break output
                print(f"[JR_Qwen3TTS] [MultiTalk] Speaker not found: '{spk}'. Using narrator/default voice instead.")
                prompt = narrator_prompt

            with torch.inference_mode():
                # Avoid the pseudo-streaming path by default. Only pass if supported.
                extra = {}
                try:
                    import inspect
                    if "non_streaming_mode" in inspect.signature(model.generate_voice_clone).parameters:
                        extra["non_streaming_mode"] = bool(non_streaming_mode)
                except Exception:
                    pass

                wavs, sr = model.generate_voice_clone(
                    text=utt,
                    language=lang,
                    voice_clone_prompt=prompt,
                    **gen_kwargs,
                    **extra,
                )
            # Optional per-sentence cleanup to reduce VRAM fragmentation / cache thrash.
            if cleanup_each_sentence and torch.cuda.is_available():
                try:
                    if do_cuda_synchronize_before_cleanup:
                        torch.cuda.synchronize()
                    if (len(wav_parts) + 1) % max(1, int(cleanup_every_n)) == 0:
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                except Exception:
                    pass
            print(f"清理缓存完毕:'{spk}'.")
            w = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            try:
                del wavs
            except Exception:
                pass
            if out_sr is None:
                out_sr = int(sr)
            elif int(sr) != int(out_sr):
                raise ValueError(f"Sample rate mismatch in multi-talk generation: got {sr} but expected {out_sr}.")
            wav_parts.append(w)

        if out_sr is None:
            out_sr = 16000

        final_wav = self._concat_with_gaps(wav_parts, out_sr, gap_ms=gap_ms)

        # Output control:
        # - merge_output=True  -> return a single merged AUDIO
        # - merge_output=False -> return a list[AUDIO], one per segment (ComfyUI will map downstream nodes)
        if merge_output:
            out_audio = _np_to_audio(final_wav, out_sr)
        else:
            out_audio = [_np_to_audio(w, out_sr) for w in wav_parts]

        # Optional: unload model and clear caches after generation
        if unload_model_after:
            try:
                if hasattr(model, "unload") and callable(getattr(model, "unload")):
                    model.unload()
                else:
                    clear_cache()
            except Exception as e:
                print(f"[JR_Qwen3TTS] [MultiTalk] unload_model_after failed: {type(e).__name__}: {e}")
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return (out_audio,)

# ---------------------------
# Voice prompt preprocess I/O
# ---------------------------

class JR_Qwen3TTS_ExtractAndSaveVoicePrompt:
    """
    Extract a reusable "voice prompt" from reference audio and persist it to disk.
    The saved artifact can later be loaded and fed into JR_Qwen3TTS_Generate via `ref_voice_data`,
    eliminating the need to re-enter long `ref_text` or repeatedly run extraction.

    Notes:
    - For most users, set `x_vector_only_mode=True` to avoid providing `ref_text`.
      This uses only the speaker embedding (no ICL).
    - If `x_vector_only_mode=False`, `ref_text` becomes mandatory (ICL mode).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "ref_audio": ("AUDIO",),
                "filename": ("STRING", {"default": "voice_prompt_001.pt"}),
                "output_dir": ("STRING", {"default": ""}),
                "x_vector_only_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("QWEN3_TTS_VOICE_PROMPT", "STRING")
    RETURN_NAMES = ("voice_prompt", "path")
    FUNCTION = "extract_and_save"
    CATEGORY = "JR/Audio/TTS"

    def extract_and_save(
        self,
        model,
        ref_audio: Dict[str, Any],
        filename: str,
        output_dir: str,
        x_vector_only_mode: bool,
        ref_text: str = "",
    ):
        wav_np, wav_sr = _audio_to_np(ref_audio)

        # create_voice_clone_prompt will enforce ref_text when x_vector_only_mode=False
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(wav_np, wav_sr),
            ref_text=(ref_text if ref_text else None),
            x_vector_only_mode=bool(x_vector_only_mode),
        )

        out_dir = (output_dir or "").strip() or _default_voice_prompt_dir()
        os.makedirs(out_dir, exist_ok=True)

        name = (filename or "").strip() or "voice_prompt.pt"
        if not name.lower().endswith((".pt", ".pth")):
            name = name + ".pt"
        save_path = os.path.join(out_dir, name)

        # PyTorch 2.6+ defaults torch.load(weights_only=True) which blocks pickled custom classes.
        # Save as a safe payload (dict/list + tensors) to avoid allowlisting / weights_only=False.
        payload = _voice_prompt_to_safe_payload(prompt_items)
        torch.save(payload, save_path)

        return (prompt_items, save_path)


class JR_Qwen3TTS_LoadVoicePrompt:
    """Load a previously saved voice prompt (.pt/.pth) and output it as QWEN3_TTS_VOICE_PROMPT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_prompt_path": ("STRING", {"default": "voice_prompt_001.pt"}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_VOICE_PROMPT",)
    RETURN_NAMES = ("voice_prompt",)
    FUNCTION = "load"
    CATEGORY = "JR/Audio/TTS"

    def load(self, voice_prompt_path: str):
        p = _resolve_voice_prompt_path(voice_prompt_path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Voice prompt not found: {p}")
        obj = torch.load(p, map_location="cpu")

        # New safe format
        if isinstance(obj, dict) and obj.get("format") == "qwen3tts_voice_prompt_v1":
            items = obj.get("items", [])
            # Reconstruct VoiceClonePromptItem objects expected by model.generate_voice_clone(voice_clone_prompt=...)
            from .qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

            prompt_items = []
            for d in items:
                prompt_items.append(
                    VoiceClonePromptItem(
                        ref_code=d.get("ref_code", None),
                        ref_spk_embedding=d["ref_spk_embedding"],
                        x_vector_only_mode=bool(d.get("x_vector_only_mode", True)),
                        icl_mode=bool(d.get("icl_mode", False)),
                        ref_text=d.get("ref_text", None),
                    )
                )
            return (prompt_items,)

        # Old (pickled) format blocked by weights_only in PyTorch 2.6+; force user to re-extract.
        raise ValueError(
            "Unsupported voice prompt file format. "
            "This file was likely saved in the old pickled-object format and is blocked by PyTorch weights_only. "
            "Please re-run 'Extract+Save Voice Prompt' to regenerate the file with the safe format."
        )


class JR_Qwen3TTS_SaveWav:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "output_dir": ("STRING", {"default": "output"}),
                "filename_prefix": ("STRING", {"default": "JR_qwen3_tts"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "JR/Audio/TTS"
    OUTPUT_NODE = True

    def save(self, audio: Dict[str, Any], output_dir: str, filename_prefix: str):
        wf, sr = _audio_to_np(audio)
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"{filename_prefix}_{ts}.wav")
        sf.write(path, wf, sr)
        return (path,)


class JR_Qwen3TTS_ClearCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("cleared_models",)
    FUNCTION = "clear"
    CATEGORY = "JR/Audio/TTS"

    def clear(self):
        return (clear_cache(),)


NODE_CLASS_MAPPINGS = {
    "JR_Qwen3TTS_VoicePreset": JR_Qwen3TTS_VoicePreset,
    "JR_Qwen3TTS_Loader": JR_Qwen3TTS_Loader,
    "JR_Qwen3TTS_Generate": JR_Qwen3TTS_Generate,
    "JR_Qwen3TTS_MultiTalkGenerate": JR_Qwen3TTS_MultiTalkGenerate,
    "JR_Qwen3TTS_ExtractAndSaveVoicePrompt": JR_Qwen3TTS_ExtractAndSaveVoicePrompt,
    "JR_Qwen3TTS_LoadVoicePrompt": JR_Qwen3TTS_LoadVoicePrompt,
    "JR_Qwen3TTS_SaveWav": JR_Qwen3TTS_SaveWav,
    "JR_Qwen3TTS_ClearCache": JR_Qwen3TTS_ClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JR_Qwen3TTS_VoicePreset": "JR Qwen3 TTS Voice Preset",
    "JR_Qwen3TTS_Loader": "JR Qwen3 TTS Loader",
    "JR_Qwen3TTS_Generate": "JR Qwen3 TTS Generate",
    "JR_Qwen3TTS_MultiTalkGenerate": "JR Qwen3 TTS multi-talk Generate",
    "JR_Qwen3TTS_ExtractAndSaveVoicePrompt": "JR Qwen3 TTS Extract+Save Voice Prompt",
    "JR_Qwen3TTS_LoadVoicePrompt": "JR Qwen3 TTS Load Voice Prompt",
    "JR_Qwen3TTS_SaveWav": "JR Qwen3 TTS Save WAV",
    "JR_Qwen3TTS_ClearCache": "JR Qwen3 TTS Clear Cache",
}
