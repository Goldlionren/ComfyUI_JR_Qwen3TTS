from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple, Optional

import numpy as np
import soundfile as sf
import torch

from .qwen_wrapper import load_qwen3_tts, clear_cache
import folder_paths

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
            if ref_audio is None:
                raise ValueError("mode=voice_clone requires optional input 'ref_audio' (AUDIO).")
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
    "JR_Qwen3TTS_Loader": JR_Qwen3TTS_Loader,
    "JR_Qwen3TTS_Generate": JR_Qwen3TTS_Generate,
    "JR_Qwen3TTS_SaveWav": JR_Qwen3TTS_SaveWav,
    "JR_Qwen3TTS_ClearCache": JR_Qwen3TTS_ClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JR_Qwen3TTS_Loader": "JR Qwen3 TTS Loader",
    "JR_Qwen3TTS_Generate": "JR Qwen3 TTS Generate",
    "JR_Qwen3TTS_SaveWav": "JR Qwen3 TTS Save WAV",
    "JR_Qwen3TTS_ClearCache": "JR Qwen3 TTS Clear Cache",
}
