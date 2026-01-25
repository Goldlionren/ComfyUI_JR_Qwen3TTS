---

````markdown
# ComfyUI_JR_Qwen3TTS

A ComfyUI custom node implementation for **Qwen3-TTS**, supporting **Voice Design**, **Voice Clone**, and **Custom Voice** generation modes.

This project focuses on **practical engineering integration** rather than model re-training, providing a **stable, high-performance, and reusable** TTS workflow inside ComfyUI.

---

## ✨ Features

- 🔊 Qwen3-TTS integration for ComfyUI
- 🎭 Voice Design (instruction-based voice generation)
- 🎙 Voice Clone (reference-audio-based speaker cloning)
- 🧑‍🎤 Custom Voice (official premium speakers)
- 🎚 Model loader with dropdown presets
- 🎛 **Voice Preset system** (extract once, reuse like a model)
- 🚀 Optional warmup for faster first inference
- 🛡 Safe prompt serialization (PyTorch 2.6+ compatible, no pickle)

---

## 📦 Supported Models

The following official Qwen3-TTS models are supported:

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

Model weights are **not included** in this repository.

---

## 📁 Model Path Resolution (IMPORTANT)

This plugin **automatically prefers local model paths** before downloading from Hugging Face.

### Recommended local directory layout

```text
ComfyUI/
└─ models/
   └─ qwen3_tts/
      ├─ Qwen3-TTS-12Hz-1.7B-Base/
      ├─ Qwen3-TTS-12Hz-1.7B-CustomVoice/
      ├─ Qwen3-TTS-12Hz-1.7B-VoiceDesign/
      ├─ Qwen3-TTS-12Hz-0.6B-Base/
      ├─ voice_presets/
      │  ├─ GXM.pt
      │  ├─ Vivian_ICL.pt
      │  └─ ...
````

### Resolution logic

When you select a model such as:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

The loader will first try:

```text
ComfyUI/models/qwen3_tts/Qwen3-TTS-12Hz-1.7B-Base
```

If the directory exists, it will be used **without any network download**.

---

## 🧠 Generation Modes

### 1️⃣ Voice Design

Generate a voice based on a natural language description.

Example:

> “A calm and warm female voice, suitable for narration.”

---

### 2️⃣ Voice Clone

Clone a voice using reference audio.

⚠️ **Important**

For `voice_clone`, `do_sample` **must be enabled**:

```text
do_sample = true
```

This is required by the model to avoid degenerate long audio generation.

---

### 3️⃣ Custom Voice

Use official built-in premium speakers.

Supported speakers:

* Vivian
* Serena
* Uncle_Fu
* Dylan
* Eric
* Ryan
* Aiden
* Ono_Anna
* Sohee

---

## 🎛 Voice Preset System (Recommended)

### What is a Voice Preset?

A **Voice Preset** is a **pre-extracted voice prompt** saved to disk and reused later.

It allows you to:

* Extract voice characteristics **once**
* Avoid re-entering long `ref_text`
* Avoid re-processing reference audio
* Ensure consistent and fast voice cloning

Voice Presets behave **like model presets**: select from a dropdown and use directly.

Preset files are stored in:

```text
ComfyUI/models/qwen3_tts/voice_presets/
```

---

## 🔁 Workflow A: Create / Update a Voice Preset (One-Time)

Use this workflow **only once per speaker**.

1. `Load Audio` → reference WAV / MP3 / FLAC
2. `JR Qwen3 TTS Loader`
3. `JR Qwen3 TTS Voice Preset`

   * `action = save_or_update`
   * `preset_name_override = GXM` (example)

### x-vector Only Mode (Fast, No ref_text)

```text
x_vector_only_mode = true
ref_text = (empty)
```

* Fastest extraction
* No reference text required
* Recommended for most users

### ICL Mode (Higher Fidelity)

```text
x_vector_only_mode = false
ref_text = (required, once only)
```

* Reference text is embedded into the preset
* Higher voice similarity
* Slightly slower extraction (one-time cost)

---

## ▶️ Workflow B: Use a Voice Preset (Daily Use)

1. `JR Qwen3 TTS Loader`
2. `JR Qwen3 TTS Voice Preset`

   * `action = load`
   * select preset from dropdown
3. `JR Qwen3 TTS Generate`

   * `mode = voice_clone`
   * connect `ref_voice_data`

### Important Behavior

When **`ref_voice_data` is connected**:

* ❗ `ref_audio` is ignored
* ❗ `ref_text` is ignored
* ❗ `x_vector_only_mode` is ignored

All voice behavior is **fully determined by the preset**.

This guarantees:

* Maximum performance
* Reproducible results
* No parameter mismatch

---

## 📂 Example Workflows

The `example/` directory contains:

* 📷 Workflow screenshots
* 📄 Step-by-step explanations

These examples demonstrate:

* Voice preset creation
* Preset-based voice cloning
* Correct node connections

---

## 🔐 PyTorch 2.6+ Compatibility (Security Note)

PyTorch 2.6 changed the default behavior of:

```python
torch.load(weights_only=True)
```

This project **does NOT rely on pickle-based objects** for voice prompts.

Instead:

* Voice presets are saved as **safe payloads** (dict + tensors only)
* No `weights_only=False`
* No `add_safe_globals`
* No security warnings

This ensures long-term compatibility and safe sharing of presets.

---

## 🔧 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI_JR_Qwen3TTS.git
```

Restart ComfyUI after installation.

---

## 📜 License

This project is released under the **MIT License**.

---

# 中文说明

## 项目简介

**ComfyUI_JR_Qwen3TTS** 是一个将 **Qwen3-TTS** 模型完整接入 ComfyUI 的自定义节点项目，支持多种语音生成模式，并引入了工程化的 **Voice Preset（语音预设）** 体系。

本项目 **不包含模型权重**，仅提供 ComfyUI 推理节点与封装逻辑。

---

## 核心特性

* Qwen3-TTS 的 ComfyUI 工程化集成
* 支持 Voice Design / Voice Clone / Custom Voice
* 模型与语音均支持下拉选择
* 语音预设一次提取，多次复用
* 兼容 PyTorch 2.6+ 的安全加载机制
* 保留 legacy 节点，方便后续扩展与二次开发

---

## Voice Preset 说明

Voice Preset 是将 **参考音频（及可选 ref_text）预处理并固化** 的语音配置文件。

使用 Voice Preset 后：

* 不再需要每次输入 ref_text
* 不再重复处理 reference audio
* 推理速度显著提升
* 行为完全可复现

推荐在所有 **voice_clone** 场景中使用。

---

## Voice Clone 注意事项

使用 **voice_clone** 模式时：

```text
do_sample = true
```

否则模型可能生成异常长的无效音频，这是模型本身的限制。

---

## License 与模型声明

* 本项目代码遵循 **MIT License**
* Qwen3-TTS 模型及权重遵循官方 License
* 商业或分发前请自行确认模型授权条款

---

## 致谢

* Qwen Team for Qwen3-TTS
* ComfyUI community

```
