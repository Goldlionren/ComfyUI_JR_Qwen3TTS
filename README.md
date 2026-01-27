# ComfyUI_JR_Qwen3TTS

A ComfyUI custom node implementation for **Qwen3-TTS**, supporting **Voice Design**, **Voice Clone**, **Custom Voice**, and **Multi-Speaker Audiobook / Dialogue generation**.

This project focuses on **practical engineering integration** rather than model re-training, providing a **stable, high-performance, and reusable** TTS workflow inside ComfyUI — especially for **long-form audio and multi-character narration**.

---

## ✨ Features

* 🔊 Qwen3-TTS integration for ComfyUI
* 🎭 Voice Design (instruction-based voice generation)
* 🎙 Voice Clone (reference-audio-based speaker cloning)
* 🧑‍🎤 Custom Voice (official premium speakers)
* 🎚 Model loader with dropdown presets
* 🎛 **Voice Preset system** (extract once, reuse like a model)
* 🗣 **Multi-Speaker / Multi-Role TTS (Audiobook / Dialogue)**
* 🧠 **Voice library–based design (not per-prompt cloning)**
* 🚀 Optional warmup for faster first inference
* 🧹 **Engineered cache & memory cleanup for long audio**
* 🛡 Safe prompt serialization (PyTorch 2.6+ compatible, no pickle)

---

## 📦 Supported Models

The following official Qwen3-TTS models are supported:

* `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
* `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
* `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
* `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
* `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

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
```

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

Clone a voice using reference audio or a **Voice Preset**.

⚠️ **Important**

For `voice_clone`, `do_sample` **must be enabled**:

```text
do_sample = true
```

This is required by the model to avoid degenerate audio generation.

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

## 🎛 Voice Preset System (Core Design)

### What is a Voice Preset?

A **Voice Preset** is a **pre-extracted voice representation** stored on disk and reused later.

It represents a **speaker’s timbre library**, not a temporary prompt.

With Voice Presets you can:

* Extract voice characteristics **once**
* Avoid re-processing reference audio
* Avoid re-entering long `ref_text`
* Guarantee **consistent speaker identity**
* Dramatically improve performance in long or repeated generations

Voice Presets behave **like model presets**: select from a dropdown and reuse directly.

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

### x-vector Only Mode (Fast, Recommended)

```text
x_vector_only_mode = true
ref_text = (empty)
```

* Fastest extraction
* No reference text required
* Recommended for most users

### ICL Mode (Higher Fidelity, One-Time Cost)

```text
x_vector_only_mode = false
ref_text = (required, once only)
```

* Reference text embedded into the preset
* Higher voice similarity
* Slightly slower extraction (one-time)

---

## ▶️ Workflow B: Single-Speaker Generation (Daily Use)

1. `JR Qwen3 TTS Loader`
2. `JR Qwen3 TTS Voice Preset`

   * `action = load`
3. `JR Qwen3 TTS Generate`

   * `mode = voice_clone`
   * connect `ref_voice_data`

### Important Behavior

When **`ref_voice_data` is connected**:

* `ref_audio` is ignored
* `ref_text` is ignored
* `x_vector_only_mode` is ignored

All voice behavior is **fully determined by the preset**.

---

## 🗣 Multi-Speaker / Multi-Role TTS (Audiobook / Dialogue)

### Overview

This project provides a dedicated node:

**`JR Qwen3 TTS Multi-Talk Generate`**

Designed for:

* Audiobooks
* Radio dramas
* Visual novels
* Multi-character narration
* Long-form dialogue

### Design Philosophy

Unlike prompt-based speaker switching, this implementation is:

* ✅ **Voice library–driven** (each role maps to a Voice Preset)
* ✅ Stable for **long text & many sentences**
* ✅ Optimized for **GPU memory reuse & cleanup**
* ✅ Suitable for production-scale narration

---

### Text Format

Each sentence starts with a speaker tag:

```text
[旁白]: 夜色渐深，城市陷入沉睡。
[Tom 01]: Are you still awake?
[Alice]: 是的，我在等你。
```

Speaker names support:

* Chinese
* English
* Numbers
* Spaces

---

### Node Inputs

* Up to **10 speakers**
* Each speaker:

  * `speaker_name`
  * `ref_voice_data` (Voice Preset output)

### Output Modes

* **Merged output**: one complete audio with configurable gaps
* **Per-sentence output**: one audio per sentence (for post-processing)

---

### Engine-Level Optimizations

To support **long audio generation**, the Multi-Talk node includes:

* Sentence-level inference isolation
* Explicit GPU cache cleanup
* Optional per-sentence memory release
* Safe non-streaming inference path
* Designed to avoid audio degradation in long runs

This allows stable generation of **long dialogues and audiobooks** without the common “audio collapse” issues.

---

## 📂 Example Workflows

The `example/` directory contains:

* 📷 Workflow screenshots
* 📄 Step-by-step explanations

Including:

* Voice preset creation
* Multi-speaker dialogue generation
* Recommended parameter settings

---

## 🔐 PyTorch 2.6+ Compatibility (Security Note)

PyTorch 2.6 changed the default behavior of:

```python
torch.load(weights_only=True)
```

This project **does NOT rely on pickle-based objects**.

* Voice presets are saved as **safe tensor payloads**
* No `weights_only=False`
* No unsafe globals
* Safe for sharing & long-term use

---

## 🔧 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI_JR_Qwen3TTS.git
```

Restart ComfyUI after installation.

---

## 📜 License

* Code: **MIT License**
* Models: subject to Qwen official licenses

---

# 中文说明（简要）

## 项目定位

**ComfyUI_JR_Qwen3TTS** 是一个以 **工程稳定性与可复用性** 为核心目标的 Qwen3-TTS ComfyUI 插件。

其核心思想是：

> **先建立人声音色库（Voice Preset），再基于音色库进行生成**

而不是在每次生成中临时拼接 prompt。

---

## 多角色有声小说能力

* 基于 Voice Preset 的多角色系统
* 一个角色 = 一个稳定音色
* 支持长文本、多角色连续生成
* 内置缓存与显存清理优化
* 适合有声书、广播剧、剧情配音等场景

---

## 致谢

* Qwen Team for Qwen3-TTS
* ComfyUI Community


