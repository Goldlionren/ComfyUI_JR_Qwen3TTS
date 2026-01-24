## README.md（完整版）

````markdown
# ComfyUI_JR_Qwen3TTS

A ComfyUI custom node implementation for **Qwen3-TTS**, supporting **Voice Design**, **Voice Clone**, and **Custom Voice** generation modes.

This project focuses on **practical engineering integration** rather than model re-training, providing a stable and user-friendly TTS workflow inside ComfyUI.

---

## ✨ Features

- 🔊 Qwen3-TTS integration for ComfyUI
- 🎭 Voice Design (instruction-based voice generation)
- 🎙 Voice Clone (reference-audio-based speaker cloning)
- 🧑‍🎤 Custom Voice (official premium speakers)
- ⚙️ Model loader with dropdown presets
- 🚀 Optional warmup for faster first inference
- 🛡 Safe parameter handling (prevents degenerate long audio)

---

## 📦 Supported Models

The following official Qwen3-TTS models are supported:

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

Model weights are **not included** in this repository and will be downloaded automatically from Hugging Face.

---

## 🧠 Generation Modes

### 1️⃣ Voice Design
Generate a voice based on a natural language description.

Example:
> “A calm and warm female voice, suitable for narration.”

### 2️⃣ Voice Clone
Clone a voice using reference audio.

⚠️ **Important**  
For `voice_clone`, `do_sample` **must be enabled** to avoid degenerate long audio generation.

### 3️⃣ Custom Voice
Use official built-in premium speakers.

Supported speakers:
- Vivian
- Serena
- Uncle_Fu
- Dylan
- Eric
- Ryan
- Aiden
- Ono_Anna
- Sohee

---

## 🔧 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI_JR_Qwen3TTS.git
````

Restart ComfyUI after installation.

---

## ⚠️ Notes on License & Models

* This repository contains **only integration code**.
* Qwen3-TTS model weights are subject to the original license provided by Qwen.
* Please refer to the official Qwen3-TTS repository for model usage terms.

---

## 📜 License

This project is released under the **MIT License**.

---

# 中文说明

## 项目简介

**ComfyUI_JR_Qwen3TTS** 是一个将 **Qwen3-TTS** 语音合成模型完整接入 ComfyUI 的自定义节点项目，支持多种语音生成模式，专注于工程可用性与稳定性。

本项目 **不包含模型权重**，仅提供 ComfyUI 节点与推理封装。

---

## 功能特性

* Qwen3-TTS 的 ComfyUI 工程化集成
* 支持 Voice Design / Voice Clone / Custom Voice
* 模型选择下拉菜单
* 可选 warmup，减少首次推理卡顿
* 针对 voice_clone 的安全参数处理，避免异常长音频

---

## 使用说明

### Voice Clone 注意事项

在使用 **voice_clone** 模式时，必须开启 `do_sample`：

```text
do_sample = true
```

这是模型本身的特性要求，否则可能生成数分钟的无效音频。

---

## License 与模型声明

* 本项目代码使用 **MIT License**
* Qwen3-TTS 模型及其权重遵循官方 License
* 请在商业或分发前自行确认模型授权条款

---

## 致谢

* Qwen Team for the Qwen3-TTS model
* ComfyUI community

````
