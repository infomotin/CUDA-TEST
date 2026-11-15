# 🧠 AI Models & LLaMA.cpp GGUF Architecture Hub

This directory contains the AI models, fine-tuned weights, and LLaMA.cpp compatible GGUF binaries for the **CUDA Multimodal GPGPU & AI Synthesis Studio**.

## 📁 Directory Structure
```
models/
├── finetuned_model.gguf    # Exported LLaMA.cpp compatible GGUF model
├── huggingface/            # Local Hugging Face cache (F: Drive inside app, not in C:)
│   └── hub/                # Downloaded model shards & safetensors
├── torch/                  # PyTorch model weights cache
└── README.md               # Model specifications and usage instructions
```

## 🚀 Supported Models & Architectures
1. **LLaMA.cpp GGUF Exported Model (`finetuned_model.gguf`)**:
   - **Architecture**: `llama` / `nanogpt`
   - **Quantization**: `F16 (Half Precision)` / `F32`
   - **Context Length**: `512` - `2048` tokens
   - **Inference Speed**: ~**141+ Tokens/sec** on NVIDIA GeForce GTX 1650
   - **Compatible with**: LM Studio, Ollama, llama.cpp CLI, Jan, and the in-app Web Studio.

2. **Hugging Face Pretrained Models (Cached In-App)**:
   - `Qwen2.5-0.5B-Instruct` (~0.9GB) — Best Bengali & Multilingual reasoning.
   - `SmolLM2-135M-Instruct` (~270MB) — Ultra-lightweight conversational model.
   - `TinyLlama-1.1B-Chat` (~2.2GB) — Compact chat assistant.
   - `DistilGPT-2` (~82M Params) — Fast local baseline.

## 💾 Storage Policy
All model downloads are explicitly constrained to the application directory on `F:\` drive (`os.environ["HF_HOME"] = "F:/CUDA TEST/models/huggingface"`). Zero files are placed in the `C:\` drive system partition.
