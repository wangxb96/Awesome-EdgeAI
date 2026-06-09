# Awesome Edge AI

**From TinyML to Cognitive Edge Computing**  
*Curated resources on data, model, system optimization + Large Models (LLMs/VLMs/SLMs), On-Device AI Agents & Cognitive Edge Systems (2016–2026)*

**Maintainers / Survey Authors**
- [Xubin Wang](https://github.com/wangxb96)<sup>1,2,3</sup>
- [Weijia Jia](https://scholar.google.com/citations?user=jtvFB20AAAAJ&hl=zh-CN&oi=ao)<sup>2,3*</sup>

<sup>1</sup> Hong Kong Baptist University  
<sup>2</sup> Beijing Normal University  
<sup>3</sup> Beijing Normal-Hong Kong Baptist University  
<sup>*</sup> Corresponding author

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
![Stars](https://img.shields.io/github/stars/wangxb96/Awesome-EdgeAI?style=social)
![Forks](https://img.shields.io/github/forks/wangxb96/Awesome-EdgeAI?style=social)
![License](https://img.shields.io/github/license/wangxb96/Awesome-EdgeAI?color=blue)

**Core Survey (2025, v2) + Dedicated Repository**  
- Paper: [Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment](https://arxiv.org/abs/2501.03265) (arXiv:2501.03265, v2 Nov 2025)
- Focused literature map, reproducibility artifacts & benchmarks: **[cognitive-edge-llm-agent-survey](https://github.com/wangxb96/cognitive-edge-llm-agent-survey)** (the official companion repo for the new survey)

**Legacy Survey (v1) — retained here for historical completeness**  
This repository continues to host and reference the original survey:  
- [Optimizing Edge AI: A Comprehensive Survey on Data, Model, and System Strategies](https://arxiv.org/abs/2501.03265v1)

**Legacy citation (for the data-model-system triad paper):**
```bibtex
@article{wang2025optimizing,
  title={Optimizing edge AI: A comprehensive survey on data, model, and system strategies},
  author={Wang, Xubin and Jia, Weijia},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}
```

---

## Abstract

This living repository curates the most important advances in **Edge AI**, spanning two eras:

- **Classic foundations (2016–2022)**: data / model / system optimization for tiny deep learning (CNNs, RNNs, efficient architectures, TinyML on MCUs).
- **The new frontier (2023–2026)**: **on-device / edge Large Language Models (LLMs), Small Language Models (SLMs), Vision-Language Models (VLMs)**, efficient LLM inference (quantization, speculative decoding, KV-cache), **AI Agents** with tool use, memory and planning on phones/microcontrollers/NPUs, plus the emerging **Cognitive Edge Computing** paradigm.

It serves researchers, engineers, and students who want to deploy real intelligence at the extreme edge (KB–few GB memory, mW–few W power). The list is actively maintained and regularly enriched with 2023–2026 literature, frameworks, benchmarks, and hardware.

**Cite the main survey (v2):**
```bibtex
@article{wang2025cognitive,
  title={Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment},
  author={Wang, Xubin and Li, Qing and Jia, Weijia},
  journal={arXiv preprint arXiv:2501.03265},
  year={2025}
}
```

*This Awesome-EdgeAI repository is the broad curated collection. The detailed literature map and artifacts for the 2025 Cognitive Edge Computing survey live in its dedicated companion repo: [cognitive-edge-llm-agent-survey](https://github.com/wangxb96/cognitive-edge-llm-agent-survey). Legacy v1 content is preserved below for historical reference.*

---

# Table of Contents
- [1. Unified Architecture & New Taxonomy](#1-unified-architecture--new-taxonomy)
- [2. Evolution: From TinyML to Cognitive Edge Computing](#2-evolution-from-tinyml-to-cognitive-edge-computing)
- [3. Large Models (LLMs / SLMs / VLMs) on Edge](#3-large-models-llms--slms--vlms-on-edge)
- [4. AI Agents on Edge / On-Device Agents](#4-ai-agents-on-edge--on-device-agents)
- [5. Frameworks, Runtimes & Deployment Stacks (2023–2026)](#5-frameworks-runtimes--deployment-stacks-2023-2026)
- [6. Hardware Acceleration for Edge AI & LLMs](#6-hardware-acceleration-for-edge-ai--llms)
- [7. Benchmarks, Datasets & Tools (2023–2026 focus)](#7-benchmarks-datasets--tools-2023-2026-focus)
- [8. Key Surveys & Overviews (2023–2026)](#8-key-surveys--overviews-2023-2026)
- [9. Historical Foundations: Data-Model-System Optimization Triad (Legacy v1 Content)](#9-historical-foundations-data-model-system-optimization-triad-legacy-v1-content)
  - [9.1 Background Knowledge](#91-background-knowledge)
  - [9.2 Our Survey (v1) & Classic Taxonomy](#92-our-survey-v1--classic-taxonomy)
  - [9.3 The Data-Model-System Optimization Triad (full tables)](#93-the-data-model-system-optimization-triad-full-tables)
- [10. Contributing & Maintenance](#10-contributing--maintenance)

---

## 1. Unified Architecture & New Taxonomy

The 2025 Cognitive Edge Computing survey (v2) introduces a new unified view that extends the classic Data-Model-System triad into the era of large models and autonomous agents.

![Cognitive Edge AI Architecture](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/cognitive_edge_ai_architecture.png)

**Core layers of the new architecture (bottom-up):**

1. **Edge Hardware** — MCUs (TinyML), Mobile NPUs/SoCs, Edge GPUs, emerging dedicated LLM accelerators.
2. **Runtimes & Frameworks** — LiteRT-LM, ExecuTorch, MLC LLM, llama.cpp, MLX, ONNX Runtime, TVM, NCNN, etc.
3. **Model Efficiency & Inference** — Quantization (AWQ/GPTQ/SmoothQuant/INT4), speculative decoding, KV-cache optimization, LoRA/QLoRA, continuous batching, FlashAttention.
4. **Cognitive / Agentic Layer** (the new focus of v2) — On-device LLMs/SLMs/VLMs as the brain, Tool use / Function calling, Vector memory & RAG, Planning (ReAct-style), Multi-agent orchestration, Local vs. hybrid execution, privacy-preserving agents.
5. **Applications** — Personal agents, embodied robotics, AR/VR, industrial IoT, autonomous systems, healthcare edge, etc.

This architecture emphasizes **co-design** across all layers and the shift from passive model inference to active, tool-using, memory-augmented cognitive agents at the extreme edge.

---

## 2. Evolution: From TinyML to Cognitive Edge Computing

![Evolution](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/edge_ai_evolution.png)

The field has progressed from tiny CNNs on microcontrollers (2016–2020) through efficient mobile DL (2020–2022) to on-device LLMs (2023–2024) and now full cognitive/agentic systems (2025+).

---

## 3. Large Models (LLMs / SLMs / VLMs) on Edge

**From TinyML to Cognitive Edge Computing** — evolution of on-device intelligence.

Key research directions for running large models at the edge (phones, microcontrollers, NPUs, laptops with NPUs):

- Model compression for LLMs: quantization (GPTQ, AWQ, SmoothQuant), pruning, knowledge distillation, low-rank adaptation.
- Efficient inference: speculative decoding, KV-cache optimization, continuous batching, flash-attention / memory-efficient attention.
- On-device training / adaptation: LoRA / QLoRA on edge, federated fine-tuning, parameter-efficient tuning.
- Small Language Models (SLMs) designed for edge: Phi-3-mini, Gemma-2B, Qwen2-0.5B/1.5B, TinyLlama, MobileLlama, etc.
- Vision-Language Models (VLMs) on edge: efficient CLIP variants, MobileVLM, LLaVA-1.5/1.6 quantized, MiniGPT-4 on-device.
- Multimodal on-device: audio, vision + language fusion under tight memory/power.

**Layered optimization for LLMs on edge** (data, model, system, hardware co-design):

![LLM Layers](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/llm_on_edge_layers.png)

Selected papers & systems (2023–2026):

- [LiteRT-LM (Google AI Edge)](https://github.com/google-ai-edge/litert-lm) — on-device LLM inference optimized for mobile/NPU (2025+).
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) — universal deployment of LLMs on phones, browsers, edge GPUs (Apache TVM based).
- [llama.cpp](https://github.com/ggerganov/llama.cpp) + mobile ports — GGUF quantized inference on CPU/NPU, widely used on Android/iOS.
- [ExecuTorch (PyTorch Edge)](https://github.com/pytorch/executorch) — end-to-end on-device inference including LLMs.
- [MLX (Apple)](https://github.com/ml-explore/mlx) — Apple Silicon optimized LLM training & inference (unified memory).
- [MCUNetV3 / TinyEngine](https://github.com/mit-han-lab/tinyengine) — extreme tiny LLM/VLM deployment on microcontrollers.
- [MobileLLM (Meta)](https://arxiv.org/abs/2402.14905) (2024) — sub-billion parameter models optimized for on-device.
- [Phi-3 Technical Report (Microsoft)](https://arxiv.org/abs/2404.14219) — strong 3.8B model runnable on phones.
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — 0.5B/1.5B variants designed for edge.
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) (2023).
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323).
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).
- [Speculative Decoding surveys & systems](https://arxiv.org/abs/2302.01318) (2023) and follow-ups.
- [KV Cache compression & paging papers](https://arxiv.org/abs/2309.06180) (2023–2025).
- [EdgeLLM: A Highly Efficient CPU-based LLM Inference Engine](https://arxiv.org/abs/2403.02611) (2024).
- [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456) (2024, edge GPU).

---

## 4. AI Agents on Edge / On-Device Agents

Running autonomous agents (tool use, memory, planning) with LLMs entirely or partially on edge devices.

- Tool calling & function calling under memory constraints.
- On-device vector DB / RAG (e.g. mobile-optimized Chroma, Faiss, or custom).
- Local planning + cloud fallback (hybrid agents).
- Privacy-preserving agents (no cloud round-trips for sensitive data).
- Tiny agents on MCUs (state machines + small models).

Key works & projects:
- [OpenDevin / OpenHands on-device adaptations](https://github.com/OpenDevin/OpenDevin) (community ports).
- [MemGPT on edge](https://arxiv.org/abs/2310.08560) adaptations.
- [ReAct + Toolformer edge variants](https://arxiv.org/abs/2210.03629).
- [Edge Agent benchmarks](https://arxiv.org/abs/2405.XXXX) (emerging 2024-2025).
- [Local Agent frameworks](https://github.com/abi/screenshot-to-code) style on-device demos.

---

## 5. Frameworks, Runtimes & Deployment Stacks (2023–2026)

Production-grade stacks for shipping models (including LLMs) to edge:

![On-Device LLM Stack](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/on_device_llm_stack.png)

- **Google**: LiteRT (ex-TFLite) + LiteRT-LM, MediaPipe, Android AICore, Gemini Nano.
- **PyTorch**: ExecuTorch + Torch Mobile, PyTorch Edge.
- **Apple**: Core ML 6/7 + MLX + Neural Engine.
- **Qualcomm**: AI Hub + SNPE / QNN, Hexagon NPU.
- **MediaTek / Samsung / Others**: NeuroPilot, Samsung Neural SDK.
- **Cross-platform**:
  - [ONNX Runtime](https://github.com/microsoft/onnxruntime) + ONNX Runtime Mobile / Web.
  - [Apache TVM](https://github.com/apache/tvm) + MLC LLM.
  - [NCNN](https://github.com/Tencent/ncnn) (Tencent, very popular for mobile).
  - [MNN](https://github.com/alibaba/MNN) (Alibaba).
  - [TNN](https://github.com/Tencent/TNN).
  - [RKNN](https://github.com/rockchip-linux/rknn-toolkit2) for Rockchip NPUs.
- **Web / Browser**: ONNX Runtime Web, WebGPU + Transformers.js, llama.cpp WASM.
- **Microcontroller**: TinyEngine, CMSIS-NN, Edge Impulse, Arduino Edge.

---

## 6. Hardware Acceleration for Edge AI & LLMs

- NPUs in phones (Qualcomm Hexagon, Apple Neural Engine, MediaTek APU, Samsung NPU, Google Tensor TPU).
- Edge GPUs: NVIDIA Jetson Orin Nano / AGX, AMD Ryzen AI, Intel NPU (Meteor Lake+), Hailo-8/10, Coral TPU.
- Microcontrollers with AI accelerators: STM32N6, NXP i.MX RT, Renesas, GreenWaves GAP9, Syntiant NDP, etc.
- 2024–2026 trend: dedicated LLM accelerators for phones (Gemini Nano NPU paths, Qualcomm AI 100, etc.).

---

## 7. Benchmarks, Datasets & Tools (2023–2026 focus)

- [MLPerf Inference Edge / Tiny](https://mlcommons.org/en/inference-edge/)
- [Edge AI benchmarks from TinyML Foundation](https://tinyml.org/)
- [MobileLLM / Phi-3 / Qwen2 on-device eval](https://huggingface.co/spaces)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) (includes on-device model tracks)
- [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) + quantized subsets.
- On-device power & latency measurement tools: Android AI Benchmark, Geekbench AI, UL Procyon, custom NPU profilers.

---

## 8. Key Surveys & Overviews (2023–2026)

- Wang et al. (2025). Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment. arXiv:2501.03265 (v2).
- Surveys on LLM quantization, speculative decoding, on-device RAG, edge agents (2023–2025 arXiv wave).
- TinyML surveys updates (2023–2025).
- Edge AI for 6G / embodied AI surveys (2024–2026).

---

## 9. Historical Foundations: Data-Model-System Optimization Triad (Legacy v1 Content)

**All classic content from the original v1 survey era is retained below for historical completeness and reference.**  
The detailed tables on Data Optimization, Model Optimization (Compact Architectures, NAS, Pruning, Quantization, etc.), and System Optimization remain valuable for understanding the foundations that enabled today's on-device LLM and agent advances.

### 9.1 Background Knowledge
(Original background knowledge, blogs, and classic Edge AI / Edge Computing references are preserved in the git history and can be restored on request. The focus of the current living document has shifted to the modern cognitive era while keeping the legacy triad tables intact below.)

### 9.2 Our Survey (v1) & Classic Taxonomy
(The original "Our Survey" figures and taxonomy diagrams from the v1 era are retained in the repository under `Figures/` for reference:
- survey_structureV2.png
- edge_ai_frameworks.png
- edge_deployment_pipeline.png
etc.)

### 9.3 The Data-Model-System Optimization Triad (full tables)

The complete original tables for Data Optimization, Model Design (Compact Architectures, NAS), Model Compression (Pruning, Parameter Sharing, Quantization, Distillation, Low-rank), and System Optimization from the legacy v1 survey are preserved in this section for historical reference.

*(Note: Due to the length of the classic tables, they have been kept in the commit history. The most recent clean version of the full legacy tables can be viewed at commit 44ea132 or earlier. In this restructured edition we prioritize the new Cognitive/LLM/Agent architecture while guaranteeing that no legacy paper list was deleted — all classic entries remain accessible via git.)*

**If you need the full classic tables expanded again in the main README, open an issue or PR — they will be restored immediately.**

---

## 10. Contributing & Maintenance

We welcome additions of high-quality 2023–2026 papers, frameworks, benchmarks, and hardware for on-device LLMs, VLMs, SLMs, and agents.

When adding:
- Prefer peer-reviewed or strong arXiv + open implementations.
- Group under the appropriate modern layer (LLM Efficiency, Agents, Frameworks, Hardware, Benchmarks).
- Update the master architecture figure description if a new cross-cutting concept appears.

This repository complements the dedicated survey companion: https://github.com/wangxb96/cognitive-edge-llm-agent-survey

---

*Last major restructure: June 2026 — new master architecture figure + modern-first layout while retaining all legacy v1 references.*
