# Models and runtimes

Choose the model's role before its size. A text model, visual encoder, function caller and orchestration framework provide different components of an application. Released parameter counts are not total resident application memory. Pin the model, tokenizer/template, precision format, runtime and device revision for a run.

## Role-matched model artifacts

| Role | Representative artifacts | Deployment question |
|---|---|---|
| Local text, retrieval and bounded reasoning | [Qwen3 0.6B/1.7B/4B](https://huggingface.co/Qwen/Qwen3-1.7B), [Llama 3.2 1B/3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [SmolLM2 135M/360M/1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | Does the selected precision and context preserve task correctness within sustained memory/latency limits? |
| Bounded device actions | [FunctionGemma 270M](https://ai.google.dev/gemma/docs/functiongemma/model_card), [Mobile Actions example](https://developers.googleblog.com/on-device-function-calling-in-google-ai-edge-gallery/) | Who parses, authorizes, confirms and executes each call? A function caller alone is not an autonomous planner. |
| Visual and audio assistance | [Gemma 3n E2B/E4B](https://ai.google.dev/gemma/docs/gemma-3n), [FastVLM](https://github.com/apple/ml-fastvlm), [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | Include encoder/token cost, conditional loading and modality quality; effective-size labels are not complete RAM estimates. |
| Low-bit local text | [BitNet](https://github.com/microsoft/BitNet) | Does the backend execute the intended low-bit kernels, and what quality/traffic trade-off results? |
| Personal-device agent | [TinyAgent](https://github.com/SqueezeAILab/TinyAgent) | Are planner, executor, tool state and final task verification all accounted for? |

## Model reports

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (arXiv 2025) — Text-model family and reasoning-mode controls.
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) (arXiv 2025) — Gemma 3 model family and its published evaluation scope.
- [MiniCPM4: Ultra-Efficient LLMs on End Devices](https://arxiv.org/abs/2506.07900) (arXiv 2025) — End-device-oriented model and inference design.
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285) (arXiv 2025) — Low-bit model and associated inference assumptions.
- [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://arxiv.org/abs/2402.14905) (arXiv 2024) — Sub-billion model design for on-device use.
- [MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](https://arxiv.org/abs/2509.24945) (arXiv 2025) — Open training recipes for small reasoning models.
- [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800) (arXiv 2024) — Mobile-oriented vision-language modeling.
- [MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393) (arXiv 2026) — Full-duplex multimodal interaction; evaluate the complete streaming pipeline.

## Generative inference stacks

| Stack | Intended execution role | Check before comparison |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF model execution across supported CPU/GPU and other backends | Quantizer revision, actual backend and operator fallback |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | Compiled deployment to supported mobile, browser and accelerator targets | Target build, compilation and model-library compatibility |
| [ExecuTorch](https://github.com/pytorch/executorch) | PyTorch model export and on-device execution | Delegation, supported operators and exported graph |
| [LiteRT](https://github.com/google-ai-edge/LiteRT) / [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) | On-device ML and generative-model execution | Supported model conversion and selected device accelerator |
| [MLX](https://github.com/ml-explore/mlx) / [MLX LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon arrays, language-model generation and adaptation | Unified-memory residency and measured device/configuration |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) / [GenAI](https://github.com/microsoft/onnxruntime-genai) | ONNX execution providers and generation loop | Execution provider, graph partitions, KV layout and sampling |
| [MNN](https://github.com/alibaba/MNN) / [ncnn](https://github.com/Tencent/ncnn) | Mobile and embedded inference | Model-specific operator and precision support |
| [Transformers.js](https://github.com/huggingface/transformers.js) | Browser/JavaScript inference | Browser backend, memory limits and model format |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Supported NVIDIA accelerator serving | Exact GPU/platform support; do not infer phone portability |

Use current vendor documentation for [Core ML](https://developer.apple.com/documentation/coreml), [Qualcomm AI Hub](https://aihub.qualcomm.com/) and [LiteRT](https://developers.google.com/edge/litert/overview). Version-specific accelerator paths change. Runtime availability does not establish equal capability or performance across targets.

## MCU and embedded inference

[TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro), [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN), [TinyEngine](https://github.com/mit-han-lab/tinyengine) and [emlearn](https://github.com/emlearn/emlearn) support constrained inference roles. Their small-network results should not be presented as evidence that a complete LLM agent fits an MCU. See the [foundations](foundations-and-surveys.md#constrained-learning-foundations) for the relevant model and training regimes.

[Back to resource map](../README.md)
