# Awesome Edge AI

**From TinyML to Cognitive Edge Computing**  
*Curated resources on data, model, system optimization + Large Models (LLMs/VLMs), Agents & On-Device AI (2016–2026)*

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

This living repository curates the most important advances in **Edge AI**, spanning:

- Classic **data / model / system** optimization for tiny deep learning (CNNs, RNNs, efficient architectures).
- The new frontier: **on-device / edge Large Language Models (LLMs), Vision-Language Models (VLMs), Small Language Models (SLMs)**, efficient inference, quantization, speculative decoding, KV-cache optimization, on-device training, and **AI Agents** that run with tool use on phones, microcontrollers, and NPUs.

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

---

## New: Unified Architecture & Cognitive Edge (2023–2026)

**Master Overview Figure** — the Cognitive Edge Computing unified stack (Fig 1):

![Cognitive Edge AI Architecture](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/cognitive_edge_ai_architecture.png)

> **Five-layer architecture**: **Hardware (NPU/GPU/MCU)** → **Runtimes (llama.cpp, MLC-LLM, ExecuTorch)** → **Model Efficiency (Quantization, Pruning, KD)** → **Agentic/Cognitive (LLM Agents, RAG, Planning)** → **Applications (Healthcare, Smart Home, Autonomous)** — with cross-cutting concerns (security, energy, benchmarks, data pipeline, feedback loops, networking, standardization) on the left. See the full **Modern Era** section below for detailed 2023–2026 literature and supporting figures.

---

# Table of Contents
  - [New: Unified Architecture & Cognitive Edge (2023–2026)](#new-unified-architecture--cognitive-edge-2023-2026)
    - [Modern Era: Large Models, Agents & Cognitive Systems](#modern-era-large-models-agents--cognitive-systems-on-edge-2023-2026)
    - [Large Models (LLMs/SLMs/VLMs) on Edge](#large-models-llmsslmsvlms-on-edge)
    - [AI Agents on Edge](#ai-agents-on-edge--on-device-agents)
    - [Frameworks & Runtimes (2023–2026)](#frameworks--runtimes-for-edge--on-device-deployment-2023-2026)
    - [Hardware Acceleration](#hardware-acceleration-for-edge-ai--llms)
    - [Benchmarks & Tools (2023–2026)](#benchmarks-datasets--tools-2023-2026-focus)
    - [Recent Surveys & Highlights](#recent-surveys--overviews-2023-2026)
  - [New: Federated Learning on Edge](#new-federated-learning-on-edge)
  - [New: TinyML & Microcontroller AI](#new-tinyml--microcontroller-ai)
  - [New: Edge AI Security & Privacy](#new-edge-ai-security--privacy)
  - [New: On-Device Training & Personalization](#new-on-device-training--personalization)
  - [New: Multimodal & Embodied Edge AI](#new-multimodal--embodied-edge-ai)
  - [New: Real-World Applications & Case Studies](#new-real-world-applications--case-studies)
  - [Historical Foundations (Legacy v1 Content — fully retained)](#historical-foundations-legacy-v1-content-fully-retained)
    - [Background Knowledge](#1-background-knowledge)
    - [Our Survey (v1) & Classic Taxonomy](#2-our-survey-v1--classic-taxonomy)
    - [The Data-Model-System Optimization Triad (full tables)](#3-the-data-model-system-optimization-triad)
  - [Contributing](#contributing)
    
---

## New: Federated Learning on Edge

Federated Learning (FL) enables collaborative model training across distributed edge devices without centralizing raw data, crucial for privacy-preserving edge AI. Recent advances combine FL with LLMs, PEFT, and heterogeneous edge hardware.

### Federated Learning Foundations & Systems

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)](https://proceedings.mlr.press/v54/mcmahan17a.html) (AISTATS 2017) | Google | -- |
| [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492) (arXiv 2016) | Google | -- |
| [Federated Optimization in Heterogeneous Networks (FedProx)](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html) (MLSys 2020) | University of Michigan | [Code](https://github.com/litian96/FedProx) |
| [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a.html) (ICML 2020) | EPFL | -- |
| [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) (ICLR 2021) | Google Research | -- |
| [Federated Learning with Matched Averaging (FedMA)](https://openreview.net/forum?id=BkluqlSFDS) (ICLR 2020) | IBM Research | -- |

### Federated Learning for LLMs & Edge Models

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Federated Learning of Large Language Models via Parameter-Efficient Tuning](https://arxiv.org/abs/2310.06466) (arXiv 2023) | Zhejiang University | -- |
| [FedPETuning: Federated Parameter-Efficient Tuning for Large Language Models](https://arxiv.org/abs/2310.05811) (arXiv 2023) | -- | -- |
| [Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes](https://arxiv.org/abs/2312.06353) (ICML 2024) | -- | -- |
| [DeepFusion: Accelerating MoE Training via Federated Knowledge Distillation from Heterogeneous Edge Devices](https://arxiv.org/abs/2602.14301) (arXiv 2026) | -- | -- |
| [FedGen-Edge: Parameter-Efficient and Personalized Federated Training of Generative Models at the Edge](https://arxiv.org/abs/2511.11585) (arXiv 2025) | -- | -- |
| [Federated Black-box Prompt Tuning System for Large Language Models on the Edge](https://doi.org/10.1145/3636534.3698856) (MobiCom 2024) | -- | -- |
| [Towards Federated Learning on the Edge: A Survey of Systems, Challenges, and Opportunities](https://arxiv.org/abs/2402.12094) (ACM CSUR 2024) | -- | -- |
| [Split Learning for Distributed Deep Neural Networks](https://arxiv.org/abs/1812.00557) (arXiv 2018) | -- | -- |
| [Efficient Split Learning for Collaborative Edge AI](https://ieeexplore.ieee.org/) (IEEE IoTJ 2023) | -- | -- |

---

## New: TinyML & Microcontroller AI

TinyML pushes AI inference to ultra-low-power microcontrollers (MCUs) with KB-level memory and mW-level power budgets. The field has rapidly evolved from basic CNNs to on-device Transformers and small language models.

### Foundational TinyML Papers

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [MCUNet: Tiny Deep Learning on IoT Devices](https://proceedings.neurips.cc/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html) (NeurIPS 2020) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/mcunet) |
| [MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning](https://proceedings.neurips.cc/paper/2021/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) (NeurIPS 2021) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/mcunet) |
| [MCUNetV3: On-Device Training Under 256KB Memory](https://proceedings.neurips.cc/paper_files/paper/2022/hash/963ba4661bc7947d498241f7c46e48ba-Abstract.html) (NeurIPS 2022) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/mcunet) |
| [TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning](https://proceedings.neurips.cc/paper/2020/hash/69eba34671b3ef1ef38ee85caae6b2a1-Abstract.html) (NeurIPS 2020) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/tinytl) |
| [MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers](https://proceedings.mlsys.org/paper/2021/hash/d67d8ab4f4c10bf22aa353e27879133c-Abstract.html) (MLSys 2021) | Arm Research | -- |
| [TinyML: Current Progress and Future Directions](https://arxiv.org/abs/2009.06034) (arXiv 2020) | Harvard / MIT | -- |
| [On-Device Training Under 256KB Memory](https://arxiv.org/abs/2206.15472) (NeurIPS 2022) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/tiny-training) |
| [TinyEngine: Efficient Training and Inference on Microcontrollers](https://github.com/mit-han-lab/tinyengine) (2023) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/tinyengine) |

### TinyML Frameworks & Tools

- [TensorFlow Lite for Microcontrollers (TFLM)](https://github.com/tensorflow/tflite-micro) — Google's framework for MCU inference.
- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) — ARM's optimized NN kernels for Cortex-M.
- [Edge Impulse](https://www.edgeimpulse.com/) — End-to-end TinyML development platform.
- [Arduino Edge / TinyML Kit](https://www.arduino.cc/) — Arduino's TinyML hardware + software.
- [microTVM](https://tvm.apache.org/docs/topic/microtvm/index.html) — Apache TVM for microcontrollers.
- [Neuton TinyML](https://neuton.ai/) — AutoML for ultra-tiny models.

---

## New: Edge AI Security & Privacy

As edge devices handle increasingly sensitive data with on-device LLMs, security and privacy have become paramount. Key topics include adversarial robustness, model extraction defense, differential privacy, and TEE-based secure inference.

### Security & Adversarial Robustness

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (ICLR 2018) | MIT / UC Berkeley | -- |
| [Adversarial Examples for Semantic Segmentation and Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/html/Xie_Adversarial_Examples_for_ICCV_2017_paper.html) (ICCV 2017) | Johns Hopkins | -- |
| [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://ieeexplore.ieee.org/abstract/document/8294186/) (IEEE Access 2018) | -- | -- |
| [Differential Privacy: A Survey of Results](https://link.springer.com/chapter/10.1007/978-3-540-79228-4_1) (TAMC 2008) | Microsoft Research | -- |
| [Speed Kills: Exploring Confused Deputy Attacks Through Edge AI Accelerators](https://arxiv.org/abs/2605.17707) (CVE-2025-66425, arXiv 2026) | -- | -- |
| [Competition for Attention Predicts Good-to-Bad Tipping in Edge AI](https://arxiv.org/abs/2602.14370) (arXiv 2026) | -- | -- |
| [Integer-Arithmetic-Only Certified Robustness for Quantized Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/html/Lin_Integer-Arithmetic-Only_Certified_Robustness_for_Quantized_Neural_Networks_ICCV_2021_paper.html) (ICCV 2021) | USC | -- |

### Privacy-Preserving Edge AI

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (CCS 2016) | Google | -- |
| [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://ieeexplore.ieee.org/) (IEEE TIFS 2020) | -- | -- |
| [DPFinLLM: Privacy-Enhanced Lightweight LLM for On-Device Financial Applications](https://arxiv.org/abs/2509.08995) (arXiv 2025) | -- | -- |
| [SecureInfer: Heterogeneous TEE-GPU Architecture for Privacy-Critical LLM Deployment](https://arxiv.org/abs/2510.19979) (IEEE ICEdge 2025) | -- | -- |
| [On-Device Generative AI for GDPR-Compliant Visual Monitoring](https://arxiv.org/abs/2605.30544) (arXiv 2026) | -- | -- |
| [PolyLink: A Blockchain-Based Decentralized Edge AI Platform for LLM Inference](https://arxiv.org/abs/2510.02395) (arXiv 2025) | PolyU | [Code](https://github.com/IMCL-PolyLink/PolyLink) |
| [Privacy-Preserving Multimodal Wearable for Local Voice-and-Vision Inference](https://arxiv.org/abs/2511.11811) (arXiv 2025) | UMD | -- |
| [Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust](https://arxiv.org/abs/2508.19870) (arXiv 2025) | -- | -- |

---

## New: On-Device Training & Personalization

Beyond inference, enabling on-device training and fine-tuning allows models to adapt to user-specific data and changing environments without privacy leakage.

### On-Device Training & Fine-tuning

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [On-Device Training Under 256KB Memory](https://arxiv.org/abs/2206.15472) (NeurIPS 2022) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/tiny-training) |
| [TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning](https://proceedings.neurips.cc/paper/2020/hash/69eba34671b3ef1ef38ee85caae6b2a1-Abstract.html) (NeurIPS 2020) | MIT HAN Lab | [Code](https://github.com/mit-han-lab/tinytl) |
| [Enabling On-Device CNN Training by Self-Supervised Instance Filtering and Error Map Pruning](https://ieeexplore.ieee.org/abstract/document/9211513/) (IEEE TCAD 2020) | University of Pittsburgh | -- |
| [Octo: INT8 Training with Loss-aware Compensation and Backward Quantization for Tiny On-device Learning](https://www.usenix.org/system/files/atc21-zhou.pdf) (USENIX ATC 2021) | PolyU | [Code](https://github.com/kimihe/Octo) |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (ICLR 2022) | Microsoft | [Code](https://github.com/microsoft/LoRA) |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html) (NeurIPS 2023) | University of Washington | [Code](https://github.com/artidoro/qlora) |
| [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) (ICML 2024) | -- | [Code](https://github.com/NVlabs/DoRA) |
| [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454) (ICLR 2024) | -- | -- |
| [Unlocking the Edge Deployment and On-Device Acceleration of Multi-LoRA Enabled One-for-All Foundational LLM](https://arxiv.org/abs/2604.18655) (ACL 2026) | Samsung | -- |
| [Parameter-Efficient and Personalized Federated Training of Generative Models at the Edge (FedGen-Edge)](https://arxiv.org/abs/2511.11585) (arXiv 2025) | -- | -- |
| [PL-NPU: An Energy-Efficient Edge-Device DNN Training Processor With Posit-Based Logarithm-Domain Computing](https://ieeexplore.ieee.org/abstract/document/9803862/) (IEEE TCAS-I 2022) | Tsinghua | -- |

---

## New: Multimodal & Embodied Edge AI

Multimodal models (vision + language + audio) and embodied AI systems (robots, AR/VR, autonomous vehicles) running on edge devices represent the next frontier. Key challenges include fusing modalities under tight memory/power budgets.

### Multimodal Models on Edge

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800) (arXiv 2024) | OpenBMB / THU | [Code](https://github.com/OpenBMB/MiniCPM-V) |
| [MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393) (arXiv 2026) | OpenBMB | -- |
| [MobileVLM: Vision-Language Model for Mobile Devices](https://arxiv.org/abs/2312.16886) (arXiv 2023) | Meituan | [Code](https://github.com/Meituan-AutoML/MobileVLM) |
| [LLaVA-1.5: Improved Baselines for Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) (NeurIPS 2024) | UW-Madison / Microsoft | [Code](https://github.com/haotian-liu/LLaVA) |
| [Self-adapting Large Visual-Language Models to Edge Devices Across Visual Modalities](https://doi.org/10.1007/978-3-031-73390-1_18) (ECCV 2024) | -- | -- |
| [VaVLM: Toward Efficient Edge-Cloud Video Analytics With Vision-Language Models](https://doi.org/10.1109/tbc.2025.3549983) (IEEE TBC 2025) | -- | -- |
| [AdaVFM: Adaptive Vision Foundation Models for Edge Intelligence via LLM-Guided Execution](https://arxiv.org/abs/2604.15622) (arXiv 2026) | Intel Labs / CMU | -- |
| [FastReasonSeg: Fast Reasoning Segmentation for Images and Videos on Edge](https://arxiv.org/abs/2511.12368) (arXiv 2025) | -- | -- |

### Embodied AI & Robotics on Edge

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [VLA-Perf: How Fast Can I Run My VLA? Demystifying VLA Inference Performance](https://arxiv.org/abs/2602.18397) (arXiv 2026) | Stanford | -- |
| [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) (arXiv 2023) | Google DeepMind | -- |
| [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) (RSS 2024) | UC Berkeley | [Code](https://github.com/octo-models/octo) |
| [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) (arXiv 2024) | Physical Intelligence | -- |
| [An Agentic AI Framework with LLMs and CoT for UAV-Assisted Logistics Scheduling with MEC](https://arxiv.org/abs/2605.13221) (arXiv 2026) | NTU | -- |

---

## New: Real-World Applications & Case Studies

Edge AI is transforming diverse industries. This section highlights real-world deployments and application-focused research.

### Healthcare & Wellness

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [ECG Foundation Models and Medical LLMs for Agentic Cardiovascular Intelligence at the Edge](https://arxiv.org/abs/2604.02501) (arXiv 2026) | KAUST | -- |
| [A Memory-Efficient Retrieval Architecture for RAG-Enabled Wearable Medical LLMs-Agents](https://arxiv.org/abs/2510.27107) (BioCAS 2025) | HKUST | -- |
| [Edge2Analysis: A Novel AIoT Platform for Atrial Fibrillation Recognition and Detection](https://ieeexplore.ieee.org/abstract/document/9769989/) (IEEE JBHI 2022) | Sun Yat-Sen | -- |
| [Accessible Melanoma Detection Using Smartphones and Mobile Image Analysis](https://ieeexplore.ieee.org/abstract/document/8316868/) (IEEE TMM 2018) | SUTD | -- |
| [Edge-Based Compression and Classification for Smart Healthcare Systems](https://www.sciencedirect.com/science/article/pii/S0957417418305967) (ESWA 2019) | Qatar University | -- |

### Smart Home & IoT

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [AIoT Smart Home via Autonomous LLM Agents](https://doi.org/10.1109/jiot.2024.3471904) (IEEE IoTJ 2024) | -- | -- |
| [BitRL-Light: 1-bit LLM Agents with DRL for Energy-Efficient Smart Home Lighting](https://arxiv.org/abs/2512.20623) (IPCCC 2025) | -- | -- |
| [VoiceAlign: A Shimming Layer for Enhancing the Usability of Legacy VUI Systems](https://arxiv.org/abs/2602.22374) (IUI 2026) | -- | -- |
| [Privacy-Preserving Multimodal Wearable for Local Voice-and-Vision Inference](https://arxiv.org/abs/2511.11811) (arXiv 2025) | UMD | -- |

### Autonomous Driving & Transportation

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Edge Computing for Autonomous Driving: Opportunities and Challenges](https://ieeexplore.ieee.org/abstract/document/8744265/) (Proc. IEEE 2019) | Wayne State | -- |
| [Edge Intelligence for Autonomous Driving in 6G Wireless System](https://ieeexplore.ieee.org/abstract/document/9430907/) (IEEE Wireless Comm. 2021) | -- | -- |
| [LLM-Generated Fault Scenarios for Evaluating Perception-Driven Lane Following in Autonomous Edge Systems](https://arxiv.org/abs/2604.07362) (arXiv 2026) | -- | -- |
| [Efficient On-Device Training for Object Detection at the Edge](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Farhadi_Enabling_Incremental_Knowledge_Transfer_for_Object_Detection_at_the_Edge_CVPRW_2020_paper.pdf) (CVPRW 2020) | ASU | -- |

### Industrial IoT & Manufacturing

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Edge Computing in Industrial Internet of Things: Architecture, Advances and Challenges](https://ieeexplore.ieee.org/abstract/document/9139976/) (IEEE COMST 2020) | -- | -- |
| [Artificial Intelligence-Driven Mechanism for Edge Computing-Based Industrial Applications](https://ieeexplore.ieee.org/abstract/document/8658105/) (IEEE TII 2019) | -- | -- |
| [A Reconfigurable Method for Intelligent Manufacturing Based on Industrial Cloud and Edge Intelligence](https://ieeexplore.ieee.org/abstract/document/8887246/) (IEEE IoTJ 2019) | -- | -- |
| [Rethinking On-Device LLM Reasoning for IoT DDoS Detection](https://arxiv.org/abs/2601.14343) (arXiv 2026) | -- | -- |

### Edge AI in 6G Networks

| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [6G Needs Agents: Toward Agentic AI-Native Networks for Autonomous Intelligence](https://arxiv.org/abs/2605.01546) (arXiv 2026) | -- | -- |
| [CORE: Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of LLM Agents Over Hierarchical Edge](https://arxiv.org/abs/2601.21822) (IEEE Comm. Mag. 2026) | -- | -- |
| [GELATO: Generative Entropy- and Lyapunov-based Adaptive Token Offloading for Device-Edge Speculative LLM Inference](https://arxiv.org/abs/2605.10124) (arXiv 2026) | BJTU | -- |
| [Fast Collaborative Inference via Distributed Speculative Decoding (TSLT)](https://arxiv.org/abs/2512.16273) (arXiv 2025) | -- | -- |
| [Communication-Efficient Collaborative LLM Inference via Distributed Speculative Decoding (TK-SLT)](https://arxiv.org/abs/2509.04576) (WCSP 2025) | -- | -- |
| [A Survey on Cloud-Edge-Terminal Collaborative Intelligence in AIoT Networks](https://arxiv.org/abs/2508.18803) (arXiv 2025) | -- | -- |

---

## Historical Foundations (Legacy v1 Content — fully retained)

**All previous papers, tables, and classic Data-Model-System content from the original v1 survey are preserved below without any deletions.**

---

## 1. Background Knowledge
### 1.1. Edge Computing
Edge computing is a distributed computing paradigm that brings computation and data storage closer to the sources of data generation. This proximity is expected to improve response times, reduce bandwidth consumption, and enable real-time analytics.

- [What is edge computing? Everything you need to know](https://www.techtarget.com/searchdatacenter/definition/edge-computing)
- [Machine Learning at the Edge — μML](https://heartbeat.comet.ml/machine-learning-at-the-edge-%CE%BCml-2802f1af92de)

### 1.2. Edge AI
Edge AI refers to the deployment of artificial intelligence (AI) algorithms and models directly on edge devices, such as mobile phones, Internet of Things (IoT) devices, and smart sensors. By processing data locally, Edge AI enables real-time decision-making, reduces the need for data transmission to remote servers, and enhances data privacy and security. The proliferation of edge devices and the demand for intelligent, low-latency applications have made Edge AI a critical area of research and development.

#### 1.2.1. Blogs About Edge AI
- [Edge AI – What is it and how does it Work?](https://micro.ai/blog/edge-ai-what-is-it-and-how-does-it-work)
- [What is Edge AI?](https://www.advian.fi/en/what-is-edge-ai)
- [Edge AI – Driving Next-Gen AI Applications in 2022](https://viso.ai/edge-ai/edge-ai-applications-and-trends/)
- [Edge Intelligence: Edge Computing and Machine Learning (2023 Guide)](https://viso.ai/edge-ai/edge-intelligence-deep-learning-with-edge-computing/)
- [What is Edge AI, and how does it work?](https://xailient.com/blog/a-comprehensive-guide-to-edge-ai/)
- [Edge AI 101- What is it, Why is it important, and How to implement Edge AI?](https://www.seeedstudio.com/blog/2021/04/02/edge-ai-what-is-it-and-what-can-it-do-for-edge-iot/)
- [Edge AI: The Future of Artificial Intelligence](https://softtek.eu/en/tech-magazine-en/artificial-intelligence-en/edge-ai-el-futuro-de-la-inteligencia-artificial/)
- [What is Edge AI? Machine Learning + IoT](https://www.digikey.com/en/maker/projects/what-is-edge-ai-machine-learning-iot/4f655838138941138aaad62c170827af)
- [What is edge AI computing?](https://www.telusinternational.com/insights/ai-data/article/what-is-edge-ai-computing)
- [在边缘实现机器学习都需要什么？](https://www.infoq.cn/article/shdudgbwmho0ewwpmk5i)
- [边缘计算 | 在移动设备上部署深度学习模型的思路与注意点](https://www.cnblogs.com/showmeai/p/16627579.html)

## 2. Our Survey (To be released)
### 2.1 The Taxonomy of the Discussed Topics
![Framework](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/survey_structureV2.png)
### 2.2 Edge AI Optimization Triad
We introduce a data-model-system optimization triad for edge deployment.
![Scope](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/edge_ai_frameworks.png)
### 2.3 The Edge AI Deployment Pipeline
An overview of edge deployment. The figure shows a general pipeline from the three aspects of data, model and system. Note that not all steps are necessary in real applications.
![Pipeline](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/edge_deployment_pipeline.png)


##  3. The Data-Model-System Optimization Triad
### 3.1. Data Optimization
An overview of data optimization operations. Data cleaning improves data quality by removing errors and inconsistencies in the raw data. Feature compression is used to eliminate irrelevant and redundant features. For scarce data, data augmentation is employed to increase the data size.
![Data](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/data_optimization.png)

#### 3.1.1. Data Cleaning
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Active label cleaning for improved dataset quality under resource constraints[J]. Nature communications, 2022.](https://www.nature.com/articles/s41467-022-28818-3) | Microsoft Research Cambridge |  [Code](https://github.com/microsoft/InnerEye-DeepLearning/tree/1606729c7a16e1bfeb269694314212b6e2737939/InnerEye-DataQuality) |
| [Locomotion mode recognition using sensory data with noisy labels: A deep learning approach. IEEE Trans. on Mobile Computing.](https://ieeexplore.ieee.org/abstract/document/9653808/) | Indian Institute of Technology BHU Varanasi | [Code](https://github.com/errahulm/LRNL_approach) |
| [Big data cleaning based on mobile edge computing in industrial sensor-cloud[J]. IEEE Trans. on Industrial Informatics, 2019.](https://ieeexplore.ieee.org/abstract/document/8822503/) | Huaqiao University | -- |
| [Federated data cleaning: Collaborative and privacy-preserving data cleaning for edge intelligence[J]. IoTJ, 2020.](https://ieeexplore.ieee.org/abstract/document/9210000/) | Xidian University | -- |
| [A data stream cleaning system using edge intelligence for smart city industrial environments[J]. IEEE Trans. on Industrial Informatics, 2021.](https://ieeexplore.ieee.org/abstract/document/9424956/) | Hangzhou Dianzi University | -- |
| [Protonn: Compressed and accurate knn for resource-scarce devices[C] ICML, 2017.](https://proceedings.mlr.press/v70/gupta17a.html) | Microsoft Research, India | [Code](https://github.com/Microsoft/ELL) |
| [Intelligent data collaboration in heterogeneous-device iot platforms[J]. ACM Trans. on Sensor Networks (TOSN), 2021.](https://dl.acm.org/doi/abs/10.1145/3427912) | Hangzhou Dianzi University | -- |


#### 3.1.2. Feature Compression
##### 3.1.2.1. Feature Selection
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Accessible melanoma detection using smartphones and mobile image analysis[J]. IEEE Trans. on Multimedia, 2018.](https://ieeexplore.ieee.org/abstract/document/8316868/) | Singapore University of Technology and Design | -- |
| [ActID: An efficient framework for activity sensor based user identification[J]. Computers & Security, 2021.](https://sciencedirect.53yu.com/science/article/pii/S0167404821001437) | University of Houston-Clear Lake | -- |
| [Descriptor Scoring for Feature Selection in Real-Time Visual Slam[C] ICIP, 2020.](https://cz5waila03cyo0tux1owpyofgoryroob.oss-cn-beijing.aliyuncs.com/17/B9/05/17B90503C6B685151A7C1EC364D10815.pdf) | Processor Architecture Research Lab, Intel Labs | -- |
| [Edge2Analysis: a novel AIoT platform for atrial fibrillation recognition and detection[J]. IEEE Journal of Biomedical and Health Informatics, 2022.](https://ieeexplore.ieee.org/abstract/document/9769989/) | Sun Yat-Sen University | -- |
| [Feature selection with limited bit depth mutual information for portable embedded systems[J]. Knowledge-Based Systems, 2020.](https://sci-hub.et-fine.com/10.1016/j.knosys.2020.105885) | CITIC, Universidade da Coruña | -- |
| [Seremas: Self-resilient mobile autonomous systems through predictive edge computing[C] SECON, 2021.](https://ieeexplore.ieee.org/abstract/document/9491618/) | University of California, Irvine | -- |
| [A covid-19 detection algorithm using deep features and discrete social learning particle swarm optimization for edge computing devices[J]. ACM Trans. on Internet Technology (TOIT), 2021.](https://dl.acm.org/doi/abs/10.1145/3453170) | Hubei Province Key Laboratory of Intelligent Information Processing and Real-time Industrial System | -- |

##### 3.1.2.2. Feature Extraction
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Supervised compression for resource-constrained edge computing systems[C] WACV, 2022.](https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html) | University of California, Irvine | [Code](https://github.com/yoshitomo-matsubara/supervised-compression) |
| ["Blessing of dimensionality: High-dimensional feature and its efficient compression for face verification." CVPR, 2013.](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Chen_Blessing_of_Dimensionality_2013_CVPR_paper.html) | University of Science and Technology of China | -- |
| [Toward intelligent sensing: Intermediate deep feature compression[J]. TIP, 2019.](https://ieeexplore.ieee.org/abstract/document/8848858/) | Nangyang Technological University | -- | 
| [Selective feature compression for efficient activity recognition inference[C] ICCV, 2021.](http://openaccess.thecvf.com/content/ICCV2021/html/Liu_Selective_Feature_Compression_for_Efficient_Activity_Recognition_Inference_ICCV_2021_paper.html) | Amazon Web Services | -- |
| [Video coding for machines: A paradigm of collaborative compression and intelligent analytics[J]. TIP, 2020.](https://ieeexplore.ieee.org/abstract/document/9180095/) | Peking University | -- |
| [Communication-computation trade-off in resource-constrained edge inference[J]. IEEE Communications Magazine, 2020.](https://ieeexplore.ieee.org/abstract/document/9311935/) | The Hong Kong Polytechnic University | [Code](https://github.com/shaojiawei07/Edge_Inference_three-step_framework) |
| [Edge-based compression and classification for smart healthcare systems: Concept, implementation and evaluation[J]. ESWA, 2019.](https://www.sciencedirect.com/science/article/pii/S0957417418305967) | Qatar University | -- |
| [EFCam: Configuration-adaptive fog-assisted wireless cameras with reinforcement learning[C] SECON, 2021.](https://ieeexplore.ieee.org/abstract/document/9491609/) | Nanyang Technological University | -- | 
| [Edge computing for smart health: Context-aware approaches, opportunities, and challenges[J]. IEEE Network, 2019.](https://ieeexplore.ieee.org/abstract/document/8674240/) | Qatar University | -- |
| [DEEPEYE: A deeply tensor-compressed neural network for video comprehension on terminal devices[J]. TECS, 2020.](https://dl.acm.org/doi/abs/10.1145/3381805) | Shanghai Jiao Tong University | -- |
| [CROWD: crow search and deep learning based feature extractor for classification of Parkinson’s disease[J]. TOIT, 2021.](https://dl.acm.org/doi/abs/10.1145/3418500) | Taif University | -- |
| ["Deep-Learning Based Monitoring Of Fog Layer Dynamics In Wastewater Pumping Stations", Water research 202 (2021): 117482.](https://cz5waila03cyo0tux1owpyofgoryroob.oss-cn-beijing.aliyuncs.com/50/65/C8/5065C803C133DF389836C73A6267F1E0.pdf) | Deltares | -- |
| [Distributed and efficient object detection via interactions among devices, edge, and cloud[J]. IEEE Trans. on Multimedia, 2019.](https://ieeexplore.ieee.org/abstract/document/8695132/) | Central South University | -- |

#### 3.1.3. Data Augmentation
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [An effective litchi detection method based on edge devices in a complex scene[J]. Biosystems Engineering, 2022.](https://sciencedirect.53yu.com/science/article/pii/S1537511022001714) | Beihang University | -- |
| [Segmentation of drivable road using deep fully convolutional residual network with pyramid pooling[J]. Cognitive Computation, 2018.](https://linkspringer.53yu.com/article/10.1007/s12559-017-9524-y) | Tsinghua University | -- |
| [Multiuser physical layer authentication in internet of things with data augmentation[J]. IoTJ, 2019.](https://ieeexplore.ieee.org/abstract/document/8935162/) | University of Electronic Science and Technology of China | -- |
| [Data-augmentation-based cellular traffic prediction in edge-computing-enabled smart city[J]. TII, 2020.](https://ieeexplore.ieee.org/abstract/document/9140397/) | University of Electronic Science and Technology of China | -- |
| [Towards light-weight and real-time line segment detection[C] AAAI, 2022.](https://ojs.aaai.org/index.php/AAAI/article/view/19953) | NAVER/LINE Corp. | [Code](https://github.com/navervision/mlsd) |
| [Intrusion Detection System After Data Augmentation Schemes Based on the VAE and CVAE[J]. IEEE Trans. on Reliability, 2022.](https://ieeexplore.ieee.org/abstract/document/9761959/) | Guangdong Ocean University | -- |
| [Magicinput: Training-free multi-lingual finger input system using data augmentation based on mnists[C] ICIP, 2021.](https://dl.acm.org/doi/abs/10.1145/3412382.3458261) | Shanghai Jiao Tong University | -- |


### 3.2. Model Optimization
An overview of model optimization operations. Model design involves creating lightweight models through manual and automated techniques, including architecture selection, parameter tuning, and regularization. Model compression involves using various techniques, such as pruning, quantization, and knowledge distillation, to reduce the size of the model and obtain a compact model that requires fewer resources while maintaining high accuracy.
![Model](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/model_optimization.png)

#### 3.2.1. Model Design
##### 3.2.1.1. Compact Architecture Design
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv, 2017.](https://arxiv.53yu.com/abs/1704.04861) | Google Inc. | [Code](https://github.com/Zehaos/MobileNet) |
| [Mobilenetv2: Inverted residuals and linear bottlenecks[C] CVPR, 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) | Google Inc. | [Code](https://github.com/d-li14/mobilenetv2.pytorch) |  
| [Searching for mobilenetv3[C]// ICCV, 2019.](http://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | Google Inc. | [Code](https://github.com/xiaolai-sqlai/mobilenetv3) | 
| [Rethinking bottleneck structure for efficient mobile network design[C] ECCV, 2020.](https://arxiv.org/abs/2007.02269) | National University of Singapore | [Code](https://github.com/zhoudaquan/rethinking_bottleneck_design) |
| [Mnasnet: Platform-aware neural architecture search for mobile[C] CVPR, 2019.](http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper) | Google Brain | [Code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) |
| [Shufflenet: An extremely efficient convolutional neural network for mobile devices[C] CVPR, 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html) | Megvii Inc (Face++) | [Code](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1) |
| [Shufflenet v2: Practical guidelines for efficient cnn architecture design[C] ECCV, 2018.](http://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html) | Megvii Inc (Face++) | [Code](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2) |
| [Single path one-shot neural architecture search with uniform sampling[C] ECCV, 2020.](https://linkspringer.53yu.com/chapter/10.1007/978-3-030-58517-4_32) | Megvii Inc (Face++) | [Code](https://github.com/megvii-model/ShuffleNet-Series/tree/master/OneShot) |
| [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size[J]. arXiv, 2016.](https://arxiv.53yu.com/abs/1602.07360) | DeepScale∗ & UC Berkeley | [Code](https://github.com/forresti/SqueezeNet) |
| [Squeezenext: Hardware-aware neural network design[C] CVPR Workshops. 2018.](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w33/html/Gholami_SqueezeNext_Hardware-Aware_Neural_CVPR_2018_paper.html) | UC Berkeley | [Code](https://github.com/Timen/squeezenext-tensorflow) |
| [Ghostnet: More features from cheap operations[C] CVPR, 2020.](http://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html) | Noah’s Ark Lab, Huawei Technologies | [Code](https://github.com/huawei-noah/ghostnet) |
| [Efficientnet: Rethinking model scaling for convolutional neural networks[C] ICML, 2019.](http://proceedings.mlr.press/v97/tan19a.html) | Google Brain | [Code](https://github.com/lukemelas/EfficientNet-PyTorch) |
| [Efficientnetv2: Smaller models and faster training[C] ICML, 2021.](http://proceedings.mlr.press/v139/tan21a.html) | Google Brain | [Code](https://github.com/google/automl/tree/master/efficientnetv2) |
| [Efficientdet: Scalable and efficient object detection[C] CVPR, 2020.](http://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) | Google Brain | [Code](https://github.com/google/automl/tree/master/efficientdet) |
| [Condensenet: An efficient densenet using learned group convolutions[C] CVPR, 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Huang_CondenseNet_An_Efficient_CVPR_2018_paper.html) | Cornell University | [Code](https://github.com/ShichenLiu/CondenseNet) |
| [Condensenet v2: Sparse feature reactivation for deep networks[C] CVPR, 2021.](http://openaccess.thecvf.com/content/CVPR2021/html/Yang_CondenseNet_V2_Sparse_Feature_Reactivation_for_Deep_Networks_CVPR_2021_paper.html) | Tsinghua University |[Code](https://github.com/jianghaojun/CondenseNetV2) |
| [Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation[C] ECCV, 2018.](http://openaccess.thecvf.com/content_ECCV_2018/html/Sachin_Mehta_ESPNet_Efficient_Spatial_ECCV_2018_paper.html) | University of Washington | [Code](https://github.com/sacmehta/ESPNet/) |
| [Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network[C] CVPR, 2019.](http://openaccess.thecvf.com/content_CVPR_2019/html/Mehta_ESPNetv2_A_Light-Weight_Power_Efficient_and_General_Purpose_Convolutional_Neural_CVPR_2019_paper.html) | University of Washington | [Code](https://github.com/sacmehta/ESPNetv2) |
| [Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search[C] CVPR, 2019.](http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html) | UC Berkeley | [Code](https://github.com/facebookresearch/mobile-vision) |
| [Fbnetv2: Differentiable neural architecture search for spatial and channel dimensions[C] CVPR, 2020.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_FBNetV2_Differentiable_Neural_Architecture_Search_for_Spatial_and_Channel_Dimensions_CVPR_2020_paper.pdf) | UC Berkeley | [Code](https://github.com/facebookresearch/mobile-vision) |
| [Fbnetv3: Joint architecture-recipe search using predictor pretraining[C] CVPR, 2021.](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.pdf) | Facebook Inc. & UC Berkeley | -- |
| [Pelee: A real-time object detection system on mobile devices[J]. NeurIPS, 2021.](https://proceedings.neurips.cc/paper/2018/file/9908279ebbf1f9b250ba689db6a0222b-Paper.pdf) | University of Western Ontario | [Code](https://github.com/Robert-JunWang/Pelee) |
| [Going deeper with convolutions[C] CVPR, 2015.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) | Google Inc. | [Code](https://github.com/conan7882/GoogLeNet-Inception) |
| [Batch normalization: Accelerating deep network training by reducing internal covariate shift[C] ICML, 2015.](https://arxiv.org/pdf/1502.03167.pdf) | Google Inc. | [Code](https://github.com/shanglianlm0525/PyTorch-Networks/blob/master/ClassicNetwork/InceptionV2.py) |
| [Rethinking the inception architecture for computer vision[C]// CVPR, 2016.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) | Google Inc. | [Code](https://github.com/shanglianlm0525/PyTorch-Networks/blob/master/ClassicNetwork/InceptionV3.py) |
| [Inception-v4, inception-resnet and the impact of residual connections on learning[C] AAAI, 2017.](https://ojs.aaai.org/index.php/aaai/article/view/11231) | Google Inc. | [Code](https://github.com/shanglianlm0525/PyTorch-Networks/blob/master/ClassicNetwork/InceptionV4.py) |
| [Xception: Deep learning with depthwise separable convolutions[C] CVPR, 2017.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf) | Google, Inc. | [Code](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/xception.py) |
| [Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer[J]. arXiv, 2021.](https://arxiv.org/pdf/2110.02178.pdf) | Apple | [Code](https://github.com/apple/ml-cvnets) |
| [Lite transformer with long-short range attention[J]. arXiv, 2020.](https://arxiv.org/pdf/2004.11886.pdf) | Massachusetts Institute of Technology | [Code](https://github.com/mit-han-lab/lite-transformer) |
| [Coordinate attention for efficient mobile network design[C] CVPR, 2021.](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.pdf) | National University of Singapore | [Code](https://github.com/houqb/CoordAttention) |
| [ECA-Net: Efficient channel attention for deep convolutional neural networks[C] CVPR, 2020.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) | Tianjin University | [Code](https://github.com/BangguWu/ECANet) |
| [Sa-net: Shuffle attention for deep convolutional neural networks[C] ICASSP, 2021.](https://ieeexplore.ieee.org/abstract/document/9414568) | Nanjing University | [Code](https://github.com/wofmanaf/SA-Net) |
| [Triplet Attention: Rethinking the Similarity in Transformers[C] KDD, 2021.](https://dl.acm.org/doi/abs/10.1145/3447548.3467241) | Beihang University | [Code](https://github.com/zhouhaoyi/TripletAttention) |
| [Resnest: Split-attention networks[C] CVPR, 2020.](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf) | Meta | [Code](https://github.com/zhanghang1989/ResNeSt) |    

#### 3.2.1.2. Neural Architecture Search (NAS)
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [FTT-NAS: Discovering fault-tolerant convolutional neural architecture[J]. TODAES), 2021.](https://dl.acm.org/doi/abs/10.1145/3460288) | Tsinghua University | [Code](https://github.com/walkerning/aw_nas) |
| [An adaptive neural architecture search design for collaborative edge-cloud computing[J]. IEEE Network, 2021.](https://ieeexplore.ieee.org/abstract/document/9606812/) | Nanjing University of Posts and Telecommunications | -- |
| [Binarized neural architecture search for efficient object recognition[J]. IJCV, 2021.](https://linkspringer.53yu.com/article/10.1007/s11263-020-01379-y) | Beihang University | -- |
| [Multiobjective reinforcement learning-based neural architecture search for efficient portrait parsing[J]. IEEE Trans. on Cybernetics, 2021.](https://ieeexplore.ieee.org/abstract/document/9524839/) | University of Electronic Science and Technology of China | -- |
| [Intermittent-aware neural architecture search[J]. ACM Transactions on Embedded Computing Systems (TECS), 2021.](https://dl.acm.org/doi/abs/10.1145/3476995) | Academia Sinica and National Taiwan University | [Code](https://github.com/EMCLab-Sinica/Intermittent-aware-NAS) |
| [Hardcore-nas: Hard constrained differentiable neural architecture search[C] ICML, 2021.](http://proceedings.mlr.press/v139/nayman21a.html) | Alibaba Group, Tel Aviv, Israel | [Code](https://github.com/Alibaba-MIIL/HardCoReNAS) |
| [MemNAS: Memory-efficient neural architecture search with grow-trim learning[C] CVPR, 2020.](http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_MemNAS_Memory-Efficient_Neural_Architecture_Search_With_Grow-Trim_Learning_CVPR_2020_paper.html) | Beijing University of Posts and Telecommunications | -- |
| [Pvnas: 3D neural architecture search with point-voxel convolution[J]. TPAMI, 2021.](https://ieeexplore.ieee.org/abstract/document/9527118/) | Massachusetts Institute of Technology | -- |
| [Toward tailored models on private aiot devices: Federated direct neural architecture search[J]. IoTJ, 2022.](https://ieeexplore.ieee.org/abstract/document/9721425/) | Northeastern University, Qinhuangdao | -- |
| [Automatic design of convolutional neural network architectures under resource constraints[J]. TNNLS, 2021.](https://ieeexplore.ieee.org/abstract/document/9609007/) | Sichuan University | -- |


### 3.2.2. Model Compression
#### 3.2.2.1. Model Pruning
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Supervised compression for resource-constrained edge computing systems[C] WACV, 2022.](https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html) | University of Pittsburgh | -- |
| [Train big, then compress: Rethinking model size for efficient training and inference of transformers[C] ICML, 2020.](http://proceedings.mlr.press/v119/li20m.html) | UC Berkeley | -- |
| [Hrank: Filter pruning using high-rank feature map[C]// CVPR, 2020.](http://openaccess.thecvf.com/content_CVPR_2020/html/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.html) |  Xiamen University | [Code](https://github.com/lmbxmu/HRank) | 
| [Clip-q: Deep network compression learning by in-parallel pruning-quantization[C] CVPR, 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html) | Simon Fraser University | -- |
| [Sparse: Sparse architecture search for cnns on resource-constrained microcontrollers[J]. NeurIPS, 2019.](https://proceedings.neurips.cc/paper/2019/hash/044a23cadb567653eb51d4eb40acaa88-Abstract.html) | Arm ML Research | -- |
| [Deepadapter: A collaborative deep learning framework for the mobile web using context-aware network pruning[C] INFOCOM, 2020.](https://ieeexplore.ieee.org/abstract/document/9155379/) |  Beijing University of Posts and Telecommunications | -- |
| [SCANN: Synthesis of compact and accurate neural networks[J]. IEEE Trans. on Computer-Aided Design of Integrated Circuits and Systems, 2021.](https://ieeexplore.ieee.org/abstract/document/9552199/) | Princeton University | -- |
| [Directx: Dynamic resource-aware cnn reconfiguration framework for real-time mobile applications[J]. IEEE Trans. on Computer-Aided Design of Integrated Circuits and Systems, 2020.](https://ieeexplore.ieee.org/abstract/document/9097286/) | George Mason University | -- |
| [Pruning deep reinforcement learning for dual user experience and storage lifetime improvement on mobile devices[J]. IEEE Trans. on Computer-Aided Design of Integrated Circuits and Systems, 2020.](https://ieeexplore.ieee.org/abstract/document/9211447/) | City University of Hong Kong | -- |
| [SuperSlash: A unified design space exploration and model compression methodology for design of deep learning accelerators with reduced off-chip memory access volume[J]. IEEE Trans. on Computer-Aided Design of Integrated Circuits and Systems, 2020.](https://ieeexplore.ieee.org/abstract/document/9211496/) | Information Technology University | -- |
| [Penni: Pruned kernel sharing for efficient CNN inference[C] ICML, 2020.](http://proceedings.mlr.press/v119/li20d.html) | Duke University | [Code](https://github.com/timlee0212/PENNI) | 
| [Fast operation mode selection for highly efficient iot edge devices[J]. IEEE Trans. on Computer-Aided Design of Integrated Circuits and Systems, 2019.](https://ieeexplore.ieee.org/abstract/document/8634947) | Karlsruhe Institute of Technology | -- |
| [Efficient on-chip learning for optical neural networks through power-aware sparse zeroth-order optimization[C] AAAI, 2021.](https://ojs.aaai.org/index.php/AAAI/article/view/16928) | University of Texas at Austin | -- |
| [A Fast Post-Training Pruning Framework for Transformers[C]// NeurIPS](https://arxiv.org/pdf/2204.09656.pdf) | UC Berkeley |  [Code](https://github.com/WoosukKwon/retraining-free-pruning) |
| [Radio frequency fingerprinting on the edge[J]. TMC, 2021.](https://ieeexplore.ieee.org/abstract/document/9372779/) | Northeastern University, Boston | -- |
| [Exploring sparsity in image super-resolution for efficient inference[C] CVPR, 2021.](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Exploring_Sparsity_in_Image_Super-Resolution_for_Efficient_Inference_CVPR_2021_paper.html?ref=https://githubhelp.com) | National University of Defense Technology | [Code](https://github.com/LongguangWang/SMSR) |
| [O3BNN-R: An out-of-order architecture for high-performance and regularized BNN inference[J]. TPDS, 2020.](https://ieeexplore.ieee.org/abstract/document/9154597/) | Boston University | -- |
| [Enabling on-device cnn training by self-supervised instance filtering and error map pruning[J]. TCAD, 2020.](https://ieeexplore.ieee.org/abstract/document/9211513/) | University of Pittsburgh | -- |
| [Dropnet: Reducing neural network complexity via iterative pruning[C] ICML, 2020.](http://proceedings.mlr.press/v119/tan20a.html) | National University of Singapore | [Code](https://github.com/tanchongmin/DropNet) |
| [Edgebert: Sentence-level energy optimizations for latency-aware multi-task nlp inference[C] MICRO-54, 2021.](https://dl.acm.org/doi/abs/10.1145/3466752.3480095) | Harvard University | [Code](https://github.com/harvard-acc/EdgeBERT) |
| [Fusion-catalyzed pruning for optimizing deep learning on intelligent edge devices[J]. TCAD, 2020.](https://ieeexplore.ieee.org/abstract/document/9211462/) | Chinese Academy of Sciences | -- |
| [3D CNN acceleration on FPGA using hardware-aware pruning[C] DAC, 2020.](https://ieeexplore.ieee.org/abstract/document/9218571/) | Northeastern University, MA | -- |
| [Width & depth pruning for vision transformers[C] AAAI, 2020.](https://ojs.aaai.org/index.php/AAAI/article/view/20222) | Institute of Computing Technology, Chinese Academy of Sciences | -- |
| [Prive-hd: Privacy-preserved hyperdimensional computing[C] DAC, 2020.](https://ieeexplore.ieee.org/abstract/document/9218493/) | UC San Diego | -- |
| [NestFL: efficient federated learning through progressive model pruning in heterogeneous edge computing[C] MobiCom, 2022.](https://dl.acm.org/doi/abs/10.1145/3495243.3558248) | Purple Mountain Laboratories, Nanjing | -- |


### 3.2.2.2. Parameter Sharing 
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Deep k-means: Re-training and parameter sharing with harder cluster assignments for compressing deep convolutions[C] ICML, 2018.](http://proceedings.mlr.press/v80/wu18h.html) | Texas A&M University | [Code](https://github.com/Sandbox3aster/Deep-K-Means) |
| [T-basis: a compact representation for neural networks[C] ICML, 2020.](http://proceedings.mlr.press/v119/obukhov20a/obukhov20a.pdf) | ETH Zurich | [Code](http://obukhov.ai/tbasis) |
| [ "Soft Weight-Sharing for Neural Network Compression." International Conference on Learning Representations.](https://arxiv.org/pdf/1702.04008.pdf) | University of Amsterdam | [Code](https://github.com/KarenUllrich/Tutorial-SoftWeightSharingForNNCompression) |
| [ShiftAddNAS: Hardware-inspired search for more accurate and efficient neural networks[C] ICML, 2022.](https://proceedings.mlr.press/v162/you22a.html) | Rice University | [Code](https://github.com/RICE-EIC/ShiftAddNAS) |
| [EfficientTDNN: Efficient architecture search for speaker recognition[J]. IEEE/ACM Trans. on Audio, Speech, and Language Processing, 2022.](https://ieeexplore.ieee.org/abstract/document/9798861/) | Tongji University | [Code](https://github.com/mechanicalsea/sugar) |
| [A generic network compression framework for sequential recommender systems[C] SIGIR, 2020.](https://dl.acm.org/doi/abs/10.1145/3397271.3401125) | University of Science and Technology | [Code](https://github.com/siat-nlp/CpRec) |
| [Neural architecture search for LF-MMI trained time delay neural networks[J]. IEEE/ACM Trans. on Audio, Speech, and Language Processing, 2022.](https://ieeexplore.ieee.org/abstract/document/9721103/) | The Chinese University of Hong Kong | -- |
| [Structured transforms for small-footprint deep learning[J]. NeurIPS, 2015.](https://proceedings.neurips.cc/paper/2015/hash/851300ee84c2b80ed40f51ed26d866fc-Abstract.html) | Google, New York | -- | 


### 3.2.2.3. Model Quantization
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Fractrain: Fractionally squeezing bit savings both temporally and spatially for efficient dnn training[J]. NeurIPS, 2020.](https://proceedings.neurips.cc/paper_files/paper/2020/file/8dc5983b8c4ef1d8fcd5f325f9a65511-Paper.pdf) | Rice University | [Code](https://github.com/RICE-EIC/FracTrain) |
| [Edgebert: Sentence-level energy optimizations for latency-aware multi-task nlp inference[C] MICRO-54, 2021.](https://dl.acm.org/doi/abs/10.1145/3466752.3480095) | Harvard University | -- |
| [Stochastic precision ensemble: self-knowledge distillation for quantized deep neural networks[C] AAAI, 2021.](https://ojs.aaai.org/index.php/AAAI/article/view/16839) | Seoul National University | -- |
| [Q-capsnets: A specialized framework for quantizing capsule networks[C] DAC, 2020.](https://ieeexplore.ieee.org/abstract/document/9218746/) | Technische Universitat Wien (TU Wien) | [Code](https://git.io/JvDIF) |
| [Fspinn: An optimization framework for memory-efficient and energy-efficient spiking neural networks[J]. TCAD, 2020.](https://ieeexplore.ieee.org/abstract/document/9211568/) | Technische Universität Wien | -- |
| [Octo: INT8 Training with Loss-aware Compensation and Backward Quantization for Tiny On-device Learning[C] USENIX Annual Technical Conference. 2021.](https://www.usenix.org/system/files/atc21-zhou.pdf) | Hong Kong Polytechnic University | [Code](https://github.com/kimihe/Octo) |
| [Hardware-centric automl for mixed-precision quantization[J]. IJCV, 2020.](https://linkspringer.53yu.com/article/10.1007/s11263-020-01339-6) | Massachusetts Institute of Technology | -- |
| [An automated quantization framework for high-utilization rram-based pim[J]. TCAD, 2021.](https://ieeexplore.ieee.org/abstract/document/9360862/) | Capital Normal University | -- |
| [Exact neural networks from inexact multipliers via fibonacci weight encoding[C] DAC, 2021.](https://ieeexplore.ieee.org/abstract/document/9586245/) | Swiss Federal Institute of Technology Lausanne (EPFL) | -- |
| [Integer-arithmetic-only certified robustness for quantized neural networks[C] ICCV, 2021.](http://openaccess.thecvf.com/content/ICCV2021/html/Lin_Integer-Arithmetic-Only_Certified_Robustness_for_Quantized_Neural_Networks_ICCV_2021_paper.html) | University of Southern California | -- |
| [Bits-Ensemble: Toward Light-Weight Robust Deep Ensemble by Bits-Sharing[J]. TCAD, 2022.](https://ieeexplore.ieee.org/abstract/document/9854091/) | McGill University | -- |
| [Similarity-Aware CNN for Efficient Video Recognition at the Edge[J]. TCAD, 2021.](https://ieeexplore.ieee.org/abstract/document/9656540/) | University of Southampton | -- |
| [ Data-Free Network Compression via Parametric Non-uniform Mixed Precision Quantization[C] CVPR, 2022.](http://openaccess.thecvf.com/content/CVPR2022/html/Chikin_Data-Free_Network_Compression_via_Parametric_Non-Uniform_Mixed_Precision_Quantization_CVPR_2022_paper.html) | Huawei Noah's Ark Lab | -- |

### 3.2.2.4. Knowledge Distillation
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Be your own teacher: Improve the performance of convolutional neural networks via self distillation[C] ICCV, 2019.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.pdf) | Tsinghua University | [Code](https://github.com/ArchipLab-LinfengZhang/) |
| [Dynabert: Dynamic bert with adaptive width and depth[J]. NeurIPS, 2020.](https://proceedings.neurips.cc/paper/2020/hash/6f5216f8d89b086c18298e043bfe48ed-Abstract.html) | Huawei Noah’s Ark Lab | [Code](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT) |
| [Scan: A scalable neural networks framework towards compact and efficient models[J]. NeurIPS, 2019.](https://proceedings.neurips.cc/paper/2019/hash/934b535800b1cba8f96a5d72f72f1611-Abstract.html) | Tsinghua University | [Code](https://github.com/ArchipLab-LinfengZhang/pytorch-scalable-neural-networks) |
| [Content-aware gan compression[C] CVPR, 2021.](http://openaccess.thecvf.com/content/CVPR2021/html/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.html) | Princeton University | -- |
| [Stochastic precision ensemble: self-knowledge distillation for quantized deep neural networks[C] AAAI, 2021.](https://ojs.aaai.org/index.php/AAAI/article/view/16839) | Seoul National University | -- |
| [Cross-modal knowledge distillation for vision-to-sensor action recognition[C] ICASSP, 2022.](https://ieeexplore.ieee.org/abstract/document/9746752/) | Texas State University | -- |
| [Learning efficient and accurate detectors with dynamic knowledge distillation in remote sensing imagery[J]. IEEE Trans. on Geoscience and Remote Sensing, 2021.](https://ieeexplore.ieee.org/abstract/document/9625952/) | Chinese Academy of Sciences | -- |
| [On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation[C] SIGIR, 2022.](https://dl.acm.org/doi/abs/10.1145/3477495.3531775) | The University of Queensland | [Code](https://github.com/xiaxin1998/OD-Rec) |
| [Personalized edge intelligence via federated self-knowledge distillation[J]. TPDS, 2022.](https://ieeexplore.ieee.org/abstract/document/9964434/) | Huazhong University of Science and Technology | -- |
| [Mobilefaceswap: A lightweight framework for video face swapping[C] AAAI, 2022.](https://ojs.aaai.org/index.php/AAAI/article/view/20203) | Baidu Inc. | -- |
| [Dynamically pruning segformer for efficient semantic segmentation[C] ICASSP, 2022.](https://ieeexplore.ieee.org/abstract/document/9747634) | Amazon Halo Health & Wellness | -- |
| [CDFKD-MFS: Collaborative Data-Free Knowledge Distillation via Multi-Level Feature Sharing[J]. IEEE Trans. on Multimedia, 2022.](https://ieeexplore.ieee.org/abstract/document/9834142/) | Beijing Institute of Technology | [Code](https://github.com/Hao840/CDFKD-MFS) |
| [Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation[J]. NeurIPS, 2022.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3bd2d73b4e96b0ac5a319be58a96016c-Abstract-Conference.html) | Beijing Institute of Technology | [Code](https://github.com/Hao840/manifold-distillation) |
| [Learning Accurate, Speedy, Lightweight CNNs via Instance-Specific Multi-Teacher Knowledge Distillation for Distracted Driver Posture Identification[J]. IEEE Trans. on Intelligent Transportation Systems, 2022.](https://ieeexplore.ieee.org/abstract/document/9750058/) | Hefei Institutes of Physical Science (HFIPS), Chinese Academy of Sciences | -- | 

### 3.2.2.5. Low-rank Factorization
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Learning low-rank deep neural networks via singular vector orthogonality regularization and singular value sparsification[C] CVPR workshops. 2020.](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Yang_Learning_Low-Rank_Deep_Neural_Networks_via_Singular_Vector_Orthogonality_Regularization_CVPRW_2020_paper.pdf) | Duke University | -- |
| [MicroNet: Towards image recognition with extremely low FLOPs[J]. arXiv, 2020.](https://arxiv.org/pdf/2011.12289.pdf) | UC San Diego | -- |
| [Locality Sensitive Hash Aggregated Nonlinear Neighborhood Matrix Factorization for Online Sparse Big Data Analysis[J]. ACM/IMS Transactions on Data Science (TDS), 2022.](https://dl.acm.org/doi/abs/10.1145/3497749) | Hunan University | -- |

### 3.3. System Optimization
An overview of system optimization operations. Software optimization involves developing frameworks for lightweight model training and inference, while hardware optimization focuses on accelerating models using hardware-based approaches to improve computational efficiency on edge devices.
![System](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/system_optimization.png)

#### 3.3.1. Software Optimization
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Hidet: Task-mapping programming paradigm for deep learning tensor programs[C] ASPLOS Conference, 2023.](https://dl.acm.org/doi/abs/10.1145/3575693.3575702) | University of Toronto | [Code](https://www.github.com/hidet-org/hidet) |
| [SparkNoC: An energy-efficiency FPGA-based accelerator using optimized lightweight CNN for edge computing[J]. Journal of Systems Architecture, 2021.](https://sciencedirect.53yu.com/science/article/pii/S1383762121000138) | Shanghai Advanced Research Institute, Chinese Academy of Sciences | -- |
| [Re-architecting the on-chip memory sub-system of machine-learning accelerator for embedded devices[C] ICCAD, 2016.](https://ieeexplore.ieee.org/abstract/document/7827590/) | Institute of Computing Technology, Chinese Academy of Sciences | -- |
| [A unified optimization approach for cnn model inference on integrated gpus[C] ICPP, 2019.](https://dl.acm.org/doi/pdf/10.1145/3337821.3337839) | Amazon Web Services | [Code](https://github.com/dmlc/tvm/) |
| [ACG-engine: An inference accelerator for content generative neural networks[C] ICCAD, 2019.](https://ieeexplore.ieee.org/abstract/document/8942169/) | University of Chinese Academy of Sciences | -- |
| [Edgeeye: An edge service framework for real-time intelligent video analytics[C] EDGESYS Conference, 2018.](https://dl.acm.org/doi/pdf/10.1145/3213344.3213345) | University of Wisconsin-Madison | -- |
| [Haq: Hardware-aware automated quantization with mixed precision[C] CVPR, 2019.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) | Massachusetts Institute of Technology | -- |
| [Source compression with bounded dnn perception loss for iot edge computer vision[C] MobiCom, 2019.](https://dl.acm.org/doi/abs/10.1145/3300061.3345448) | Hewlett Packard Labs | -- |
| [A lightweight collaborative deep neural network for the mobile web in edge cloud[J]. TMC, 2020.](https://ieeexplore.ieee.org/abstract/document/9286558/) | Beijing University of Posts and Telecommunications | -- |
| [Enabling incremental knowledge transfer for object detection at the edge[C] CVPR Workshops, 2020.](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Farhadi_Enabling_Incremental_Knowledge_Transfer_for_Object_Detection_at_the_Edge_CVPRW_2020_paper.pdf) | Arizona State university | -- |
| [DA3: Dynamic Additive Attention Adaption for Memory-Efficient On-Device Multi-Domain Learning[C] CVPR, 2022.](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Yang_DA3_Dynamic_Additive_Attention_Adaption_for_Memory-Efficient_On-Device_Multi-Domain_Learning_CVPRW_2022_paper.pdf) | Arizona State University | -- |
| [An efficient GPU-accelerated inference engine for binary neural network on mobile phones[J]. Journal of Systems Architecture, 2021.](https://sciencedirect.53yu.com/science/article/pii/S1383762121001120) | Sun Yat-sen University | [Code](https://code.ihub.org.cn/projects/915/repository/PhoneBit) |
| [RAPID-RL: A Reconfigurable Architecture with Preemptive-Exits for Efficient Deep-Reinforcement Learning[C] ICRA, 2022.](https://ieeexplore.ieee.org/abstract/document/9812320) | Purdue University | -- |
| [A variational information bottleneck based method to compress sequential networks for human action recognition[C] WACV, 2021.](https://openaccess.thecvf.com/content/WACV2021/papers/Srivastava_A_Variational_Information_Bottleneck_Based_Method_to_Compress_Sequential_Networks_WACV_2021_paper.pdf) | Indian Institute of Technology Delhi | -- |
| [EdgeDRNN: Recurrent neural network accelerator for edge inference[J]. IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 2020.](https://ieeexplore.ieee.org/abstract/document/9268992/) | University of Zürich and ETH Zürich | -- |
| [Structured pruning of recurrent neural networks through neuron selection[J]. Neural Networks, 2020.](https://www.sciencedirect.com/science/article/pii/S0893608019303776) | University of Electronic Science and Technology of China | -- |
| [Dynamically hierarchy revolution: dirnet for compressing recurrent neural network on mobile devices[J]. arXiv, 2018.](https://arxiv.org/pdf/1806.01248.pdf) | Arizona State University | -- | 
| [High-throughput cnn inference on embedded arm big. little multicore processors[J]. TCAD, 2019.](https://ieeexplore.ieee.org/abstract/document/8852739/) | National University of Singapore | -- |
| [SCA: a secure CNN accelerator for both training and inference[C] DAC, 2020.](https://ieeexplore.ieee.org/abstract/document/9218752/) | University of Pittsburgh | -- |
| [NeuLens: spatial-based dynamic acceleration of convolutional neural networks on edge[C] MobiCom, 2022.](https://dl.acm.org/doi/abs/10.1145/3495243.3560528?casa_token=ioLt8xczh7cAAAAA:EpjRGtBuz0Hy3sYw9H4v1TVXcX03I68KMtlLjk1Tt2FhheVS0MA97woEWGgg_pjfjXc_njTf2JV8sQ) | New Jersey Institute of Technology | -- |
| [Weightless neural networks for efficient edge inference[C] PACT, 2022.](https://dl.acm.org/doi/pdf/10.1145/3559009.3569680) | The University of Texas at Austin | [Code](https://github.com/ZSusskind/BTHOWeN) |
| [O3BNN-R: An out-of-order architecture for high-performance and regularized BNN inference[J]. TPDS, 2020.](https://ieeexplore.ieee.org/abstract/document/9154597/) | -- |
| [Blockgnn: Towards efficient gnn acceleration using block-circulant weight matrices[C] DAC, 2021.](https://ieeexplore.ieee.org/abstract/document/9586181/) | Peking University | -- |
| [{Hardware/Software}{Co-Programmable} Framework for Computational {SSDs} to Accelerate Deep Learning Service on {Large-Scale} Graphs[C] FAST, 2022.](https://www.usenix.org/conference/fast22/presentation/kwon) | KAIST | -- |
| [Achieving full parallelism in LSTM via a unified accelerator design[C] ICCD, 2020.](https://ieeexplore.ieee.org/abstract/document/9283568/) | University of Pittsburgh | -- |
| [Pasgcn: An reram-based pim design for gcn with adaptively sparsified graphs[J]. TCAD, 2022.](https://ieeexplore.ieee.org/abstract/document/9774869/) | Shanghai Jiao Tong University | -- |


#### 3.3.2. Hardware Optimization
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Ncpu: An embedded neural cpu architecture on resource-constrained low power devices for real-time end-to-end performance[C] MICRO, 2020.](https://ieeexplore.ieee.org/abstract/document/9251958/) | Northwestern Univeristy, Evanston, IL | -- |
| [Reduct: Keep it close, keep it cool!: Efficient scaling of dnn inference on multi-core cpus with near-cache compute[C] ISCA, 2021.](https://ieeexplore.ieee.org/abstract/document/9499927/) | ETH Zurich  | -- |
| [FARNN: FPGA-GPU hybrid acceleration platform for recurrent neural networks[J]. TPDS, 2021.](https://ieeexplore.ieee.org/abstract/document/9600618/) | Sungkyunkwan University | -- |
| [Apgan: Approximate gan for robust low energy learning from imprecise components[J]. IEEE Trans. on Computers, 2019.](https://ieeexplore.ieee.org/abstract/document/8880521/) | University of Central Florida | -- |
| [An FPGA overlay for CNN inference with fine-grained flexible parallelism[J]. TACO, 2022.](https://dl.acm.org/doi/full/10.1145/3519598) | International Institute of Information Technology | -- |
| [Pipelined data-parallel CPU/GPU scheduling for multi-DNN real-time inference[C] RTSS, 2019.](https://ieeexplore.ieee.org/abstract/document/9052147/) | University of California, Riverside | -- |
| [Deadline-based scheduling for GPU with preemption support[C] RTSS, 2018.](https://ieeexplore.ieee.org/abstract/document/8603197) | University of Modena and Reggio Emilia Modena | -- |
| [Energon: Toward Efficient Acceleration of Transformers Using Dynamic Sparse Attention[J]. TCAD, 2022.](https://ieeexplore.ieee.org/abstract/document/9763839/) | Peking University | -- |
| [Light-OPU: An FPGA-based overlay processor for lightweight convolutional neural networks[C] FPGA, 2022.](https://dl.acm.org/doi/abs/10.1145/3373087.3375311) | University of california, Los Angeles | -- |
| [Fluid Batching: Exit-Aware Preemptive Serving of Early-Exit Neural Networks on Edge NPUs[J]. arXiv, 2022.](https://arxiv.org/pdf/2209.13443.pdf) | Samsung AI Center, Cambridge | -- |
| [BitSystolic: A 26.7 TOPS/W 2b~ 8b NPU with configurable data flows for edge devices[J]. IEEE Trans. on Circuits and Systems I: Regular Papers, 2020.](https://ieeexplore.ieee.org/abstract/document/9301197/) | Duke University | -- |
| [PL-NPU: An Energy-Efficient Edge-Device DNN Training Processor With Posit-Based Logarithm-Domain Computing[J]. IEEE Trans. on Circuits and Systems I: Regular Papers, 2022.](https://ieeexplore.ieee.org/abstract/document/9803862/) | Tsinghua University | -- |

<!-- 
## 4. Important Surveys on Edge AI (Related to edge inference and model deployment)
- [Convergence of edge computing and deep learning: A comprehensive survey. IEEE Communications Surveys & Tutorials, 22(2), 869-904.](https://ieeexplore.ieee.org/abstract/document/8976180/)
![](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/relation_ei_ie.png)
- [Deep learning with edge computing: A review. Proceedings of the IEEE, 107(8), 1655-1674.](https://ieeexplore.ieee.org/abstract/document/8763885/)
![](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/DNN_inference_speedup_methods.png)
- [Machine learning at the network edge: A survey. ACM Computing Surveys (CSUR), 54(8), 1-37.](https://dl.acm.org/doi/abs/10.1145/3469029)
![](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/ML_at_edge.png)
- [Edge intelligence: The confluence of edge computing and artificial intelligence. IEEE Internet of Things Journal, 7(8), 7457-7469.](https://ieeexplore.ieee.org/abstract/document/9052677/)
![](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/research_roadmap_edge_intelligence.png)
- [Edge intelligence: Paving the last mile of artificial intelligence with edge computing. Proceedings of the IEEE, 107(8), 1738-1762.](https://ieeexplore.ieee.org/abstract/document/8736011/)
![](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/six_level_EI.png)
- [Edge intelligence: Empowering intelligence to the edge of network. Proceedings of the IEEE, 109(11), 1778-1837.](https://ieeexplore.ieee.org/abstract/document/9596610/)
![](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/classification_of_edge_ai.png)


## 2. Papers 

### 2.1. Edge Computing
- [Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge computing: Vision and challenges. IEEE internet of things journal, 3(5), 637-646.](https://ieeexplore.ieee.org/abstract/document/7488250)
- [Varghese, B., Wang, N., Barbhuiya, S., Kilpatrick, P., & Nikolopoulos, D. S. (2016, November). Challenges and opportunities in edge computing. In 2016 IEEE International Conference on Smart Cloud (SmartCloud) (pp. 20-26). IEEE.](https://ieeexplore.ieee.org/abstract/document/7796149)
- [Shi, W., & Dustdar, S. (2016). The promise of edge computing. Computer, 49(5), 78-81.](https://ieeexplore.ieee.org/abstract/document/7469991)
- [Satyanarayanan, M. (2017). The emergence of edge computing. Computer, 50(1), 30-39.](https://ieeexplore.ieee.org/abstract/document/7807196)
- [Khan, W. Z., Ahmed, E., Hakak, S., Yaqoob, I., & Ahmed, A. (2019). Edge computing: A survey. Future Generation Computer Systems, 97, 219-235.](https://www.sciencedirect.com/science/article/pii/S0167739X18319903)
- [Abbas, N., Zhang, Y., Taherkordi, A., & Skeie, T. (2017). Mobile edge computing: A survey. IEEE Internet of Things Journal, 5(1), 450-465.](https://ieeexplore.ieee.org/abstract/document/8030322)
- [Mao, Y., You, C., Zhang, J., Huang, K., & Letaief, K. B. (2017). A survey on mobile edge computing: The communication perspective. IEEE communications surveys & tutorials, 19(4), 2322-2358.](https://ieeexplore.ieee.org/abstract/document/8016573)
- [Liu, F., Tang, G., Li, Y., Cai, Z., Zhang, X., & Zhou, T. (2019). A survey on edge computing systems and tools. Proceedings of the IEEE, 107(8), 1537-1562.](https://ieeexplore.ieee.org/abstract/document/8746691/)
- [Premsankar, G., Di Francesco, M., & Taleb, T. (2018). Edge computing for the Internet of Things: A case study. IEEE Internet of Things Journal, 5(2), 1275-1284.](https://ieeexplore.ieee.org/abstract/document/8289317/)
- [Xiao, Y., Jia, Y., Liu, C., Cheng, X., Yu, J., & Lv, W. (2019). Edge computing security: State of the art and challenges. Proceedings of the IEEE, 107(8), 1608-1631.](https://ieeexplore.ieee.org/abstract/document/8741060/)
- [Sonmez, C., Ozgovde, A., & Ersoy, C. (2018). Edgecloudsim: An environment for performance evaluation of edge computing systems. Transactions on Emerging Telecommunications Technologies, 29(11), e3493.](https://onlinelibrary.wiley.com/doi/abs/10.1002/ett.3493)
- [Li, H., Ota, K., & Dong, M. (2018). Learning IoT in edge: Deep learning for the Internet of Things with edge computing. IEEE network, 32(1), 96-101.](https://ieeexplore.ieee.org/abstract/document/8270639)
- [Hassan, N., Gillani, S., Ahmed, E., Yaqoob, I., & Imran, M. (2018). The role of edge computing in internet of things. IEEE communications magazine, 56(11), 110-115.](https://ieeexplore.ieee.org/abstract/document/8450541)
- [Sun, X., & Ansari, N. (2016). EdgeIoT: Mobile edge computing for the Internet of Things. IEEE Communications Magazine, 54(12), 22-29.](https://ieeexplore.ieee.org/abstract/document/7786106)
- [Pan, J., & McElhannon, J. (2017). Future edge cloud and edge computing for internet of things applications. IEEE Internet of Things Journal, 5(1), 439-449.](https://ieeexplore.ieee.org/abstract/document/8089336)
- [Liu, S., Liu, L., Tang, J., Yu, B., Wang, Y., & Shi, W. (2019). Edge computing for autonomous driving: Opportunities and challenges. Proceedings of the IEEE, 107(8), 1697-1716.](https://ieeexplore.ieee.org/abstract/document/8744265/)
- [Shi, W., Pallis, G., & Xu, Z. (2019). Edge computing [scanning the issue]. Proceedings of the IEEE, 107(8), 1474-1481.](https://ieeexplore.ieee.org/abstract/document/8789742/)
- [Chen, B., Wan, J., Celesti, A., Li, D., Abbas, H., & Zhang, Q. (2018). Edge computing in IoT-based manufacturing. IEEE Communications Magazine, 56(9), 103-109.](https://ieeexplore.ieee.org/abstract/document/8466364/)
- [Xiong, Z., Zhang, Y., Niyato, D., Wang, P., & Han, Z. (2018). When mobile blockchain meets edge computing. IEEE Communications Magazine, 56(8), 33-39.](https://ieeexplore.ieee.org/abstract/document/8436042)
- [Porambage, P., Okwuibe, J., Liyanage, M., Ylianttila, M., & Taleb, T. (2018). Survey on multi-access edge computing for internet of things realization. IEEE Communications Surveys & Tutorials, 20(4), 2961-2991.](https://ieeexplore.ieee.org/abstract/document/8391395/)
- [Ahmed, E., Ahmed, A., Yaqoob, I., Shuja, J., Gani, A., Imran, M., & Shoaib, M. (2017). Bringing computation closer toward the user network: Is edge computing the solution?. IEEE Communications Magazine, 55(11), 138-144.](https://ieeexplore.ieee.org/abstract/document/8114564/)
- [Taleb, T., Dutta, S., Ksentini, A., Iqbal, M., & Flinck, H. (2017). Mobile edge computing potential in making cities smarter. IEEE Communications Magazine, 55(3), 38-43.](https://ieeexplore.ieee.org/abstract/document/7876955/)
- [He, Q., Cui, G., Zhang, X., Chen, F., Deng, S., Jin, H., ... & Yang, Y. (2019). A game-theoretical approach for user allocation in edge computing environment. IEEE Transactions on Parallel and Distributed Systems, 31(3), 515-529.](https://ieeexplore.ieee.org/abstract/document/8823046/)
- [Mach, P., & Becvar, Z. (2017). Mobile edge computing: A survey on architecture and computation offloading. IEEE Communications Surveys & Tutorials, 19(3), 1628-1656.](https://ieeexplore.ieee.org/abstract/document/7879258/)
- [Tran, T. X., Hajisami, A., Pandey, P., & Pompili, D. (2017). Collaborative mobile edge computing in 5G networks: New paradigms, scenarios, and challenges. IEEE Communications Magazine, 55(4), 54-61.](https://ieeexplore.ieee.org/abstract/document/7901477)
- [Lin, L., Liao, X., Jin, H., & Li, P. (2019). Computation offloading toward edge computing. Proceedings of the IEEE, 107(8), 1584-1607.](https://ieeexplore.ieee.org/abstract/document/8758310/)
- [Qiu, T., Chi, J., Zhou, X., Ning, Z., Atiquzzaman, M., & Wu, D. O. (2020). Edge computing in industrial internet of things: Architecture, advances and challenges. IEEE Communications Surveys & Tutorials, 22(4), 2462-2488.](https://ieeexplore.ieee.org/abstract/document/9139976/)
- [Luo, Q., Hu, S., Li, C., Li, G., & Shi, W. (2021). Resource scheduling in edge computing: A survey. IEEE Communications Surveys & Tutorials, 23(4), 2131-2165.](https://ieeexplore.ieee.org/document/9519636?utm_source=researcher_app&utm_medium=referral&utm_campaign=RESR_MRKT_Researcher_inbound)
- [Taleb, T., Samdanis, K., Mada, B., Flinck, H., Dutta, S., & Sabella, D. (2017). On multi-access edge computing: A survey of the emerging 5G network edge cloud architecture and orchestration. IEEE Communications Surveys & Tutorials, 19(3), 1657-1681.](https://ieeexplore.ieee.org/abstract/document/7931566/)
- [Khan, L. U., Yaqoob, I., Tran, N. H., Kazmi, S. A., Dang, T. N., & Hong, C. S. (2020). Edge-computing-enabled smart cities: A comprehensive survey. IEEE Internet of Things Journal, 7(10), 10200-10232.](https://ieeexplore.ieee.org/abstract/document/9063670/)
- [Chen, X., Shi, Q., Yang, L., & Xu, J. (2018). ThriftyEdge: Resource-efficient edge computing for intelligent IoT applications. IEEE network, 32(1), 61-65.](https://ieeexplore.ieee.org/abstract/document/8270633/)
- [Baktir, A. C., Ozgovde, A., & Ersoy, C. (2017). How can edge computing benefit from software-defined networking: A survey, use cases, and future directions. IEEE Communications Surveys & Tutorials, 19(4), 2359-2391.](https://ieeexplore.ieee.org/abstract/document/7954011/)
- [Zhang, Z., Zhang, W., & Tseng, F. H. (2019). Satellite mobile edge computing: Improving QoS of high-speed satellite-terrestrial networks using edge computing techniques. IEEE network, 33(1), 70-76.](https://ieeexplore.ieee.org/abstract/document/8610431/)
- [Abdellatif, A. A., Mohamed, A., Chiasserini, C. F., Tlili, M., & Erbad, A. (2019). Edge computing for smart health: Context-aware approaches, opportunities, and challenges. IEEE Network, 33(3), 196-203.](https://ieeexplore.ieee.org/abstract/document/8674240/)
- [Liu, Y., Yang, C., Jiang, L., Xie, S., & Zhang, Y. (2019). Intelligent edge computing for IoT-based energy management in smart cities. IEEE network, 33(2), 111-117.](https://ieeexplore.ieee.org/abstract/document/8675180/)


### 2.2. Edge AI
- [Chen, J., & Ran, X. (2019). Deep learning with edge computing: A review. Proceedings of the IEEE, 107(8), 1655-1674.](https://ieeexplore.ieee.org/abstract/document/8763885/)
- [Deng, S., Zhao, H., Fang, W., Yin, J., Dustdar, S., & Zomaya, A. Y. (2020). Edge intelligence: The confluence of edge computing and artificial intelligence. IEEE Internet of Things Journal, 7(8), 7457-7469.](https://ieeexplore.ieee.org/abstract/document/9052677)
- [Zhou, Z., Chen, X., Li, E., Zeng, L., Luo, K., & Zhang, J. (2019). Edge intelligence: Paving the last mile of artificial intelligence with edge computing. Proceedings of the IEEE, 107(8), 1738-1762.](https://ieeexplore.ieee.org/abstract/document/8736011/)
- [Liu, Y., Peng, M., Shou, G., Chen, Y., & Chen, S. (2020). Toward edge intelligence: Multiaccess edge computing for 5G and Internet of Things. IEEE Internet of Things Journal, 7(8), 6722-6747.](https://ieeexplore.ieee.org/abstract/document/9123504)
- [Sodhro, A. H., Pirbhulal, S., & De Albuquerque, V. H. C. (2019). Artificial intelligence-driven mechanism for edge computing-based industrial applications. IEEE Transactions on Industrial Informatics, 15(7), 4235-4243.](https://ieeexplore.ieee.org/abstract/document/8658105/)
- [Li, E., Zeng, L., Zhou, Z., & Chen, X. (2019). Edge AI: On-demand accelerating deep neural network inference via edge computing. IEEE Transactions on Wireless Communications, 19(1), 447-457.](https://ieeexplore.ieee.org/abstract/document/8876870/)
- [Wang, X., Han, Y., Wang, C., Zhao, Q., Chen, X., & Chen, M. (2019). In-edge ai: Intelligentizing mobile edge computing, caching and communication by federated learning. IEEE Network, 33(5), 156-165.](https://ieeexplore.ieee.org/abstract/document/8770530/)
- [Xu, D., Li, T., Li, Y., Su, X., Tarkoma, S., Jiang, T., ... & Hui, P. (2020). Edge intelligence: Architectures, challenges, and applications. arXiv preprint arXiv:2003.12172.](https://arxiv.org/abs/2003.12172)
- [Li, E., Zhou, Z., & Chen, X. (2018, August). Edge intelligence: On-demand deep learning model co-inference with device-edge synergy. In Proceedings of the 2018 Workshop on Mobile Edge Communications (pp. 31-36).](https://dl.acm.org/doi/abs/10.1145/3229556.3229562)
- [Zhang, J., & Letaief, K. B. (2019). Mobile edge intelligence and computing for the internet of vehicles. Proceedings of the IEEE, 108(2), 246-261.](https://ieeexplore.ieee.org/abstract/document/8884164/)
- [Xiao, Y., Shi, G., Li, Y., Saad, W., & Poor, H. V. (2020). Toward self-learning edge intelligence in 6G. IEEE Communications Magazine, 58(12), 34-40.](https://ieeexplore.ieee.org/abstract/document/9311932/)
- [Zhang, K., Zhu, Y., Maharjan, S., & Zhang, Y. (2019). Edge intelligence and blockchain empowered 5G beyond for the industrial Internet of Things. IEEE network, 33(5), 12-19.](https://ieeexplore.ieee.org/abstract/document/8863721/)
- [Zhang, Y., Ma, X., Zhang, J., Hossain, M. S., Muhammad, G., & Amin, S. U. (2019). Edge intelligence in the cognitive Internet of Things: Improving sensitivity and interactivity. IEEE Network, 33(3), 58-64.](https://ieeexplore.ieee.org/abstract/document/8726073/)
- [Zhang, Y., Huang, H., Yang, L. X., Xiang, Y., & Li, M. (2019). Serious challenges and potential solutions for the industrial Internet of Things with edge intelligence. IEEE Network, 33(5), 41-45.](https://ieeexplore.ieee.org/abstract/document/8863725/)
- [Tang, H., Li, D., Wan, J., Imran, M., & Shoaib, M. (2019). A reconfigurable method for intelligent manufacturing based on industrial cloud and edge intelligence. IEEE Internet of Things Journal, 7(5), 4248-4259.](https://ieeexplore.ieee.org/abstract/document/8887246/)
- [Mills, J., Hu, J., & Min, G. (2019). Communication-efficient federated learning for wireless edge intelligence in IoT. IEEE Internet of Things Journal, 7(7), 5986-5994.](https://ieeexplore.ieee.org/abstract/document/8917724/)
- [Lim, W. Y. B., Ng, J. S., Xiong, Z., Jin, J., Zhang, Y., Niyato, D., ... & Miao, C. (2021). Decentralized edge intelligence: A dynamic resource allocation framework for hierarchical federated learning. IEEE Transactions on Parallel and Distributed Systems, 33(3), 536-550.](https://ieeexplore.ieee.org/abstract/document/9479786/)
- [Muhammad, K., Khan, S., Palade, V., Mehmood, I., & De Albuquerque, V. H. C. (2019). Edge intelligence-assisted smoke detection in foggy surveillance environments. IEEE Transactions on Industrial Informatics, 16(2), 1067-1075.](https://ieeexplore.ieee.org/abstract/document/8709763/)
- [Su, X., Sperlì, G., Moscato, V., Picariello, A., Esposito, C., & Choi, C. (2019). An edge intelligence empowered recommender system enabling cultural heritage applications. IEEE Transactions on Industrial Informatics, 15(7), 4266-4275.](https://ieeexplore.ieee.org/abstract/document/8675979/)
- [Yang, B., Cao, X., Xiong, K., Yuen, C., Guan, Y. L., Leng, S., ... & Han, Z. (2021). Edge intelligence for autonomous driving in 6G wireless system: Design challenges and solutions. IEEE Wireless Communications, 28(2), 40-47.](https://ieeexplore.ieee.org/abstract/document/9430907/)
- [Dai, Y., Zhang, K., Maharjan, S., & Zhang, Y. (2020). Edge intelligence for energy-efficient computation offloading and resource allocation in 5G beyond. IEEE Transactions on Vehicular Technology, 69(10), 12175-12186.](https://ieeexplore.ieee.org/abstract/document/9158401/)
- [Al-Rakhami, M., Gumaei, A., Alsahli, M., Hassan, M. M., Alamri, A., Guerrieri, A., & Fortino, G. (2020). A lightweight and cost effective edge intelligence architecture based on containerization technology. World Wide Web, 23(2), 1341-1360.](https://link.springer.com/article/10.1007/s11280-019-00692-y)
- [Feng, C., Yu, K., Aloqaily, M., Alazab, M., Lv, Z., & Mumtaz, S. (2020). Attribute-based encryption with parallel outsourced decryption for edge intelligent IoV. IEEE Transactions on Vehicular Technology, 69(11), 13784-13795.](https://ieeexplore.ieee.org/abstract/document/8863733/)
- [Feng, C., Yu, K., Aloqaily, M., Alazab, M., Lv, Z., & Mumtaz, S. (2020). Attribute-based encryption with parallel outsourced decryption for edge intelligent IoV. IEEE Transactions on Vehicular Technology, 69(11), 13784-13795.](https://ieeexplore.ieee.org/abstract/document/9209123/)
- [Chen, B., Wan, J., Lan, Y., Imran, M., Li, D., & Guizani, N. (2019). Improving cognitive ability of edge intelligent IIoT through machine learning. IEEE network, 33(5), 61-67.](https://ieeexplore.ieee.org/abstract/document/8863728/)
- [Park, J., Samarakoon, S., Bennis, M., & Debbah, M. (2019). Wireless network intelligence at the edge. Proceedings of the IEEE, 107(11), 2204-2239.](https://ieeexplore.ieee.org/abstract/document/8865093/)
- [Tang, S., Chen, L., He, K., Xia, J., Fan, L., & Nallanathan, A. (2022). Computational intelligence and deep learning for next-generation edge-enabled industrial IoT. IEEE Transactions on Network Science and Engineering.](https://ieeexplore.ieee.org/abstract/document/9790341/)
- [Zhang, W., Zhang, Z., Zeadally, S., Chao, H. C., & Leung, V. C. (2019). MASM: A multiple-algorithm service model for energy-delay optimization in edge artificial intelligence. IEEE Transactions on Industrial Informatics, 15(7), 4216-4224.](https://ieeexplore.ieee.org/abstract/document/8632751/)
- [Ghosh, A. M., & Grolinger, K. (2020). Edge-cloud computing for internet of things data analytics: embedding intelligence in the edge with deep learning. IEEE Transactions on Industrial Informatics, 17(3), 2191-2200.](https://ieeexplore.ieee.org/abstract/document/9139356/)
- [Kang, Y., Hauswald, J., Gao, C., Rovinski, A., Mudge, T., Mars, J., & Tang, L. (2017). Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. ACM SIGARCH Computer Architecture News, 45(1), 615-629.](https://dl.acm.org/doi/abs/10.1145/3093337.3037698)
- [Zhu, G., Liu, D., Du, Y., You, C., Zhang, J., & Huang, K. (2020). Toward an intelligent edge: Wireless communication meets machine learning. IEEE communications magazine, 58(1), 19-25.](https://ieeexplore.ieee.org/abstract/document/8970161/)
- [Yang, H., Wen, J., Wu, X., He, L., & Mumtaz, S. (2019). An efficient edge artificial intelligence multipedestrian tracking method with rank constraint. IEEE Transactions on Industrial Informatics, 15(7), 4178-4188.](https://ieeexplore.ieee.org/abstract/document/8633399/)
- [Shi, Y., Yang, K., Jiang, T., Zhang, J., & Letaief, K. B. (2020). Communication-efficient edge AI: Algorithms and systems. IEEE Communications Surveys & Tutorials, 22(4), 2167-2191.](https://ieeexplore.ieee.org/abstract/document/9134426/)
- [Ding, A. Y., Peltonen, E., Meuser, T., Aral, A., Becker, C., Dustdar, S., ... & Wolf, L. (2022). Roadmap for edge AI: a Dagstuhl perspective. ACM SIGCOMM Computer Communication Review, 52(1), 28-33.](https://dl.acm.org/doi/abs/10.1145/3523230.3523235)
- [Soro, S. (2021). TinyML for ubiquitous edge AI. arXiv preprint arXiv:2102.01255.](https://arxiv.org/ftp/arxiv/papers/2102/2102.01255.pdf)
- [Letaief, K. B., Shi, Y., Lu, J., & Lu, J. (2021). Edge artificial intelligence for 6G: Vision, enabling technologies, and applications. IEEE Journal on Selected Areas in Communications, 40(1), 5-36.](https://ieeexplore.ieee.org/abstract/document/9606720/)
- [Ke, R., Zhuang, Y., Pu, Z., & Wang, Y. (2020). A smart, efficient, and reliable parking surveillance system with edge artificial intelligence on IoT devices. IEEE Transactions on Intelligent Transportation Systems, 22(8), 4962-4974.](https://ieeexplore.ieee.org/abstract/document/9061155/)
- [Marculescu, R., Marculescu, D., & Ogras, U. (2020, August). Edge AI: Systems design and ML for IoT data analytics. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 3565-3566).](https://dl.acm.org/doi/abs/10.1145/3394486.3406479)
- [Mahendran, J. K., Barry, D. T., Nivedha, A. K., & Bhandarkar, S. M. (2021). Computer vision-based assistance system for the visually impaired using mobile edge artificial intelligence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2418-2427).](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Mahendran_Computer_Vision-Based_Assistance_System_for_the_Visually_Impaired_Using_Mobile_CVPRW_2021_paper.html)
- [Stäcker, L., Fei, J., Heidenreich, P., Bonarens, F., Rambach, J., Stricker, D., & Stiller, C. (2021). Deployment of Deep Neural Networks for Object Detection on Edge AI Devices with Runtime Optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1015-1022).](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Stacker_Deployment_of_Deep_Neural_Networks_for_Object_Detection_on_Edge_ICCVW_2021_paper.html)
- [Zawish, M., Davy, S., & Abraham, L. (2022). Complexity-driven cnn compression for resource-constrained edge ai. arXiv preprint arXiv:2208.12816.](https://arxiv.org/abs/2208.12816)
- [Yao, J., Zhang, S., Yao, Y., Wang, F., Ma, J., Zhang, J., ... & Yang, H. (2022). Edge-Cloud Polarization and Collaboration: A Comprehensive Survey for AI. IEEE Transactions on Knowledge and Data Engineering.](https://ieeexplore.ieee.org/abstract/document/9783185/)

---

## Modern Era: Large Models, Agents & Cognitive Systems on Edge (2023–2026)

![Edge AI Evolution](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/edge_ai_evolution.png)

> **Key insight**: From 2016 TinyML (KB-level MCU) to 2026 Agentic Edge (LLM+RAG+Tools), the field has evolved through five distinct eras — each unlocking new capabilities at the edge.


**From TinyML to Cognitive Edge Computing** — evolution of on-device intelligence. The unified architecture above (see [top of this page](#new-unified-architecture--cognitive-edge-2023-2026)) shows the full five-layer stack from hardware to agentic applications.

![Evolution](https://raw.githubusercontent.com/wangxb96/Awesome-EdgeAI/main/Figures/edge_ai_evolution.png)

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
- [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/abs/2406.06282) (arXiv 2024) — strong phone-class LLM execution reference.
- [EdgeLLM: Fast On-Device LLM Inference with Speculative Decoding](https://doi.org/10.1109/tmc.2024.3513457) (IEEE TMC 2024).
- [SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget](https://doi.org/10.18653/v1/2024.acl-long.363) (ACL 2024).
- [Fast On-Device LLM Inference with NPUs](https://doi.org/10.1145/3669940.3707239) (ASPLOS 2025).
- [Lincoln: Real-Time 50-100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory](https://doi.org/10.1109/hpca61900.2025.00128) (HPCA 2025).
- [Marlin: Mixed-precision Auto-regressive Parallel Inference on Large Language Models](https://doi.org/10.1145/3710848.3710871) (PPoPP 2025).
- [MobileQuant: Mobile-friendly Quantization for On-device Language Models](https://doi.org/10.18653/v1/2024.findings-emnlp.570) (EMNLP Findings 2024).
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://doi.org/10.18653/v1/2024.emnlp-main.422) (EMNLP 2024).
- [QSpec: Speculative Decoding with Complementary Quantization Schemes](https://arxiv.org/abs/2410.11305) (arXiv 2024).
- [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800) (arXiv 2024) — notable on-phone multimodal deployment.
- [MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](https://arxiv.org/abs/2509.24945) (arXiv 2025).
- [Self-adapting Large Visual-Language Models to Edge Devices Across Visual Modalities](https://doi.org/10.1007/978-3-031-73390-1_18) (ECCV 2024).
- [VaVLM: Toward Efficient Edge-Cloud Video Analytics With Vision-Language Models](https://doi.org/10.1109/tbc.2025.3549983) (IEEE TBC 2025).
- [MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393) (arXiv 2026) — 9B params, runs on edge with <12GB RAM, full-duplex omni-modal.
- [DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference](https://arxiv.org/abs/2604.26557) (ICDCS 2026).
- [Unlocking the Edge deployment and ondevice acceleration of multi-LoRA enabled one-for-all foundational LLM](https://arxiv.org/abs/2604.18655) (ACL 2026) — Samsung Galaxy S24/S25, INT4, Dynamic Self-Speculative Decoding, 4-6x improvement.
- [FastTTS: Accelerating Test-Time Scaling for Edge LLM Reasoning](https://arxiv.org/abs/2509.00195) (ASPLOS 2026) — 2.2x goodput improvement on edge GPUs.
- [CSGO: Generalized Optimization for Cold Start in Wireless Collaborative Edge LLM Systems](https://arxiv.org/abs/2508.11287) (arXiv 2025).
- [FourierCompress: Layer-Aware Spectral Activation Compression for Collaborative LLM Inference](https://arxiv.org/abs/2510.16418) (arXiv 2025) — 7.6x activation size reduction, <0.3% accuracy loss.
- [SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices](https://arxiv.org/abs/2510.18544) (arXiv 2025) — up to 35x higher SLO attainment.
- [EPARA: Parallelizing Categorized AI Inference in Edge Clouds](https://arxiv.org/abs/2511.00603) (arXiv 2025) — 2.1x higher goodput in edge production.
- [A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks](https://arxiv.org/abs/2604.21399) (arXiv 2026).
- [An Evaluation of LLMs Inference on Popular Single-board Computers](https://arxiv.org/abs/2511.07425) (arXiv 2025) — 25 quantized LLMs on RPi 4/5, Orange Pi 5 Pro.
- [Characterizing and Understanding Energy Footprint and Efficiency of Small Language Model on Edges](https://arxiv.org/abs/2511.11624) (IEEE MASS 2025) — RPi 5, Jetson Nano, Jetson Orin Nano benchmarks.
- [Edge Deployment of Small Language Models: Comprehensive Comparison of CPU, GPU and NPU Backends](https://arxiv.org/abs/2511.22334) (arXiv 2025) — NPU > GPU > CPU for edge SLM inference.
- [Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers](https://arxiv.org/abs/2604.24785) (arXiv 2026).
- [Hardware-Efficient Softmax and Layer Normalization with Guaranteed Normalization for Edge Devices](https://arxiv.org/abs/2604.23647) (ISCAS 2026) — up to 14x area reduction.
- [Punching Above Precision: Small Quantized Model Distillation with Learnable Regularizer](https://arxiv.org/abs/2509.20854) (arXiv 2025).
- [Energy Efficient Software Hardware CoDesign for Machine Learning: From TinyML to LLMs](https://arxiv.org/abs/2603.23668) (ASPLOS EMC2 Workshop 2026).

## 5. AI Agents on Edge / On-Device Agents

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
- [Beyond Self-Talk: A Communication-Centric Survey of LLM-based Multi-Agent Systems](https://arxiv.org/abs/2502.14321) (arXiv 2025) — communication cost & topology for edge feasibility.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (arXiv 2025) — SLM-first agents aligned with deployment constraints.
- [A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence](https://arxiv.org/abs/2507.21046) (arXiv 2025).
- [AIoT Smart Home via Autonomous LLM Agents](https://doi.org/10.1109/jiot.2024.3471904) (IEEE IoTJ 2024).
- [LLM Multi-Agent Systems: Challenges and Open Problems](https://arxiv.org/abs/2402.03578) (arXiv 2024).
- [Federated Black-box Prompt Tuning System for Large Language Models on the Edge](https://doi.org/10.1145/3636534.3698856) (MobiCom 2024).
- [When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems](https://arxiv.org/abs/2605.30102) (ICML AIWILD Workshop 2026) — systematic study of SLM + LLM hybrid agent design space.
- [Agentic AI Reasoning for Mobile Edge General Intelligence](https://arxiv.org/abs/2509.23248) (arXiv 2025) — joint optimization of CoT prompting + distributed MoE for edge agents.
- [EmoMAS: Emotion-Aware Multi-Agent System for High-Stakes Edge-Deployable Negotiation](https://arxiv.org/abs/2604.07003) (arXiv 2026) — Bayesian multi-agent framework for edge negotiation.
- [EQ-Negotiator: Dynamic Emotional Personas Empower SLMs for Edge-Deployable Credit Negotiation](https://arxiv.org/abs/2511.03370) (arXiv 2025).
- [SolidGPT: An Edge-Cloud Hybrid AI Agent Framework for Smart App Development](https://arxiv.org/abs/2512.08286) (2025) — privacy-first edge assistant for code & workspace.
- [An Agentic AI Framework with LLMs and Chain-of-Thought for UAV-Assisted Logistics with MEC](https://arxiv.org/abs/2605.13221) (arXiv 2026).
- [ECG Foundation Models and Medical LLMs for Agentic Cardiovascular Intelligence at the Edge](https://arxiv.org/abs/2604.02501) (arXiv 2026).
- [A Memory-Efficient Retrieval Architecture for RAG-Enabled Wearable Medical LLMs-Agents](https://arxiv.org/abs/2510.27107) (BioCAS 2025).
- [Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust](https://arxiv.org/abs/2508.19870) (arXiv 2025).
- [CORE: Collaborative Orchestration Role at Edge for 6G Intelligence via LLM Agents](https://arxiv.org/abs/2601.21822) (IEEE Comm. Mag. 2026).
- [Toward Agentic Environments: GenAI and the Convergence of AI, Sustainability, and Human-Centric Spaces](https://arxiv.org/abs/2512.15787) (Sustainable Development 2025).

## 6. Frameworks & Runtimes for Edge / On-Device Deployment (2023–2026)

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
- [DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services](https://arxiv.org/abs/2502.11417) (arXiv 2025).
- [CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge](https://arxiv.org/abs/2506.02847) (arXiv 2025).
- [An Edge-Cloud Collaboration Framework for Generative AI Service Provision with Synergetic Big Cloud Model and Small Edge Models](https://doi.org/10.1109/mnet.2024.3420755) (IEEE Network 2024).
- [EdgeShard: Efficient LLM Inference via Collaborative Edge Computing](https://doi.org/10.1109/jiot.2024.3524255) (IEEE IoTJ 2024).
- [Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts](https://doi.org/10.1109/tmc.2025.3590969) (IEEE TMC 2025).

## 7. Hardware Acceleration for Edge AI & LLMs

- NPUs in phones (Qualcomm Hexagon, Apple Neural Engine, MediaTek APU, Samsung NPU, Google Tensor TPU).
- Edge GPUs: NVIDIA Jetson Orin Nano / AGX, AMD Ryzen AI, Intel NPU (Meteor Lake+), Hailo-8/10, Coral TPU.
- Microcontrollers with AI accelerators: STM32N6, NXP i.MX RT, Renesas, GreenWaves GAP9, Syntiant NDP, etc.
- 2024–2026 trend: dedicated LLM accelerators for phones (Gemini Nano NPU paths, Qualcomm AI 100, etc.).
- [Cambricon-LLM: A Chiplet-based Hybrid Architecture for On-Device Inference of 70B LLM](https://doi.org/10.1109/micro61859.2024.00108) (MICRO 2024).
- [PAISE: PIM-Accelerated Inference Scheduling Engine for Transformer-based LLM](https://doi.org/10.1109/hpca61900.2025.00126) (HPCA 2025).
- [FACIL: Flexible DRAM Address Mapping for SoC-PIM Cooperative On-device LLM Inference](https://doi.org/10.1109/hpca61900.2025.00127) (HPCA 2025).
- [T-MAC: CPU Renaissance via Table Lookup for Low-bit LLM Deployment on Edge](https://doi.org/10.1145/3689031.3696099) (EuroSys 2025).
- [Understanding the Potential of FPGA-based Spatial Acceleration for Large Language Model Inference](https://doi.org/10.1145/3656177) (ACM TRETS 2024).
- [Pushing up to the Limit of Memory Bandwidth and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA](https://doi.org/10.23919/date64628.2025.10993087) (DATE 2025).
- [Efficient Kernel Mapping and Comprehensive System Evaluation of LLM Acceleration on a CGLA](https://doi.org/10.1109/ACCESS.2025.3636266) (IEEE Access 2025) — CGRA achieves 44.4x PDP improvement vs RTX 4090.
- [Energy Efficient Software Hardware CoDesign for Machine Learning: From TinyML to Large Language Models](https://arxiv.org/abs/2603.23668) (ASPLOS EMC2 Workshop 2026) — comprehensive review of energy-efficient HW/SW co-design.
- [Hardware-Efficient Softmax and Layer Normalization with Guaranteed Normalization for Edge Devices](https://arxiv.org/abs/2604.23647) (ISCAS 2026) — up to 14x area reduction in 28nm CMOS.
- [Edge Deployment of Small Language Models: CPU, GPU and NPU Backends](https://arxiv.org/abs/2511.22334) (arXiv 2025) — NPU dominates edge SLM inference across EDP metrics.
- [Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers](https://arxiv.org/abs/2604.24785) (arXiv 2026) — multi-dimensional evaluation with NPUs, GPUs on SBCs.

## 8. Benchmarks, Datasets & Tools (2023–2026 focus)

- [MLPerf Inference Edge / Tiny](https://mlcommons.org/en/inference-edge/)
- [Edge AI benchmarks from TinyML Foundation](https://tinyml.org/)
- [MobileLLM / Phi-3 / Qwen2 on-device eval](https://huggingface.co/spaces)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) (includes on-device model tracks)
- [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) + quantized subsets.
- On-device power & latency measurement tools: Android AI Benchmark, Geekbench AI, UL Procyon, custom NPU profilers.
- [MobileAIBench: Benchmarking LLMs/LMMs for On-Device Use Cases](https://arxiv.org/abs/2406.10290) (arXiv 2024).
- [Understanding Large Language Models in Your Pockets: Performance Study on COTS Mobile Devices](https://arxiv.org/abs/2410.03613) (arXiv 2024).
- [Mobile and Edge Evaluation of Large Language Models](https://doi.org/10.36227/techrxiv.172115025.57884352/v1) (TechRxiv / ICML ES-FoMo Workshop 2024).
- [Laskaridis et al. works on realistic edge LLM benchmarking](https://doi.org/10.36227/techrxiv.172115025.57884352/v1).
- [Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers](https://arxiv.org/abs/2604.24785) (arXiv 2026).
- [An Evaluation of LLMs Inference on Popular Single-board Computers](https://arxiv.org/abs/2511.07425) (arXiv 2025) — 25 models, 3 SBCs, Ollama vs Llamafile.
- [Characterizing and Understanding Energy Footprint and Efficiency of SLM on Edges](https://arxiv.org/abs/2511.11624) (IEEE MASS 2025).
- [Edge Deployment of Small Language Models: CPU, GPU and NPU Backends Comparison](https://arxiv.org/abs/2511.22334) (arXiv 2025).
- [VLA-Perf: Demystifying VLA Inference Performance for Embodied AI](https://arxiv.org/abs/2602.18397) (arXiv 2026).
- [SUSTAINABLE LLM INFERENCE FOR EDGE AI: Evaluating Quantized LLMs for Energy Efficiency](https://dl.acm.org/) (ACM TOSEM 2025).

## 9. Recent Surveys & Overviews (2023–2026)

- Wang et al. (2025). Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment. arXiv:2501.03265 (v2).
- Surveys on LLM quantization, speculative decoding, on-device RAG, edge agents (2023–2025 arXiv wave).
- TinyML surveys updates (2023–2025).
- Edge AI for 6G / embodied AI surveys (2024–2026).
- Zheng et al., [A Review on Edge Large Language Models: Design, Execution, and Applications](https://doi.org/10.1145/3719664) (ACM CSUR 2025) — closest neighboring review focused on edge-side LLM execution.
- [Lu et al., Small Language Models: Survey, Measurements, and Insights](https://arxiv.org/abs/2409.15790) (arXiv 2024).
- [Yan et al., Beyond Self-Talk: A Communication-Centric Survey of LLM-based Multi-Agent Systems](https://arxiv.org/abs/2502.14321) (arXiv 2025).
- [Belcak et al., Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (arXiv 2025).
- [Mobile Edge Intelligence for Large Language Models: A Contemporary Survey](https://ieeexplore.ieee.org/) (IEEE COMST 2025) — 334+ citations.
- [On-Device Language Models: A Comprehensive Review](https://arxiv.org/abs/2409.00088) (arXiv 2024).
- [A Survey of AI Inference Technologies for On-Device Systems](https://ieeexplore.ieee.org/) (IEEE IoTJ 2025).
- [A Comprehensive Survey on Large Language Model Compression for AI Applications in Edge Systems](https://ieeexplore.ieee.org/) (IEEE IoTJ 2026).
- [Edge Large Language Models: A Comprehensive Survey](https://link.springer.com/) (CCF Trans. 2026).
- [A Survey on Cloud-Edge-Terminal Collaborative Intelligence in AIoT Networks](https://arxiv.org/abs/2508.18803) (arXiv 2025).
- [Energy Efficient Software Hardware CoDesign for ML: From TinyML to LLMs](https://arxiv.org/abs/2603.23668) (ASPLOS EMC2 2026).
- [A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence](https://arxiv.org/abs/2507.21046) (arXiv 2025).

---

**2024–2026 Top-tier Highlights (NeurIPS/ICLR/ICML/SOSP/HPCA/MICRO/ASPLOS/MobiCom/EMNLP/ECCV/ACL + IEEE TMC/IoTJ)** (added for completeness while preserving all legacy content):

**Systems/Serving & Optimization** (see detailed lists in sections 4 & 6):
- PowerInfer-2, EdgeLLM, SwapMoE, Marlin, EAGLE-2, QSpec, MobileQuant, FastTTS (ASPLOS'26), DUAL-BLADE (ICDCS'26), FourierCompress, SLICE, EPARA, GELATO, CSGO, DiSCo, CLONE.

**Hardware-Aware & Acceleration**:
- Cambricon-LLM (MICRO'24), PAISE/FACIL (HPCA'25), T-MAC (EuroSys'25), Lincoln (HPCA'25), FPGA spatial acceleration (ACM TRETS/DATE'24/25), Fast On-Device LLM with NPUs (ASPLOS'25), HW-Efficient Softmax/LayerNorm (ISCAS'26), CGLA for LLM acceleration (IEEE Access'25).

**Collaboration, Routing & Agentic**:
- EdgeShard, QoS-Aware LLM Routing (IEEE TMC'25), CLONE, DiSCo, Federated Black-box Prompt Tuning (MobiCom'24), LLM Multi-Agent Systems surveys and case studies (AIoT Smart Home, V2V-LLM), Hybrid Multi-Agent Systems (ICML AIWILD'26), EmoMAS, EQ-Negotiator, SolidGPT.

**Multimodal & Applications**:
- MiniCPM-V, MiniCPM-o 4.5 (full-duplex omni-modal), Self-adapting VLMs (ECCV'24), MobileLLM-R1, VaVLM (IEEE TBC'25), Industrial IoT with LLMs (IEEE TMC'24), FastReasonSeg (0.6B, 7.79 FPS), AdaVFM.

**On-Device Training & Personalization**:
- Multi-LoRA on Samsung Galaxy (ACL'26), FedGen-Edge, QLoRA (NeurIPS'23), DoRA (ICML'24), MCUNetV3 (NeurIPS'22).

**Evaluation & Trust**:
- MobileAIBench, pocket-scale performance studies, Mobile and Edge Evaluation of LLMs, VLA-Perf, SBC benchmarks (RPi 4/5, Orange Pi, Jetson), Energy Footprint studies (IEEE MASS'25).

**Security & Privacy**:
- Confused Deputy Attacks on Edge AI Accelerators (CVE-2025-66425), Attention Competition Tipping in Edge AI, SecureInfer (TEE-GPU, ICEdge'25), DPFinLLM, PolyLink (Blockchain Edge AI), Zero-Trust Multi-LLM Agents.

**Emerging Trends (2025-2026)**:
- Full-duplex omni-modal edge interaction (MiniCPM-o 4.5)
- Test-time scaling for edge reasoning (FastTTS, ASPLOS'26)
- Hybrid cloud-device multi-agent systems (ICML'26)
- Agentic AI-native 6G networks
- Federated MoE training from edge devices (DeepFusion)
- Blockchain-based decentralized edge AI platforms (PolyLink)
- Emotion-aware strategic agents for edge negotiation

For the exhaustive categorized map (Foundations, Challenges, Optimization, Agentic, Applications, Evaluation & Trust) and reproducibility artifacts, see the dedicated companion: https://github.com/wangxb96/cognitive-edge-llm-agent-survey

---

*This repository (Awesome-EdgeAI) is the broad curated collection that retains the complete legacy Data-Model-System content for historical reference while adding high-quality 2023–2026 top-tier literature. The detailed literature map and artifacts for the 2025 Cognitive Edge Computing survey live in its official companion repo: [cognitive-edge-llm-agent-survey](https://github.com/wangxb96/cognitive-edge-llm-agent-survey).*
