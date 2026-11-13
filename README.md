# Awesome Edge AI

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Stars](https://img.shields.io/github/stars/wangxb96/Awesome-EdgeAI?style=social)](https://github.com/wangxb96/Awesome-EdgeAI/stargazers)
[![License](https://img.shields.io/github/license/wangxb96/Awesome-EdgeAI?color=blue)](LICENSE.md)

**Cognitive capabilities within edge budgets.** A curated guide to language and multimodal models, tool-using agents, and the model–runtime–network choices that make them useful on client devices and nearby servers.

A deployment must preserve a declared task outcome while fitting memory, latency, energy, connectivity and trust constraints. We organize resources around that decision: what the task requires, which bottleneck a technique changes, where it executes, and what its evidence actually measures.

![Cognitive edge resource map](survey/figures/resource_map.png)

## Find a resource

| Your question | Start here | What you will find |
|---|---|---|
| Which model and execution stack fit my task? | [Models and runtimes](docs/models-and-runtimes.md) | Text, multimodal and function-calling roles; official artifacts; client, gateway and MCU toolchains |
| Which bottleneck should I optimize? | [Optimization](docs/optimization.md) | Context/KV state, precision, decoding, hardware mapping, adaptation and federated learning |
| How do I complete a useful task across devices? | [Agents and applications](docs/agents-and-applications.md) | Planning, tools, RAG, delegation, wireless control and application boundaries |
| How should I compare quality, cost and risk? | [Evaluation and trust](docs/evaluation-and-trust.md) | Sustained measurements, episode outcomes, privacy, security and benchmarks |
| Where should I begin reading? | [Foundations and surveys](docs/foundations-and-surveys.md) | Selected edge/TinyML foundations and complementary recent syntheses |

## Follow the deployment argument

1. **Define acceptance.** Specify the useful outcome, required evidence, deadline and permitted actions. A valid model response can still be an incomplete user task.
2. **Measure the bottleneck.** Separate weights, retained KV, workspace, prefill, decode, tools, communication and sustained thermal behavior.
3. **Choose compatible controls.** Context selection, precision, reasoning depth, scheduling and placement change different costs. Verify their interaction on the intended runtime.
4. **Validate the entire episode.** Include retries, failed/deferred tasks, tool effects and all declared devices in the accounting. Use task success alongside quality proxies and execution metrics.

For example, a private mobile assistant may retrieve one decisive fact locally, use a compact text model or bounded function caller, validate the proposed tool action, and defer if neither local nor permitted remote execution meets the acceptance conditions. A workstation agent, a phone demonstration and a network simulator support different deployment claims.

## Research artifacts

This is the official artifact repository for *Cognitive Edge Computing: A Survey on Optimizing Large Language Models and Autonomous Agents for the Edge*. The public preprint is [Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment](https://arxiv.org/abs/2501.03265). Its January 2025 v1, *Optimizing Edge AI*, is the initial version of the same manuscript.

| Artifact | Purpose |
|---|---|
| [Survey protocol](survey/protocol.md) | Scope, source verification, E1/E2/E3 evidence support and search-record limitations |
| [Benchmark protocol](survey/benchmark_protocol.md) | Request/episode boundaries, failure accounting and energy attribution |
| [Descriptive analysis helper](survey/compute_ci.py) | Summaries of supplied request records; does not infer complete-task success |
| [Figure sources and exports](survey/figures/README.md) | Editable diagrams and analytical memory illustration |
| [Literature audit](survey/verify_corpus.py) / [repository audit](survey/verify_repos.py) | Check linked resources across this README and the topic guides |

The manuscript's retained-link census is pinned to [commit `82b6110`](https://github.com/wangxb96/Awesome-EdgeAI/tree/82b6110), with 145 arXiv identifiers, 137 title-audited labels and 123 repository roots. These are different link units, not a study-screening flow. Current guides are curated separately; their manifests do not replace that frozen snapshot. The [protocol](survey/protocol.md#reproduce-the-frozen-census) explains how to reproduce it.

For the current guides:

```sh
python3 survey/verify_corpus.py --offline
python3 survey/verify_repos.py
```

The first command reuses cached primary metadata and reports missing records; the second checks live repository reachability. A successful link check does not establish official ownership, experimental replication or suitability for a particular device.

## Citation

```bibtex
@article{wang2025cognitive,
  title={Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment},
  author={Wang, Xubin and Li, Qing and Jia, Weijia},
  journal={arXiv preprint arXiv:2501.03265},
  year={2025}
}
```

Maintained by [Xubin Wang](https://github.com/wangxb96), Qing Li and Weijia Jia.

[Contribution guide](CONTRIBUTING.md) · [License](LICENSE.md)
