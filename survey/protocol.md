# Survey protocol and source verification

This repository accompanies *Cognitive Edge Computing: A Survey on Optimizing Large Language Models and Autonomous Agents for the Edge*. The organizing question is which deployment choices preserve a declared cognitive task capability within memory, latency, energy, thermal, connectivity and trust budgets.

The living README, the manuscript bibliography and the frozen retained-link census are different objects. None is a reconstructed systematic study-screening flow.

## Scope and source selection

Included work addresses constrained LLM/SLM/VLM execution, context and memory, model/runtime efficiency, adaptation, device–edge–cloud collaboration, tool-using agents, measurement or security at a stated deployment boundary. Foundational serving and TinyML mechanisms are retained when their transfer limits are clear. A server result does not establish mobile performance.

Conventional perception without an LLM/agent connection, cloud-only training without a deployment role, unverifiable product claims, unrepaired placeholders and withdrawn records are outside the survey scope. The foundations section provides selected prerequisites without counting them as new cognitive-edge validation.

| Dimension | Source boundary |
|---|---|
| Literature | arXiv, IEEE Xplore, ACM Digital Library, Google Scholar for citation discovery, SpringerLink and publisher DOI records |
| Artifacts | Primary project/publisher documentation, model cards, official code and benchmark suites |
| Manuscript coverage | 2015–13 November 2026, emphasizing recent deployment mechanisms; targeted primary-source updates are distinct from the frozen census |
| Frozen link coverage | Through 10 November 2026; snapshot recorded 11 November at `82b6110` |

Search concepts combine deployment tier, model/agent role, optimization mechanism and evaluation constraint:

1. Tier: edge, on-device, mobile, MEC, device–edge–cloud.
2. Capability: language/multimodal models, reasoning, context memory, planning, tool use.
3. Mechanism: quantization, distillation, KV management, decoding, partitioning, offloading, adaptation.
4. Constraint: latency, memory, energy, thermal state, connectivity, task quality, reproducibility.

For example: `("on-device" OR "edge") AND ("LLM" OR "agentic") AND ("quantization" OR "KV cache") AND ("energy" OR "latency")`.

These are reproducible query concepts. Historical database-specific queries, exact run dates, initial hit totals and screening/exclusion counts were not retained. Exhaustive coverage and PRISMA-style selection are therefore not claimed.

## Evidence support and deployment interpretation

Grades apply to the representative result, separately from publication status, model quality or runtime confidence. Public artifacts were inspected rather than independently rerun for every study.

| Grade | Replication support | Permitted use |
|---|---|---|
| E1 | Archival quantitative study with public implementation/evaluation artifacts supporting a rerun in principle | Result-specific comparison under the reported platform/workload conditions |
| E2 | Inspectable quantitative evidence with incomplete replication support, regardless of publication status | Qualified findings with missing conditions stated |
| E3 | Exploratory or incomplete summaries whose quantitative comparison cannot be reconstructed | Historical disclosure or open questions; no numerical deployment recommendation |

Direct task outcomes, language-model quality proxies and execution-efficiency measurements are another axis. Code availability does not turn a token-throughput result into complete-agent validation. Source-specific hardware, scorer, precision, context, power and failure boundaries must accompany transferred claims. Conditional deployment guidance does not establish a universally optimal combination.

## Living resource guide

The [README](../README.md) contains all five topic sections in one reading flow. Each paper has one canonical topic entry in the README; related topics cross-link rather than repeat the same record. Entries identify a deployment role, with a primary source and bounded interpretation. Superseded duplicates, placeholders, unsupported artifact associations and context-free performance rankings are removed during curation.

Both link-audit tools read `README.md` through `doc_sources.py`:

```sh
python3 survey/check_docs.py
python3 survey/verify_corpus.py --offline
python3 survey/verify_repos.py
```

Run these from the repository root. `check_docs.py` checks local targets, heading anchors, canonical arXiv placement and obsolete internal pointers. The literature audit extracts linked arXiv IDs, source titles, dates and withdrawal flags; title similarity below 0.62 is a human-review flag, not automatic exclusion. Repository checks use `git ls-remote HEAD`, which establishes reachability only.

Current results are written to `survey/current_corpus_manifest.json` and `survey/current_repos_manifest.json`. They record input files, generation time and live/cached mode. Offline mode never calls the network: it reuses cached records and reports any missing IDs/status. Cached metadata is not a current live revalidation. Running the literature audit without `--offline` queries the arXiv API. Inspect changed titles, versions and flags before accepting a refresh.

## Reproduce the frozen census

The manuscript uses immutable [commit `82b6110`](https://github.com/wangxb96/Awesome-EdgeAI/tree/82b6110), not the current README. Its archived README and manifests contain:

| Unit | Archived result |
|---|---:|
| Unique arXiv identifiers | 145 resolved, 0 unresolved, 0 withdrawal flags |
| Selected title-audited labels | 137; 9 low-similarity flags manually inspected |
| Records from 2023–2026 | 123/145 (84.8%) |
| Linked repository roots | 123 with a live HEAD at the audit date |

The 137 count is one selected label per labelled ID, not every Markdown occurrence. Repository roots and arXiv IDs are distinct units, not successive screening stages. Title flags reflect abbreviations, descriptive labels or version-title changes. Link deduplication does not resolve DOI/preprint identity at the study level. The manuscript bibliography also includes archival records and documentation outside this census.

Use a disposable checkout because the archived scripts rewrite their manifests:

```sh
git clone https://github.com/wangxb96/Awesome-EdgeAI.git
cd Awesome-EdgeAI
git checkout --detach 82b6110
python3 survey/verify_corpus.py --offline
python3 survey/verify_repos.py --offline
```

These commands replay saved metadata and HEAD status. They do not check today's source state or recover historical searches. Root files `corpus_manifest.json` and `repos_manifest.json` under `survey/` remain the frozen inputs; the living audits use separate `current_*` outputs.

## Measurement and historical limits

The [benchmark protocol](benchmark_protocol.md) specifies new request logs, task-episode linkage, failures, repetition and power boundaries. The [analysis helper](compute_ci.py) summarizes supplied request observations; it cannot infer complete-task success, certify logging completeness, validate power integration or recover missing inputs. Its analysis revision must be pinned separately from the census.

The manuscript's Supplement S1 retains incomplete historical server summaries as E3. Missing prompts, exact configurations, outputs, repetitions, failure denominators and power attribution cannot be restored. Those summaries establish neither mobile energy/performance nor a validated model ranking. Context-averaged p95 values are not pooled request percentiles, and an unverified board power/throughput ratio is not integrated system energy. No recommendation depends on those historical numbers.

## Artifact map

The [public supplement](supplement/[submission identifier removed]-supp.pdf), [source/build helper](supplement/README.md), [bibliographic database](references.bib) and all files in the [public inventory](public_artifacts.json) are distributed at [tag `supplement-2026-11-13`](https://github.com/wangxb96/Awesome-EdgeAI/tree/supplement-2026-11-13). Run `python3 survey/check_public_artifacts.py` for local file/dependency closure and add `--online` to check anonymous access to every inventoried file. This artifact edition does not replace the frozen census.

| Scientific purpose | Location |
|---|---|
| Task-to-resource and cross-tier argument | [Overview, placement and coupling diagrams](figures/README.md) |
| Published quantization scores | [QSpec replot](figures/quantization_fidelity.py), [reported source values](figures/quantization_fidelity_source.csv) |
| Analytical weight/KV illustration | [Source](figures/memory_budget.py), [CSV](figures/memory_budget_analytical.csv) |
| Communication/state boundaries | [Collaboration diagram](figures/collaboration_patterns.pdf) |
| Failure-aware request/episode accounting | [Workflow diagram](figures/workflow_evaluation.pdf), [benchmark protocol](benchmark_protocol.md) |
| Permissioned assistant example | [Editable source](figures/assistant_workflow.py), [diagram](figures/assistant_workflow.pdf) |
| Request-log format and expected output | [Clearly labelled synthetic fixture](examples/README.md), not a measurement archive |
| Primary model and runtime artifacts | [Models and runtimes](../README.md#models-and-runtimes) |

The main manuscript retains the scientific argument, representative evidence and acceptance criteria. Extended identifiers, configuration locators, reporting schemas and prospective traces are supplementary detail. Missing historical search and measurement records remain limitations rather than reconstructed evidence.
