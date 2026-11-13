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


The [cited-reference index](cited_reference_index.json) identifies all 96 references in the compiled main paper and the 30 used by the supplement (a subset of those 96). These include primary studies, surveys and documentation; they are not a count of independent experiments or all eligible literature. The 275-entry bibliographic database is broader, and the seventeen matrix records are purposive synthesis examples.

## Evidence support and deployment interpretation

Grades apply to the representative result and combine publication maturity with artifact support; E1 requires archival publication. Publication type is also displayed explicitly. Grades do not measure model quality or runtime confidence. Public artifacts were inspected rather than independently rerun for every study.

| Grade | Publication and replication basis | Permitted use |
|---|---|---|
| E1 | Archival quantitative finding with result-linked implementation, workload/configuration, comparator where needed, and scoring/measurement route supporting a rerun in principle | Result-specific comparison under the reported platform/workload conditions |
| E2 | Inspectable quantitative evidence with incomplete replication support, regardless of publication status | Qualified findings with missing conditions stated |
| E3 | Exploratory or incomplete summaries whose quantitative comparison cannot be reconstructed | Historical disclosure or open questions; no numerical deployment recommendation |

The [result-specific ledger](evidence_ledger.json) follows the seventeen examples in the manuscript matrices. Selection purposively spans controls, execution tiers and outcome types; it is neither exhaustive nor a reconstructed screening flow. Each entry records the source version, result being graded, dated artifact snapshot, visible execution/scoring components, missing or unverified links, independent-rerun status and mobile-transfer limit. Composite rows take the least-established replication path among their material outcomes. A generic engine benchmark does not identify the workload or comparator for a specific published result. Current HEADs are inspection snapshots, not inferred historical experiment revisions. FastTTS uses its paper-linked Zenodo evaluation archive; inference/export or throughput demos alone do not establish a matching task-score reproduction path.

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

The living repository extractor case-folds owner/repository roots before deduplication and cache lookup; it does not resolve redirects, renamed projects or aliases. Current results are written to `survey/current_corpus_manifest.json` and `survey/current_repos_manifest.json`. They record input files, generation time and live/cached mode. Offline mode never calls the network: it reuses cached records and reports any missing IDs/status. Cached metadata is not a current live revalidation. Running the literature audit without `--offline` queries the arXiv API. Inspect changed titles, versions and flags before accepting a refresh.

## Reproduce the frozen census

The manuscript uses immutable [commit `82b6110`](https://github.com/wangxb96/Awesome-EdgeAI/tree/82b6110), not the current README. Its archived README and manifests contain:

| Unit | Archived result |
|---|---:|
| Unique arXiv identifiers | 145 resolved, 0 unresolved, 0 withdrawal flags |
| Selected title-audited labels | 137; 9 low-similarity flags manually inspected |
| Records from 2023–2026 | 123/145 (84.8%) |
| Distinct recorded repository URL-root strings | 123 with a live HEAD at the audit date |
| Case-folded repository roots (separate recount) | 122; capitalization variants collapse, redirects/aliases remain unresolved |

The 137 count is one selected label per labelled ID, not every Markdown occurrence. The archived 123 count is case-sensitive; `google-ai-edge/LiteRT-LM` and `google-ai-edge/litert-lm` are two recorded spellings of one normalized root. The frozen manifest remains unchanged. Repository URL strings and arXiv IDs are distinct units, not successive screening stages. Title flags reflect abbreviations, descriptive labels or version-title changes. Link deduplication does not resolve DOI/preprint identity at the study level. The manuscript bibliography also includes archival records and documentation outside this census.

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

Supplement S1 preserves the original server aggregate and provides a [recovered-record reanalysis](server/README.md). Thirteen campaigns contain 2350 logged requests and 189 recorded errors. The complete five-model campaign (1200 requests) and later Qwen follow-up (200 requests) remain separate. Scorer replay and raw-request percentiles are recalculable, giving E2 descriptive support. The old score was answer containment, actual prompts were short, an old TTFT field was prompt-evaluation duration, and energy used two endpoint-power samples. Exact execution settings remain unrecorded; unsupported E3 matched-capability/edge claims are withdrawn. The synthetic format example is separate from these observations. No recommendation depends on a numerical server ranking.

## Artifact map

The [public supplement PDF](supplement/cognitive_edge_supplement.pdf), [bibliographic database](references.bib) and files in the [public inventory](public_artifacts.json) are distributed at [tag `public-artifacts-2026-11-13-v5`](https://github.com/wangxb96/Awesome-EdgeAI/tree/public-artifacts-2026-11-13-v5). Editable manuscript sources and TeX build inputs are intentionally excluded. Run `python3 survey/check_public_artifacts.py` for local file closure and add `--online` to check anonymous access to every inventoried file. This artifact edition does not replace the frozen census.

| Scientific purpose | Location |
|---|---|
| Historical empirical record | [Archive, failure ledger and reproducible analysis](server/README.md) |
| Task-to-resource and cross-tier argument | [Overview, placement and coupling diagrams](figures/README.md) |
| Published quantization scores | [QSpec replot](figures/quantization_fidelity.py), [reported source values](figures/quantization_fidelity_source.csv) |
| Analytical weight/KV illustration | [Source](figures/memory_budget.py), [CSV](figures/memory_budget_analytical.csv) |
| Communication/state boundaries | [Collaboration diagram](figures/collaboration_patterns.pdf) |
| Failure-aware request/episode accounting | [Workflow diagram](figures/workflow_evaluation.pdf), [benchmark protocol](benchmark_protocol.md) |
| Permissioned assistant example | [Editable source](figures/assistant_workflow.py), [diagram](figures/assistant_workflow.pdf) |
| Request-log format and expected output | [Clearly labelled synthetic fixture](examples/README.md), not a measurement archive |
| Primary model and runtime artifacts | [Models and runtimes](../README.md#models-and-runtimes) |

The main manuscript retains the scientific argument, representative evidence and acceptance criteria. Extended identifiers, configuration locators, reporting schemas and prospective traces are supplementary detail. Missing historical search and measurement records remain limitations rather than reconstructed evidence.
