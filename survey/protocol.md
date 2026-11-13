# Survey Protocol and Reproducible Corpus — [submission identifier removed]

This document is the executable methodology companion to

> X. Wang, Q. Li, and W. Jia, *Cognitive Edge Computing: A Survey on Optimizing
> Large Language Models and Autonomous Agents for the Edge*, IEEE Transactions
> on Mobile Computing, revision, manuscript **[submission identifier removed]**.

It is the canonical statement of the search, screening, verification, and
evidence-grading protocol summarized in Sec. *Survey Methodology and Evidence
Grading* of the revised manuscript. **This repository
(`wangxb96/Awesome-EdgeAI`) is the official artifact repository of that
manuscript.** Protocol revision 11 November 2026. The dated census uses the
archived README and manifests at immutable Git commit `82b6110`. This protocol
and the future benchmark helper are updated separately; the older protocol at
that census commit is not the current evidence-grading policy.

---

## 1. Scope and scientific question

The survey asks a *capability-under-budget* question: which combinations of
model-, runtime-, and agent-loop optimizations preserve reasoning quality on
resource-constrained edge and mobile devices while remaining within latency,
energy, memory, thermal, privacy, and wireless-bandwidth budgets?

Included: reasoning-capable LLM/SLM/VLM deployment at the edge; on-device
inference and training; compression and efficient architectures; runtime,
KV-cache, scheduling, offloading, and device–edge–cloud serving; autonomous
agents, tool use, and multi-agent coordination under intermittent
connectivity; measurement, benchmarking, energy, security/TEE, and 6G/token
communication aspects tied to those deployments; and foundational runtime/TinyML mechanisms whose transfer to that setting
is explicitly bounded. Inclusion of a server mechanism does not establish
mobile performance.

Excluded: conventional edge perception without LLM/agent components;
cloud-only training with no edge-facing deployment decision; isolated hardware
design that does not interact with model serving or data/context management;
non-technical product announcements; and withdrawn preprints.

---

## 2. Sources and search window

| Dimension | Value |
|---|---|
| Bibliographic sources | arXiv, IEEE Xplore, ACM Digital Library, Google Scholar (forward/backward snowballing), SpringerLink and publisher DOI landing pages for archival verification |
| Artifact sources | Official GitHub organizations (project repositories, model cards, benchmark suites), MLCommons, vendor technical reports |
| Coverage window | 2015–10 Nov 2026, with emphasis on 2023–2026 |
| Census snapshot date | 11 Nov 2026; frozen README/manifests at `82b6110`; literature coverage ends 10 Nov |
| Language | English-language publications and technical reports |

The four query families below describe the search concepts (tier × model family
× optimization lever × evaluation), with backward/forward snowballing from the
seed set and recent surveys. The database-specific query logs, exact run dates
for each source, initial hit totals, and screening/exclusion counts were not
retained. The strings below are therefore a reproducible search template, not a
verbatim record of every historical database query:

1. **Deployment tier:** `("edge" OR "on-device" OR "mobile" OR "MEC" OR "edge-cloud" OR "device-edge-cloud")`
2. **Model / agent family:** `("LLM" OR "large language model" OR "SLM" OR "VLM" OR "multimodal" OR "agentic" OR "autonomous agent" OR "tool use" OR "function calling")`
3. **Optimization lever:** `("quantization" OR "pruning" OR "distillation" OR "sparsity" OR "KV cache" OR "speculative decoding" OR "offloading" OR "partitioning" OR "serving" OR "LoRA" OR "on-device training" OR "federated")`
4. **Evaluation / constraint:** `("latency" OR "energy-per-token" OR "J/token" OR "thermal" OR "memory footprint" OR "p95" OR "reasoning fidelity" OR "benchmark" OR "reproducibility")`

Example composite string:
`("on-device" OR "edge") AND ("LLM" OR "agentic") AND ("quantization" OR "KV cache") AND ("energy" OR "latency")`.

---

## 3. Inclusion / exclusion and evidence grading

**Inclusion.** (I1) The work addresses constrained LLM/agent execution or a
mechanism transferable to that setting, with server-to-device limits stated. (I2) It
provides a method, system, measurement, or critical synthesis with enough
detail to be classified. (I3) Its primary source resolves to an authoritative
record (arXiv abstract page, publisher DOI, or official repository HEAD).

**Exclusion.** Conventional perception without an LLM/agent connection, cloud-only
training, unverifiable/non-technical claims, withdrawn preprints, and unrepaired
source placeholders. Census deduplication is by arXiv ID and repository root; it
is not study-level DOI/preprint deduplication or a recovered screening flow.

**Evidence grades** apply to the representative result used, not a whole method
family. Publication status is recorded separately. Public artifacts were
inspected, not independently rerun for every result.

| Grade | Definition | Permitted interpretation |
|---|---|---|
| **E1** | Archival quantitative study with public implementation/evaluation artifacts supporting a rerun in principle | Result-specific comparisons within the reported platform/workload boundary |
| **E2** | Inspectable quantitative study with incomplete replication support, irrespective of publication status | Qualified findings with the missing replication conditions stated |
| **E3** | Exploratory claims or incomplete summaries whose quantitative comparison cannot be reconstructed | Open questions or explicitly historical records; no numerical deployment recommendation or validated superiority claim |

The main study-level matrix records the result, hardware, capability proxy,
artifact boundary, and qualified implication. Conflicting measurements retain
their different conditions; they are not averaged or resolved by a blanket
family grade. Deployment rules labelled as heuristics are not empirical
optimality claims, and research evidence grade is not runtime confidence in an
agent's context. The historical server summaries are E3 and do not derive the
survey's recommendations.

Withdrawn preprints are excluded by construction (an automated check on the
arXiv `comment` and abstract fields), e.g. the withdrawn *SplitLite* preprint
was removed from the 2026 refresh.

---

## 4. Screening and verification workflow

The final corpus is a **machine-checkable census**, not an unverifiable
narrative count. We deliberately publish the extraction and verification code
rather than a claimed database-hit total that cannot be independently
recomputed.

```
README.md (living curated corpus)
   │
   ├─ survey/verify_corpus.py  ──► arXiv Atom API ──► survey/corpus_manifest.json
   │     • extract all arXiv IDs (Markdown links + bare URLs)
   │     • resolve authoritative title / date / withdrawal status
   │     • flag title↔ID mismatches (SequenceMatcher < 0.62)
   │     • emit year and topic-section distributions
   │
   └─ survey/verify_repos.py   ──► git ls-remote HEAD ──► survey/repos_manifest.json
         • extract all github.com/<owner>/<repo> links
         • confirm every linked repository resolves to a live HEAD
```

| Stage | Rule | Count (11 Nov 2026 snapshot) |
|---|---|---|
| S0 Candidate records | arXiv identifiers and GitHub repositories linked in `README.md` (not all papers in the manuscript bibliography) | 145 arXiv IDs + 123 repositories |
| S1 Deduplication | Unique identifiers / repository roots | 145 + 123 |
| S2 Primary-source resolution | arXiv Atom API; `git ls-remote` | 145/145 resolved; 123/123 live |
| S3 Withdrawn / dead-link exclusion | Automated withdrawal flag; repository HEAD check | 0 withdrawn; 0 dead repositories (one historical dead link repaired) |
| S4 Title↔ID audit | Compare README link label to authoritative title | 9 low-similarity labels among 137 labelled links; all inspected and attributed to short names, descriptive bibliographic labels, or version-title changes |
| S5 Retained audited links | Distinct identifiers and repository roots, not evidence-graded study screening | **145 arXiv records + 123 verified repositories** |

**Corpus composition (from `corpus_manifest.json`):** 2015: 1, 2016: 2,
2017: 2, 2018: 2, 2020: 5, 2021: 2, 2022: 8, 2023: 11, 2024: 19, 2025: 37,
2026: 56. Hence **123/145 (84.8%) are from 2023–2026**, consistent with the
manuscript's emphasis on recent work while retaining foundational references.
The manuscript BibTeX file (`references.bib`, 260 records) also includes
archival venue records, standards, and hardware documentation outside this
arXiv/repository census. The 145-record census is therefore not the total
number of studies cited or screened for the manuscript.

---

## 5. Reproducing the census

The reproducible object is the archived census at commit `82b6110`, not the
changing default branch. In a disposable checkout (Python 3.9+, standard library
only; `git` required):

```bash
git clone https://github.com/wangxb96/Awesome-EdgeAI.git
cd Awesome-EdgeAI
git checkout --detach 82b6110
python3 survey/verify_corpus.py --offline
python3 survey/verify_repos.py --offline
```

These offline commands recompute counts from the frozen README and saved API/HEAD
results. The scripts rewrite their output manifests, so use a disposable checkout
if preserving a clean working tree matters. For a strictly read-only check, load
the archived script, README and manifest contents using `git show` and intercept
output writes in memory. The outputs are 145 identifiers, 137 labelled IDs,
9 title flags, and 123 repository roots with archived OK status. They do not
revalidate today's source state or replay historical searches. The 137 count is
one selected Markdown label per unique arXiv ID, not all label occurrences.

Running the scripts without `--offline` performs a **new live audit** and may
change counts and manifests; record its date and commit separately. A live HEAD
checks reachability, not official ownership, source-paper association, or
experimental replicability. The 260-entry bibliography count is a separate
manuscript-input count, not a value computed by these link-audit scripts.

---

## 6. Artifact ↔ manuscript crosswalk

| Manuscript element | Artifact in this repository |
|---|---|
| Source/query concepts and retained-census limitations | `survey/protocol.md` (this file) |
| E1/E2/E3 grading definitions | Section 3 of this protocol and the main study-level matrix |
| Frozen retained-link counts | `survey/corpus_manifest.json`, `survey/repos_manifest.json` at `82b6110` |
| Topic-organized literature, distinct from the frozen census | `README.md` → model, agent, hardware, application and survey sections |
| On-device model families and 30+ official tool repositories | `README.md` → *On-Device Model Families & Tool Repository Index* |
| Exploratory historical benchmark limitations | `README.md` → *[submission identifier removed] Official Repository* |
| Literature-link and repository-link audits | `survey/verify_corpus.py`, `survey/verify_repos.py` |
| Benchmark reporting protocol, energy boundary, failed-batch policy | `survey/benchmark_protocol.md` |
| Failure denominators, conditional summaries, pooled request p95, optional repetition intervals | `survey/compute_ci.py` |

---

## 7. Limitations of the protocol

This is a reproducibility-oriented *census* protocol, not a PRISMA-style
systematic review. The database-specific query logs, exact per-source search
dates, initial hit totals, and counts at the screening/exclusion stages were
not retained, so the scripts cannot reproduce the original selection process
or selection-bias estimates. Search-engine result counts are also unstable
across time, personalization, and indexing. The audit instead verifies the
retained corpus: every retained arXiv item is resolvable to a primary source,
and archived link counts can be recomputed at the pinned snapshot. Title
similarity is a heuristic that flags abbreviations and version renames for
human inspection rather than an automatic exclusion rule. Publication/source class alone does not establish evidence strength; incomplete
comparisons remain E3 regardless of source class and cannot establish
quantitative superiority. This limitation is stated explicitly in the manuscript's
*Threats to Validity* section.
