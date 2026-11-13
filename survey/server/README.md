# Historical server reference and request-log reanalysis

The reported four-RTX-4090-D experiments remain part of the survey's supplementary record. This archive separates what can be recalculated from what cannot be inferred about model capability or mobile deployment. It contains recovered empirical records, not the [synthetic helper fixture](../examples/README.md).

## Source provenance and reading route

The original [focused survey repository](https://github.com/wangxb96/cognitive-edge-llm-agent-survey/tree/a858d5349d82afa1791d00091baf4a288f07ccd2) retains benchmark script snapshots. At that inspected revision, its [runner](https://github.com/wangxb96/cognitive-edge-llm-agent-survey/blob/a858d5349d82afa1791d00091baf4a288f07ccd2/assets/reproducibility/scripts/run_ollama_server_suite.py) and [selector](https://github.com/wangxb96/cognitive-edge-llm-agent-survey/blob/a858d5349d82afa1791d00091baf4a288f07ccd2/assets/reproducibility/scripts/rebuild_clean_server_artifacts.py) are byte-identical to the copies here. This establishes public code provenance, not the exact options used during execution. The older repository contains no request CSVs or summary JSONs at that revision; the recovered data and supported analysis entry point are supplied below.

Read the campaign definitions before comparing plotted values. Use the offline analysis command to recalculate the record. Consult the metric boundaries before interpreting a score, TTFT label or energy value. The older README's synchronized-benchmark and capability-ranking interpretation is not used as evidence for this record.

| Resource | Contents |
|---|---|
| [Archive inventory](archive_inventory.json) | All thirteen CSV/summary pairs, original and published hashes; CSV line endings alone are normalized from CRLF to LF |
| [Campaign ledger](campaign_ledger.csv) | Every recovered campaign, its logged denominator and errors; pilots and connectivity probes are retained separately |
| [Task roster](task_roster.json) | Ten prompts and reference answers from the surviving source code; exact historical HTTP payloads were not archived |
| [Recovered runner](run_ollama_server_suite.py) / [historical selector](rebuild_clean_server_artifacts.py) / [legacy plotting source](plot_benchmark_results.py) | Scoring, request timing, endpoint power and the earlier latest-complete-per-model selection procedure; source defaults do not establish execution-time options |
| [Reanalysis script](analyze_archived.py) | Standard-library recalculation, score/percentile checks, archive hashes and derived CSV generation |
| [Per-campaign results](reanalysis_by_run.csv) / [per-context results](reanalysis_by_context.csv) | All declared primary/follow-up requests, descriptive dispersion, score audit, timing and energy proxies |
| [Checks](reanalysis_checks.json) | Counts, scorer replay and interpretation boundaries |
| [Results figure](../figures/server_results_reanalysis.pdf) / [measurement checks](../figures/server_measurement_boundaries.pdf) | Vector figures in Supplement S1; [drawing code](../figures/server_reanalysis.py) |

## Which observations are used

The archive has **2350 logged requests and 189 logged errors across thirteen campaigns**. These counts are a recovered-file census, not a pooled system failure rate. Warm-up calls were not logged and a process interruption could prevent rows from being saved. Operational completeness therefore cannot be established.

The complete main campaign (`1777121713`) has **1200 requests**: five recorded model tags, four configured context ceilings, ten fixed tasks and six repeat indices per cell. Every row is retained. The separate Qwen3 8B follow-up (`1777295563`) has **200 requests** and five repeat indices per cell. Its values are not pooled with the main campaign. Both have zero recorded transport errors, while some stored responses are empty. Other campaigns remain in the ledger as pilots or connectivity probes; they are not silently discarded or combined with the full campaign.

The submitted aggregate used four main-campaign model rows and the later Qwen row; the older [plot aggregates](archived/legacy_plot_aggregates.csv) used the earlier Qwen campaign. This explains the old plot/table discrepancy without making different runs equivalent. The supplementary table preserves the submitted values and identifies this provenance; the new plots use request-level results with campaign labels.

## Reproduce the record analysis

From the repository root, using Python 3.9+ with the standard library:

```sh
python3 survey/server/analyze_archived.py
```

The historical runner/selector/plotting files are provenance snapshots assuming their original project layout; only the following analysis is the supported offline entry point. This checks archive hashes, structural uniqueness, every recorded score and saved successful-summary percentiles, then regenerates the ledger, task roster, results CSVs and checks. It does **not** invoke a model or make network calls. To redraw the two figures with the [pinned packages](../requirements-figures.txt):

```sh
python3 -m pip install -r survey/requirements-figures.txt
python3 survey/figures/server_reanalysis.py
```

## What each metric means

- **Score:** the historical script lowercases/collapses whitespace and checks whether the reference answer is a substring of the prediction. It is answer containment, not strict exact match or semantic reasoning accuracy. A separately labelled exact-string audit tests normalized equality and diagnoses formatting sensitivity.
- **Timing:** wall-request median and pooled, linearly interpolated p95 are recalculated from individual requests. They differ from an arithmetic mean of context-specific p95 values. The field named `ttft_ms` is Ollama prompt-evaluation duration, not measured user-facing TTFT.
- **Variability:** request SD and sample SD across repeat-index aggregates are descriptive. The latter are supplied for containment, strict-string score, mean wall time, mean decode rate and endpoint-proxy J/token (sum of proxy joules / sum of output tokens within each index). Zero repeat SD is not certainty about general capability, and proxy SD does not estimate power-measurement bias. Repeat indices are technical repeated requests on fixed tasks, not independently restarted sessions; no population confidence interval or significance test is inferred.
- **Power:** `energy_j` is an endpoint mean of summed visible-GPU power multiplied by wall-request duration. Summed proxies divided by summed output tokens reproduce the energy-per-token proxy. Two power samples do not establish integrated or attributed energy, board identities, sampler/server alignment, or host/radio/cooling cost.
- **Context:** 512–4096 are configured ceilings. Actual recorded main-campaign prompt counts are 16–95 tokens; this is not a long-context retention stress test.
- **Output capture:** Qwen's main campaign has 240 empty response fields, all with 128 decoded tokens. The follow-up has 156 empty fields and 44 exact-answer fields; 160/200 rows have 256 decoded tokens. Stop reasons and separate reasoning-channel contents were not retained. The difference does not establish a controlled token-budget intervention or intrinsic reasoning gain.

Exact checkpoint/template digests, per-model precision, runtime/driver revisions, placement, command-line settings, complete response envelopes and power traces remain unrecorded. The recovered record supports **E2 descriptive request-log reanalysis**. Unsupported legacy matched-capability/edge interpretations remain **E3** and are withdrawn. The server results do not establish a device ranking, an energy optimum or mobile feasibility. Published device studies in the main survey retain their own source-specific boundaries.

[Back to the guide](../../README.md#research-artifacts-and-figures)
