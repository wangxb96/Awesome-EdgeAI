# Historical server reference and request-log reanalysis

The reported four-RTX-4090-D experiments remain part of the survey's supplementary record. This archive presents recalculable server observations with their source-specific interpretation and device-transfer boundaries. The recovered empirical records are catalogued separately from the [synthetic helper fixture](../examples/README.md).

## Source provenance and reading route

The original [focused survey repository](https://github.com/wangxb96/cognitive-edge-llm-agent-survey/tree/a858d5349d82afa1791d00091baf4a288f07ccd2) retains benchmark script snapshots. At that inspected revision, its [runner](https://github.com/wangxb96/cognitive-edge-llm-agent-survey/blob/a858d5349d82afa1791d00091baf4a288f07ccd2/assets/reproducibility/scripts/run_ollama_server_suite.py) and [selector](https://github.com/wangxb96/cognitive-edge-llm-agent-survey/blob/a858d5349d82afa1791d00091baf4a288f07ccd2/assets/reproducibility/scripts/rebuild_clean_server_artifacts.py) have the same parsed Python structure after generic device-label normalization. The copies here intentionally generalize device defaults and plotting labels. The inventory preserves both source and published hashes; data, scoring, timing and aggregation nodes are unchanged. This documents public code provenance; recovered campaign fields define the execution evidence reported below. The recovered data and supported analysis entry point are supplied in this archive.

Read the campaign definitions before comparing plotted values. Use the offline analysis command to recalculate the record. Consult the metric boundaries before interpreting a score, TTFT label or energy value. The current interpretation follows the archived request-level calculations and campaign provenance.

| Resource | Contents |
|---|---|
| [Archive inventory](archive_inventory.json) | All thirteen CSV/summary pairs, original and published hashes; CSV line endings alone are normalized from CRLF to LF |
| [Campaign ledger](campaign_ledger.csv) | Every recovered campaign, its logged denominator and errors; pilots and connectivity probes are retained separately |
| [Task roster](task_roster.json) | Ten prompts and reference answers recovered from the source code, alongside the request-level observations |
| [Recovered runner](run_ollama_server_suite.py) / [historical selector](rebuild_clean_server_artifacts.py) / [legacy plotting source](plot_benchmark_results.py) | Scoring, request timing, endpoint power and the earlier latest-complete-per-model selection procedure; source code documents defaults and control flow; S1 maps archived execution fields to their evidence role |
| [Reanalysis script](analyze_archived.py) | Standard-library recalculation, score/percentile checks, archive hashes and derived CSV generation |
| [Per-campaign results](reanalysis_by_run.csv) / [per-context results](reanalysis_by_context.csv) | All declared primary/follow-up requests, descriptive dispersion, score audit, timing and energy proxies |
| [Checks](reanalysis_checks.json) | Counts, scorer replay and interpretation boundaries |
| [Results figure](../figures/server_results_reanalysis.pdf) / [measurement checks](../figures/server_measurement_boundaries.pdf) | Vector figures in Supplement S1; [drawing code](../figures/server_reanalysis.py) |

## Which observations are used

The archive has **2350 logged requests and 189 logged errors across thirteen campaigns**. These counts are an archive-level census; campaign-specific denominators are retained separately. Warm-up calls and interrupted processes lie outside the saved rows, and the reported denominator is the recovered-file set.

The complete main campaign (`1777121713`) has **1200 requests**: five recorded model tags, four configured context ceilings, ten fixed tasks and six repeat indices per cell. Every row is retained. The separate Qwen3 8B follow-up (`1777295563`) has **200 requests** and five repeat indices per cell. The follow-up is reported under its own campaign identifier and denominator. Both have zero recorded transport errors, while some stored responses are empty. Other campaigns remain separately labeled in the ledger as pilot runs or connectivity probes.

The submitted aggregate used four main-campaign model rows and the later Qwen row; the older [plot aggregates](archived/legacy_plot_aggregates.csv) used the earlier Qwen campaign. This campaign provenance explains the plot/table discrepancy; each run remains labeled by its own campaign. The supplementary table preserves the submitted values and identifies this provenance; the new plots use request-level results with campaign labels.

## Reproduce the record analysis

From the repository root, using Python 3.9+ with the standard library:

```sh
python3 survey/server/analyze_archived.py
```

The historical runner/selector/plotting files are provenance snapshots assuming their original project layout; only the following analysis is the supported offline entry point. This checks archive hashes, structural uniqueness, every recorded score and saved successful-summary percentiles, then regenerates the ledger, task roster, results CSVs and checks. It runs offline and leaves model execution and network state untouched. To redraw the two figures with the [pinned packages](../requirements-figures.txt):

```sh
python3 -m pip install -r survey/requirements-figures.txt
python3 survey/figures/server_reanalysis.py
```

## What each metric means

- **Score:** the historical script lowercases/collapses whitespace and checks whether the reference answer is a substring of the prediction. The score is normalized answer containment. Strict exact-string equality and semantic reasoning accuracy are separate outcome measures. A separately labelled exact-string audit tests normalized equality and diagnoses formatting sensitivity.
- **Timing:** wall-request median and pooled, linearly interpolated p95 are recalculated from individual requests. The pooled-request percentile and mean of context-specific p95 values are distinct summaries. The field named `ttft_ms` records Ollama prompt-evaluation duration; user-facing TTFT uses a separately defined streaming boundary.
- **Variability:** request SD and sample SD across repeat-index aggregates are descriptive. The latter are supplied for containment, strict-string score, mean wall time, mean decode rate and endpoint-proxy J/token (sum of proxy joules / sum of output tokens within each index). Zero repeat SD describes identical aggregate scores on these fixed tasks; proxy SD describes variation in the endpoint-based estimates. Repeat indices are technical repeated requests on fixed tasks. Population confidence intervals and significance tests use independent campaign-level units.
- **Power:** `energy_j` is an endpoint mean of summed visible-GPU power multiplied by wall-request duration. Summed proxies divided by summed output tokens reproduce the energy-per-token proxy. The two endpoint samples define an endpoint-based proxy. Integrated energy and host/radio/cooling attribution use timestamped system-level instrumentation, with board identity and sampler alignment recorded.
- **Context:** 512–4096 are configured context ceilings; actual recorded main-campaign prompts contain 16–95 tokens. Long-context retention is a separate workload dimension.
- **Output capture:** Qwen's main campaign has 240 empty response fields, all with 128 decoded tokens. The follow-up has 156 empty fields and 44 exact-answer fields; 160/200 rows have 256 decoded tokens. Stop reasons and separate reasoning-channel contents are outside the recovered fields. The campaign-specific outputs are reported without attributing their difference to token budget or intrinsic reasoning.

The recovered archive supports **E2 descriptive request-log reanalysis**. Its source-specific execution fields are catalogued in the inventory and S1. The current interpretation uses server outcomes descriptively; device recommendations draw on published mobile/edge studies with their own source-specific conditions.

[Back to the guide](../../README.md#research-artifacts-and-figures)
