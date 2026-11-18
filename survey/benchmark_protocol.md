# Benchmark reporting protocol

Reporting contract established 11 November 2026; public example and file navigation updated 13 November 2026. This document defines the reporting contract for future runs. Historical server measurements and their reanalysis are documented in [Supplement S1 and the server archive](server/README.md). Pin the Git commit containing this protocol and helper as the analysis revision. Census snapshot `3dd39bd` anchors the frozen link manifests; this document and its versioned helper define the current benchmark reporting policy.

## 1. Historical boundary

Supplement S1 preserves the reported `4x RTX 4090 D` summaries and now provides a [recovered historical archive](server/README.md). Its thirteen campaigns contain 2350 logged requests with 189 recorded errors; the main 1200-request campaign and 200-request Qwen follow-up are analyzed separately. The archive supports reproducible score replay and raw-request summaries as E2 descriptive server evidence. Configuration fields are interpreted from the recovered files, with source-specific values and measurement boundaries documented in S1. Edge recommendations are grounded in cited device studies and their evaluated conditions.

The historical score is normalized answer containment; strict exact-string equality is reported as a separate audit. Configured context ceilings and logged input lengths are separate quantities. The archived `ttft_ms` field records prompt-evaluation duration; user-facing TTFT uses a request/stream boundary. Historical J/token is derived from two endpoint GPU-power readings and wall-request duration; integrated power traces and aligned system-level attribution define a separate energy measurement. Repeat-index variability is descriptive for these fixed task cells; independently restarted campaigns define a separate replication unit. These historical definitions are documented separately from the prospective contract below. The synthetic helper fixture supplies no historical observation.

## 2. Request log and provenance

Write one row per **attempted task request**, including failed and interrupted
requests, to your newly collected `per_trial.csv`. A [header-only template](examples/request_log_template.csv) is supplied; it contains no observations. This user-created log is not a distributed measurement file. Retries receive distinct request IDs. Counts are
attempt counts, not unique user goals or completed multi-request workflows.
Log every attempt before executing it and finalize its status even when a batch
is interrupted. Include the expected task/batch roster in the campaign manifest
so a separate audit can detect omitted attempts. Compare logged requests with the expected campaign roster as an independent completeness audit.

| Fields | Definition |
|---|---|
| `run_id`, `request_id` | Campaign ID and request ID unique within that campaign. Required for counting. |
| `model`, `context`, `task` | Exact model tag, nominal context budget, and suite task ID. `model` and `context` are required grouping keys. |
| `episode_id`, `parent_request_id` | Optional identifiers linking requests to a separate episode ledger for task-level aggregation and goal-completion scoring. |
| `repetition_id`, `batch_id` | Independent experimental repetition and batch within it. `batch_id` is required. A repetition can contain multiple dependent requests/batches. |
| `status`, `failure_reason` | `ok`, `timeout`, `empty`, `unparseable`, or `incomplete`; record the failure reason. `ok` means a parseable completed request, not necessarily a correct answer. |
| `correct` | `0` or `1` under the fixed scorer for `ok` requests. Failed requests contribute zero to unconditional task-attempt success. |
| `request_latency_ms`, `ttft_ms` | Elapsed request time from declared submission boundary to completion/termination, and time to first token. Preserve timestamps and timeout thresholds in the trace. A timeout elapsed time is a censoring bound. |
| `generation_time_s`, `generated_tokens`, `tok_per_s` | Declared generation window, generated output-token count, and rate for that same window. Do not use model-specific token counts as a matched-output guarantee. |
| `prompt_id`, `prompt_text`, `raw_output`, `normalization_rule` | Prompt instance, unmodified output, and exact normalization/scoring rule; large values may instead use immutable file references. |
| `checkpoint`, `chat_template_hash`, `precision` | Exact checkpoint/template revisions, weight/activation/KV bit widths and quantization configuration. |
| `runtime_version`, `driver_version`, `seed`, `decoding_parameters` | Runtime/driver revisions, random seed, and full sampling/decoding settings. |
| `generation_budget`, `actual_input_tokens`, `actual_output_tokens` | Generation limit and tokenized lengths. Disclose truncation and how budgets are made comparable across tokenizers. |
| `device_ids`, `power_boundary`, `power_trace_ref` | Active devices, measured components, and immutable timestamped power trace. Identify all GPU boards and any shared/overlapping activity. |
| `power_sample_interval_s`, `power_window_start`, `power_window_end`, `power_integration_method`, `idle_policy` | Sampling interval, clock-aligned integration window, numerical integration method, and idle inclusion/subtraction. |
| `gpu_board_power_w`, `integrated_energy_j`, `j_per_tok` | Reported aggregate power, integrated energy in the stated boundary, and energy divided by tokens over the identical window. Unknown values remain unknown. |

Record fields absent from collection as `UNRECORDED`; keep unknown values explicit rather than substituting defaults. Classify runs using their recorded scientific provenance and the declared use of the result. Missing structural counting keys, unknown
statuses, duplicate request IDs, or missing binary correctness for completed
requests are rejected. Archive the CSV, roster, prompts, scorer, traces, code,
configuration, and their hashes.

A task episode may contain several model/tool requests, retries, and fallback actions.
Keep its success criterion, start/termination timestamps, deadline, and energy
boundary in a separate episode ledger linked to the request traces. A new
user-level restart is a new episode. Keep episode rows in the episode ledger and constituent requests in the request log. Score episode success and deadline satisfaction from the task-level record, and retain costs for every terminal outcome.

## 3. Failures, denominators, and latency

Report attempted/completed/failed task attempts and batches, with status and
failure counts. A completed batch has `status=ok` for every logged request in
that batch; an external roster audit is needed to establish that no request is
missing. Include every failed batch in unconditional denominators. The helper
reports unconditional task-attempt success as
`number(status=ok and correct=1) / number(all logged attempts)`.

The helper separately labels numeric means over **status-ok requests** as
conditional. Report conditional means beside unconditional success so readers can distinguish the status-ok subset from all attempts. If a study reports a complete-batch subset, label its denominator and selection rule explicitly and compute it as a separate analysis.

Compute empirical request p95 from individual `request_latency_ms` values with
the nearest-rank rule `sorted(x)[ceil(0.95*n)-1]`. Calculate pooled p95 from individual request latencies, not from averaged batch percentiles. The helper provides both a status-ok p95 and a
pooled percentile of **recorded elapsed times across all statuses**. It includes
observed counts and missing-time counts. Interpret the pooled recorded-time percentile with termination thresholds and failure rates, because timeout/incomplete durations are censoring bounds. Use a separately justified censored-latency analysis for unconditional completion-time inference. Construct percentile intervals with an independent resampling unit suited to the study design.

## 4. Power integration and energy scope

State whether the boundary is one GPU board, the sum of specified GPU boards,
a SoC, or the whole system. For sampled power, integrate the timestamped trace
across that same boundary and window, then divide by the tokens produced inside
it. Specify allocation of shared-device energy, overlap, prefill, queueing,
radio/host activity, and idle subtraction. Sum only measured boards and account for overlapping activity within the declared boundary.

`mean_power_w / tok_per_s` approximates J/token when both terms share devices, interval, and aggregation. Keep device, time window, and aggregation aligned when reporting energy. GPU-only figures characterize the GPU-board boundary; include host CPU/DRAM, PSU, storage, network, and cooling measurements for system-level efficiency. The helper **summarizes supplied numeric fields**; pair it with timestamped instrumentation, attribution records, scoring checks, and explicit missing-value handling for those additional analyses.

For a separately collected task-episode ledger, let `A` be all attempted episodes,
`S` verified successes and `E_i` the energy of episode `i` under the declared
boundary. Mean energy per attempt is `sum_all(E_i) / A`. Amortized energy per
success is `sum_all(E_i) / S` for `S > 0`, and is undefined when `S = 0`.
Both numerators include failed episodes and internal retries. If reporting
`sum_successful(E_i) / S`, label it a successful-episode conditional mean;
it excludes failure energy. Compute these episode estimands from the linked task-level ledger, preserving the attempt and success denominators.

## 5. Descriptive helper and optional repetition intervals

For a runnable format check, use the [synthetic input](examples/request_records_synthetic.csv) and compare with the [expected output](examples/request_summary_synthetic.json):

```bash
python3 survey/compute_ci.py \
  --input survey/examples/request_records_synthetic.csv \
  --out request_summary_local.json
```

The nine invented requests have five status-ok completions, four correct answers and four failures. Thus the unconditional request-attempt success rate is `4/9`, while the conditional status-ok correctness mean is `4/5`. Power and energy are absent. These records exercise the schema and failure denominators; empirical quality, throughput, energy, uncertainty, and experimental-independence conclusions require measured task runs and a declared design. Legacy JSON keys containing `task` count request attempts, not the task episodes defined above. See [example notes](examples/README.md).

After collecting new observations in your own `per_trial.csv`:

```bash
python3 survey/compute_ci.py --input per_trial.csv --out ci_report.json
```

The helper groups by campaign, model, and context, reports counts and
unconditional success, and describes supplied numeric values over status-ok
requests. Missing/nonfinite/negative metric values are counted as missing.
The helper summarizes supplied fields and reports missing values explicitly. Analyze cross-context averages, matched model/settings comparisons, and Pareto frontiers with the corresponding study design and matched observations.

Intervals are **off by default**. Only when `repetition_id` identifies independently
repeated experiments under a defensible design may the caller opt in:

```bash
python3 survey/compute_ci.py --input per_trial.csv --out ci_report.json \
  --bootstrap-unit repetition_id --bootstrap 1000
```

This computes percentile bootstrap intervals for the **equally weighted mean of
repetition-specific means/rates**, resampling independent repetition summaries
with replacement. This estimand can differ from the request-weighted mean when
repetitions contain different numbers of requests. Preserve dependence within
task rows and shared batches by resampling at the independent repetition unit;
report the task/population sampling design alongside the interval. At least two
observed repetitions and 1000 resamples are required; additional independent
repetitions strengthen uncertainty characterization. For conditional metrics,
repetitions with no observed status-ok metric have no estimable conditional mean
and are reported as unavailable rather than imputed. Report the design and
independent repetition count with the interval.

The archived retained-link census is documented separately in `survey/protocol.md`.
