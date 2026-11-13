# Benchmark reporting protocol - [submission identifier removed]

Reporting contract established 11 November 2026; public example and file navigation updated 13 November 2026. This document specifies **future runs**;
it contains no measurements and cannot reconstruct the historical server run.
Use the Git commit containing the protocol and helper as the analysis revision.
The separate census snapshot `82b6110` anchors archived link manifests only;
its older benchmark protocol does not define the current reporting policy.

## 1. Historical boundary

Supplement S1 retains incomplete summaries from a `4x RTX 4090 D` server.
They are **E3, exploratory and incomplete**, not reproducible validation.
Prompts, exact model/runtime/precision settings, outputs, repetitions, failure
denominators, and power attribution were not retained and cannot be restored.
The summaries establish neither model-capability/efficiency rankings nor a
validated within-stack Pareto frontier, and do not derive survey recommendations.
Their arithmetic mean of context-specific p95 summaries is not a pooled request
p95. Their reported power/throughput ratio is an unverified GPU-board
approximation, not system or edge-device energy. No future-run schema supplies
missing historical observations or confidence intervals.

## 2. Request log and provenance

Write one row per **attempted task request**, including failed and interrupted
requests, to your newly collected `per_trial.csv`. A [header-only template](examples/request_log_template.csv) is supplied; it contains no observations. This user-created log is not a distributed measurement file. Retries receive distinct request IDs. Counts are
attempt counts, not unique user goals or completed multi-request workflows.
Log every attempt before executing it and finalize its status even when a batch
is interrupted. Include the expected task/batch roster in the campaign manifest
so a separate audit can detect omitted attempts. The helper cannot discover
unlogged work or certify batch-roster completeness.

| Fields | Definition |
|---|---|
| `run_id`, `request_id` | Campaign ID and request ID unique within that campaign. Required for counting. |
| `model`, `context`, `task` | Exact model tag, nominal context budget, and suite task ID. `model` and `context` are required grouping keys. |
| `episode_id`, `parent_request_id` | Optional linking metadata for a separate workflow ledger. The helper does not aggregate episodes or infer goal completion. |
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

Record other unavailable fields as `UNRECORDED`, never guessed or back-filled.
A run with missing required scientific provenance may remain exploratory even
when the helper can count its rows. Missing structural counting keys, unknown
statuses, duplicate request IDs, or missing binary correctness for completed
requests are rejected. Archive the CSV, roster, prompts, scorer, traces, code,
configuration, and their hashes.

A task episode may contain several model/tool requests, retries, and fallback actions.
Keep its success criterion, start/termination timestamps, deadline, and energy
boundary in a separate episode ledger linked to the request traces. A new
user-level restart is a new episode. Do not mix episode rows with constituent
request rows in one helper analysis. Episode success and deadline satisfaction
require a separate task-level analysis; a failed episode's finite termination
time is not successful completion time. Record its incurred cost regardless.

## 3. Failures, denominators, and latency

Report attempted/completed/failed task attempts and batches, with status and
failure counts. A completed batch has `status=ok` for every logged request in
that batch; an external roster audit is needed to establish that no request is
missing. Do not omit a failed batch from unconditional denominators. The helper
reports unconditional task-attempt success as
`number(status=ok and correct=1) / number(all logged attempts)`.

The helper separately labels numeric means over **status-ok requests** as
conditional. This is not the historical successful-whole-batch filter.
Conditioning can select easier or faster outcomes; it does not prevent quality
inflation or quantify the unknown historical selection bias. If a study also
reports a complete-batch-only subset, label its denominators and selection
explicitly and implement that additional analysis separately.

Compute empirical request p95 from individual `request_latency_ms` values with
the nearest-rank rule `sorted(x)[ceil(0.95*n)-1]`. Never average supplied batch
percentiles to obtain pooled p95. The helper provides both a status-ok p95 and a
pooled percentile of **recorded elapsed times across all statuses**. It includes
observed counts and missing-time counts. The latter is not an unconditional
completed-response latency percentile because timeout/incomplete values are
censored. Report termination thresholds and failure rates alongside it; use a
separately justified censored-latency analysis if an unconditional completion-time
claim is needed. No percentile confidence interval is supplied by this helper.

## 4. Power integration and energy scope

State whether the boundary is one GPU board, the sum of specified GPU boards,
a SoC, or the whole system. For sampled power, integrate the timestamped trace
across that same boundary and window, then divide by the tokens produced inside
it. Specify allocation of shared-device energy, overlap, prefill, queueing,
radio/host activity, and idle subtraction. Summed instantaneous board power must
not count unmeasured boards or double-count overlapping activity.

`mean_power_w / tok_per_s` approximates J/token only when both terms refer to
the same devices, interval, and aggregation. A product of separately
context-averaged power/rate columns cannot validate energy accounting. GPU-only
figures omit host CPU/DRAM, PSU losses, storage, networking, and cooling; they do
not establish system or device efficiency. The helper **summarizes supplied
numeric fields**; it does not integrate power, verify attribution/window alignment,
validate scoring, or infer energy when these records are missing.

For a separately collected task-episode ledger, let `A` be all attempted episodes,
`S` verified successes and `E_i` the energy of episode `i` under the declared
boundary. Mean energy per attempt is `sum_all(E_i) / A`. Amortized energy per
success is `sum_all(E_i) / S` for `S > 0`, and is undefined when `S = 0`.
Both numerators include failed episodes and internal retries. If reporting
`sum_successful(E_i) / S`, label it a successful-episode conditional mean;
it excludes failure energy. These episode estimands require a separate analysis:
the request-level helper does not construct them from constituent requests.

## 5. Descriptive helper and optional repetition intervals

For a runnable format check, use the [synthetic input](examples/request_records_synthetic.csv) and compare with the [expected output](examples/request_summary_synthetic.json):

```bash
python3 survey/compute_ci.py \
  --input survey/examples/request_records_synthetic.csv \
  --out request_summary_local.json
```

The nine invented requests have five status-ok completions, four correct answers and four failures. Thus the unconditional request-attempt success rate is `4/9`, while the conditional status-ok correctness mean is `4/5`. Power and energy are absent. These records exercise the schema and failure denominators only; they do not establish model quality, throughput, energy, uncertainty or experimental independence. Legacy JSON keys containing `task` count request attempts, not the task episodes defined above. See [example notes](examples/README.md).

After collecting new observations in your own `per_trial.csv`:

```bash
python3 survey/compute_ci.py --input per_trial.csv --out ci_report.json
```

The helper groups by campaign, model, and context, reports counts and
unconditional success, and describes supplied numeric values over status-ok
requests. Missing/nonfinite/negative metric values are counted as missing.
It never synthesizes a missing input file or absent measurement. It does not
compute cross-context averages, enforce matched models/settings, or establish a
Pareto frontier.

Intervals are **off by default**. Only when `repetition_id` identifies independently
repeated experiments under a defensible design may the caller opt in:

```bash
python3 survey/compute_ci.py --input per_trial.csv --out ci_report.json \
  --bootstrap-unit repetition_id --bootstrap 1000
```

This computes percentile bootstrap intervals for the **equally weighted mean of
repetition-specific means/rates**, resampling independent repetition summaries
with replacement. This estimand can differ from the request-weighted mean when
repetitions contain different numbers of requests. It does not bootstrap
individual dependent task rows or a shared batch p95, and does not validate
independence or account for task/population sampling. At least two observed
repetitions and 1000 resamples are required for an interval; two repetitions
still offer weak uncertainty characterization. For conditional metrics,
repetitions with no observed status-ok metric have no estimable conditional mean
and are reported as unavailable, rather than imputed. Report the design and
independent repetition count, not just a confidence interval.

The archived retained-link census is documented separately in `survey/protocol.md`.
