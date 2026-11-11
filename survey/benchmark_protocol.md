# Benchmark Reporting Protocol — [submission identifier removed]

This file is the reproducibility contract for the artifact-backed server
reference benchmark described in Supplementary Section S1 of manuscript
`[submission identifier removed]`. It defines what must be logged **per trial**, how failed
batches are treated, and how variability is computed. It does **not** contain
measurements, and it does not reconstruct values from the historical run whose
per-trial traces were not retained.

---

## 1. Scope

The benchmark is a **controlled server-side reference measurement** on a
`4x RTX 4090 D` server. It is evidence grade **E2**. It is not an edge-device,
mobile-SoC, MCU, or NPU measurement, and its absolute latency / throughput /
energy numbers must not be extrapolated to such hardware. Its only defensible
claim is a *within-stack, multi-metric* observation: no single model jointly
dominates quality, throughput, tail latency, and energy.

## 2. Mandatory per-trial record

Every trial writes exactly one row to `per_trial.csv` with these columns:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | Identifier of the measurement campaign |
| `model` | str | Exact model tag (e.g. `qwen3:8b`) |
| `checkpoint` | str | Checkpoint / manifest hash when available |
| `precision` | str | Weight/KV precision as served |
| `runtime_version` | str | Serving framework + version |
| `driver_version` | str | GPU driver / CUDA version |
| `context` | int | Context budget (512 / 1024 / 2048 / 4096) |
| `task` | str | Micro-suite task id |
| `seed` | int | Trial seed |
| `batch_id` | str | Batch this trial belongs to |
| `ttft_ms` | float | Time to first token (ms) |
| `p95_ms` | float | Request latency p95 for the batch (ms) |
| `generated_tokens` | int | Generated tokens in this trial |
| `tok_per_s` | float | Generated tokens / generation time |
| `gpu_board_power_w` | float | Mean GPU-board power over the generation window (W) |
| `j_per_tok` | float | `gpu_board_power_w / tok_per_s` (see Section 4) |
| `correct` | int | 1 if exact match, else 0 |
| `status` | str | `ok` / `timeout` / `empty` / `unparseable` / `incomplete` |
| `failure_reason` | str | Empty when `status == ok` |

Fields that cannot be determined must be recorded as the literal string
`UNRECORDED` — never guessed, never back-filled from memory.

## 3. Failed and incomplete batches

A batch is admitted to a summary **only if every trial in that
(model, context) cell has `status == ok`**. Incomplete batches, timeouts, empty
generations, and unparseable outputs are excluded as whole batches; there is no
imputation and no partial credit. Excluded batches must still be logged with
their `failure_reason`, and every released summary must report the excluded
batch count alongside the admitted count. This prevents silent inflation of
quality for models that fail more often, but it also means reported values are
conditional on successful completion — a fact that must be stated with every
table.

## 4. Energy boundary

Energy scope is **GPU-board power only**. Wall-plug/PSU losses, host CPU and
DRAM, storage, networking, cooling, and idle/standby overhead are excluded.

```
j_per_tok = mean_gpu_board_power_w * generation_time_s / generated_tokens
          ≈ mean_gpu_board_power_w / tok_per_s
```

`generation_time_s` excludes queueing and prompt-prefill-only idle time. A
wall-plug comparison would require an outlet power meter and is explicitly out
of scope for this benchmark.

## 5. Variability and confidence intervals

For each metric and each `(model, context)` group, reports must contain:

1. the arithmetic mean, sample standard deviation, min and max across trials;
2. a 95% bootstrap confidence interval from at least 1000 resamples of trials
   (percentile method, fixed seed for reproducibility);
3. min--max dispersion across the four context budgets;
4. the number of admitted and excluded batches.

`survey/compute_ci.py` implements items 1--4 from `per_trial.csv`. **If the
input file is absent, the script exits with an explanatory error; it never
synthesizes values for a historical run.** This is why Supplementary Section
S1 reports the historical five-model summary as E2 point estimates rather than
claiming confidence intervals that cannot be verified.

## 6. Reproducing

```bash
# after a measurement run has produced per_trial.csv
python3 survey/compute_ci.py --input per_trial.csv --out ci_report.json
python3 survey/verify_corpus.py
python3 survey/verify_repos.py
```

The companion literature census is described in `survey/protocol.md`.
