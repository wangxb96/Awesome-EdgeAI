# Request-log format examples

These files demonstrate the [reporting contract](../benchmark_protocol.md) and [descriptive helper](../compute_ci.py). They are not device measurements, a benchmark dataset, model predictions or reconstructed historical records.

| File | Meaning |
|---|---|
| [request_records_synthetic.csv](request_records_synthetic.csv) | Nine deliberately invented request attempts from `SYNTHETIC_NOT_A_MODEL`; every row is labelled `synthetic_format_only` |
| [request_summary_synthetic.json](request_summary_synthetic.json) | Exact helper output for that fixture, with bootstrap disabled and energy absent |
| [request_log_template.csv](request_log_template.csv) | Header-only template for a separately collected real run; no observations are supplied |

From the repository root:

```sh
python3 survey/compute_ci.py \
  --input survey/examples/request_records_synthetic.csv \
  --out request_summary_local.json
```

The generated local file must be JSON-equivalent to the supplied summary. There are nine attempts, five parseable completions, four correct answers and four failures (`timeout`, `empty`, `unparseable`, `incomplete`). Request-attempt success is `4/9`; status-ok conditional correctness is `4/5`. Of four batches, one is complete and three contain a failed request. The nearest-rank p95 is 180 ms for completed requests and 1000 ms for all recorded elapsed times; the latter includes censoring bounds and is not a completed-response latency guarantee. All timings are invented. No power or energy values, model checkpoints, actual prompts, outputs, independently repeated experiments or task-episode outcomes are supplied by the fixture.

The helper's legacy JSON fields `attempted_tasks`, `completed_tasks`, `failed_tasks` and `unconditional_task_success` refer to constituent request attempts. They do not count user goals or infer agent-workflow completion. A real run additionally requires the provenance, scorer, task roster, traces and energy boundaries specified by the reporting contract. Copy the header-only template into your own run directory and populate it from actual logs; absent historical observations remain absent.

[Back to artifacts](../../README.md#research-artifacts-and-figures)
