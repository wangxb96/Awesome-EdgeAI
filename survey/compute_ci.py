#!/usr/bin/env python3
"""Describe new, fully logged request records; never reconstruct historical data.

The historical server summaries are exploratory E3. This helper reports logged
attempt/failure denominators, unconditional task success, explicitly conditional
status-ok request summaries, and empirical pooled request percentiles. It does
not validate scoring, power integration, missing attempts, or independence.
Intervals are off by default. With --bootstrap-unit repetition_id, intervals
refer to the mean of independent repetition-specific means, not pooled p95.
See survey/benchmark_protocol.md for the CSV contract and estimands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

METRICS = ("correct", "request_latency_ms", "ttft_ms", "generation_time_s",
           "tok_per_s", "j_per_tok")
STATUSES = {"ok", "timeout", "empty", "unparseable", "incomplete"}
IDENTIFIERS = ("run_id", "model", "context", "request_id", "batch_id")
SEED = 20261110


def number(row: dict, field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"No request rows in {path}; refusing to synthesize data.")
    seen = set()
    for index, row in enumerate(rows, start=2):
        for field in IDENTIFIERS:
            value = (row.get(field) or "").strip()
            if not value or value == "UNRECORDED":
                raise SystemExit(f"CSV line {index}: {field} is required for counting.")
            row[field] = value
        row["repetition_id"] = (row.get("repetition_id") or "UNRECORDED").strip()
        row["status"] = (row.get("status") or "").strip().lower()
        if row["status"] not in STATUSES:
            raise SystemExit(f"CSV line {index}: unknown/missing request status.")
        if row["status"] == "ok" and number(row, "correct") not in (0.0, 1.0):
            raise SystemExit(f"CSV line {index}: status-ok requests require correct=0 or 1.")
        # Request IDs must be unique within a campaign, including retries.
        key = (row["run_id"], row["request_id"])
        if key in seen:
            raise SystemExit(f"CSV line {index}: duplicate request_id within run_id.")
        seen.add(key)
    return rows


def counts(rows: list[dict]) -> dict:
    batches = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "UNRECORDED") for k in
                    ("run_id", "model", "context", "repetition_id", "batch_id"))
        batches[key].append(row)
    completed = sum(row["status"] == "ok" for row in rows)
    complete_batches = sum(all(r["status"] == "ok" for r in group)
                           for group in batches.values())
    return {
        "attempted_tasks": len(rows),
        "completed_tasks": completed,
        "failed_tasks": len(rows) - completed,
        "attempted_batches": len(batches),
        "completed_batches": complete_batches,
        "failed_batches": len(batches) - complete_batches,
        "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        "failure_reasons": dict(sorted(Counter(
            r.get("failure_reason") or "UNRECORDED" for r in rows
            if r["status"] != "ok").items())),
    }


def percentile95(values: list[float]) -> float | None:
    """Empirical nearest-rank percentile, never a mean of batch percentiles."""
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def repetition_interval(values: list[float], resamples: int) -> dict:
    """Bootstrap equally weighted independent repetition means, by opt-in only."""
    result = {"n_repetitions": len(values),
              "estimate": statistics.fmean(values) if values else None,
              "ci95": None}
    if len(values) >= 2:
        rng = random.Random(SEED)
        means = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                       for _ in range(resamples))
        result["ci95"] = [means[int(0.025 * resamples)],
                          means[min(resamples - 1, int(0.975 * resamples))]]
    else:
        result["interval_unavailable_reason"] = "Fewer than two independent repetitions."
    return result


def group_report(rows: list[dict], metrics: list[str], unit: str | None,
                 resamples: int) -> dict:
    completed = [r for r in rows if r["status"] == "ok"]
    # Failed requests contribute zero to unconditional task success.
    success = sum(r["status"] == "ok" and number(r, "correct") == 1 for r in rows)
    result = {
        "counts": counts(rows),
        "unconditional_task_success": {"successful_tasks": success,
                                       "attempted_tasks": len(rows),
                                       "rate": success / len(rows)},
        "conditional_status_ok_request_metrics": {},
    }
    repeats = defaultdict(list)
    if unit:
        for row in rows:
            if row[unit] in ("", "UNRECORDED"):
                raise SystemExit("Bootstrap requested but repetition_id is missing/UNRECORDED.")
            repeats[row[unit]].append(row)
        rates = [sum(r["status"] == "ok" and number(r, "correct") == 1
                     for r in group) / len(group) for group in repeats.values()]
        result["unconditional_task_success"]["equal_repetition_mean"] = (
            repetition_interval(rates, resamples))
    for metric in metrics:
        values = [v for r in completed if (v := number(r, metric)) is not None]
        item = {"n_observed": len(values), "n_missing": len(completed) - len(values),
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None}
        if unit:
            rep_means = []
            for group in repeats.values():
                vals = [v for r in group if r["status"] == "ok"
                        and (v := number(r, metric)) is not None]
                if vals:
                    rep_means.append(statistics.fmean(vals))
            item["equal_repetition_mean"] = repetition_interval(rep_means, resamples)
            item["equal_repetition_mean"]["n_repetitions_without_metric"] = len(repeats) - len(rep_means)
        result["conditional_status_ok_request_metrics"][metric] = item
    elapsed = [v for r in rows if (v := number(r, "request_latency_ms")) is not None]
    ok_elapsed = [v for r in completed if (v := number(r, "request_latency_ms")) is not None]
    result["request_latency_percentiles"] = {
        "method": "empirical nearest-rank p95 from request_latency_ms; no p95_ms input used",
        "recorded_elapsed_all_statuses": {"n": len(elapsed),
            "n_missing": len(rows) - len(elapsed), "p95_ms": percentile95(elapsed)},
        "conditional_status_ok": {"n": len(ok_elapsed),
            "n_missing": len(completed) - len(ok_elapsed), "p95_ms": percentile95(ok_elapsed)},
        "censoring_note": "Timeout/incomplete elapsed values are censoring bounds, not completed-response latencies. No unconditional completion-latency p95 is inferred.",
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("ci_report.json"))
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    ap.add_argument("--bootstrap-unit", choices=("repetition_id",),
                    help="Opt in only when IDs identify independent experimental repetitions.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Request log not found: {args.input}\nThis helper does not fabricate measurements. Supply a new fully logged run as specified in survey/benchmark_protocol.md.")
    if args.bootstrap_unit and args.bootstrap < 1000:
        ap.error("--bootstrap must be at least 1000 when intervals are requested")
    rows = read_rows(args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_id"], row["model"], row["context"])].append(row)
    report = {
        "generated_by": "survey/compute_ci.py",
        "source": str(args.input),
        "policy": "All logged attempts define denominators. Numeric means are conditional on status=ok; this is not the historical whole-batch filter.",
        "coverage_limit": "Counts cover logged attempts only. The logger must retain every attempted request and interrupted batch; omitted attempts cannot be detected or reconstructed.",
        "energy_limit": "Provided j_per_tok values are summarized, not validated. Power/device/window integration and scoring are outside this helper.",
        "bootstrap": {"unit": args.bootstrap_unit,
                      "resamples": args.bootstrap if args.bootstrap_unit else 0,
                      "seed": SEED if args.bootstrap_unit else None,
                      "estimand": "mean of independent repetition-specific means, equally weighted; no percentile CI or independence validation"},
        "audit": counts(rows),
        "groups": [],
    }
    for (run_id, model, context), group in sorted(groups.items()):
        report["groups"].append({"run_id": run_id, "model": model, "context": context,
            **group_report(group, args.metrics, args.bootstrap_unit, args.bootstrap)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"logged attempted tasks={len(rows)} groups={len(groups)}; report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
