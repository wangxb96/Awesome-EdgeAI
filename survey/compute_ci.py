#!/usr/bin/env python3
"""Dispersion and bootstrap confidence intervals for per-trial benchmark logs.

Companion to ``survey/benchmark_protocol.md`` ([submission identifier removed], Supplementary
Section S1). It reads a per-trial CSV and reports, for each (model, context)
group and each numeric metric: n, mean, sample standard deviation, min, max,
and a 95% percentile bootstrap confidence interval (>=1000 resamples).

IMPORTANT: this script never invents data. If the per-trial log is missing, it
exits with an explanatory error. The historical 2026 server run did not retain
per-trial traces; its numbers are therefore reported as E2 point estimates
only, and no confidence interval is claimed for them.

Usage::

    python3 survey/compute_ci.py --input per_trial.csv --out ci_report.json
    python3 survey/compute_ci.py --input per_trial.csv --metrics quality tok_per_s p95_ms j_per_tok
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("quality", "tok_per_s", "ttft_ms", "p95_ms", "j_per_tok")
BOOTSTRAP = 1000
SEED = 20261110


def read_rows(path: Path) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"No trial rows in {path}; refusing to synthesize data.")
    admitted = [r for r in rows if r.get("status", "").strip().lower() == "ok"]
    excluded = [r for r in rows if r not in admitted]
    audit = {
        "total_rows": len(rows),
        "admitted_trials": len(admitted),
        "excluded_trials": len(excluded),
        "exclusion_reasons": dict(sorted(
            ((reason, sum(1 for r in excluded
                          if r.get("failure_reason", "UNRECORDED") == reason))
             for reason in {r.get("failure_reason", "UNRECORDED") for r in excluded}),
            key=lambda kv: -kv[1],
        )),
    }
    return admitted, audit


def metric_value(row: dict, metric: str) -> float | None:
    if metric == "quality":
        raw = row.get("correct")
        try:
            return 1.0 if int(raw) == 1 else 0.0
        except (TypeError, ValueError):
            return None
    raw = row.get(metric)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def bootstrap_ci(values: list[float], rng: random.Random,
                 n_boot: int = BOOTSTRAP) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="per-trial CSV (see survey/benchmark_protocol.md)")
    ap.add_argument("--out", type=Path, default=Path("ci_report.json"))
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(
            f"Per-trial log not found: {args.input}\n"
            "This script does not fabricate measurements. Run a measurement "
            "campaign that writes per_trial.csv as defined in "
            "survey/benchmark_protocol.md, then retry."
        )

    admitted, audit = read_rows(args.input)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in admitted:
        groups[(row.get("model", "UNRECORDED"),
                row.get("context", "UNRECORDED"))].append(row)

    rng = random.Random(SEED)
    report: dict = {
        "generated_by": "survey/compute_ci.py",
        "source": str(args.input),
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": SEED,
        "policy": "complete successful trials only; no imputation",
        "audit": audit,
        "groups": {},
    }
    for (model, context), rows in sorted(groups.items()):
        entry: dict = {"n_trials": len(rows), "metrics": {}}
        for metric in args.metrics:
            vals = [v for r in rows if (v := metric_value(r, metric)) is not None]
            if not vals:
                continue
            item = {
                "n": len(vals),
                "mean": round(statistics.fmean(vals), 6),
                "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else None,
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
            }
            ci = bootstrap_ci(vals, rng, args.bootstrap)
            item["ci95"] = ([round(ci[0], 6), round(ci[1], 6)] if ci else None)
            entry["metrics"][metric] = item
        report["groups"][f"{model}|{context}"] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"admitted trials={audit['admitted_trials']} "
          f"excluded={audit['excluded_trials']} "
          f"groups={len(report['groups'])}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
