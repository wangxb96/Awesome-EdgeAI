#!/usr/bin/env python3
import argparse
import glob
import json
import pathlib
import re
import subprocess


CANONICAL_MODELS = [
    "deepseek-r1:14b",
    "gemma3:4b",
    "gpt-oss:20b",
    "llama2:7b",
    "qwen3:8b",
]


def load_summary_rows(data_dir: pathlib.Path):
    rows = []
    for path_str in sorted(glob.glob(str(data_dir / "server_suite_summary_*.json"))):
        path = pathlib.Path(path_str)
        if path.name == "server_suite_summary_latest.json":
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(content, list):
            continue
        for row in content:
            if not isinstance(row, dict):
                continue
            model = row.get("model")
            ctx = row.get("context_len")
            if not model or ctx is None:
                continue
            run_id = str(row.get("run_id", ""))
            m = re.match(r"^server_suite_(\d+)_", run_id)
            ts = int(m.group(1)) if m else -1
            notes = str(row.get("notes", ""))
            fail = int(row.get("oom_or_fail", 0) or 0)
            if fail != 0:
                continue
            if "failed=" in notes or "HTTP 404" in notes or "Connection refused" in notes:
                continue
            picked = dict(row)
            picked["_ts"] = ts
            rows.append(picked)
    return rows


def pick_latest_valid(rows):
    latest = {}
    for row in rows:
        key = (row["model"], int(row["context_len"]))
        old = latest.get(key)
        if old is None or row["_ts"] > old["_ts"]:
            latest[key] = row
    out = sorted(latest.values(), key=lambda r: (r["model"], int(r["context_len"])))
    for row in out:
        row.pop("_ts", None)
    return out


def pick_latest_complete_per_model(rows, required_contexts):
    # model -> ts -> context -> row
    grouped = {}
    for row in rows:
        model = row["model"]
        if model not in CANONICAL_MODELS:
            continue
        ts = row["_ts"]
        ctx = int(row["context_len"])
        grouped.setdefault(model, {}).setdefault(ts, {})[ctx] = row

    selected = []
    required_set = set(required_contexts)
    for model in CANONICAL_MODELS:
        model_runs = grouped.get(model, {})
        chosen_ts = None
        for ts in sorted(model_runs.keys(), reverse=True):
            if required_set.issubset(model_runs[ts].keys()):
                chosen_ts = ts
                break
        # Fallback: if no complete run exists, keep latest-valid per context.
        if chosen_ts is None:
            latest_ctx = {}
            for ts, ctx_map in model_runs.items():
                for ctx, row in ctx_map.items():
                    old = latest_ctx.get(ctx)
                    if old is None or row["_ts"] > old["_ts"]:
                        latest_ctx[ctx] = row
            selected.extend(latest_ctx.values())
        else:
            for ctx in required_contexts:
                selected.append(model_runs[chosen_ts][ctx])

    out = sorted(selected, key=lambda r: (r["model"], int(r["context_len"])))
    for row in out:
        row.pop("_ts", None)
    return out


def main():
    parser = argparse.ArgumentParser(description="Rebuild clean server benchmark artifacts from latest valid summary rows.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="server")
    parser.add_argument("--required-contexts", default="512,1024,2048,4096")
    parser.add_argument("--clean-summary", default="analysis/server_suite_summary_clean_latest_valid.json")
    parser.add_argument("--out-csv", default="analysis/server_suite_aggregated.csv")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    data_dir = root / "data"
    rows = load_summary_rows(data_dir)
    required_contexts = [int(x.strip()) for x in args.required_contexts.split(",") if x.strip()]
    clean = pick_latest_complete_per_model(rows, required_contexts)

    clean_summary_path = root / args.clean_summary
    clean_summary_path.parent.mkdir(parents=True, exist_ok=True)
    clean_summary_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    cmd = [
        "python3",
        str(root / "scripts" / "plot_benchmark_results.py"),
        "--summary-json",
        str(clean_summary_path.relative_to(root)),
        "--device",
        args.device,
        "--out-csv",
        args.out_csv,
        "--figures-dir",
        args.figures_dir,
    ]
    subprocess.run(cmd, check=True, cwd=str(root))

    print(f"clean_summary={clean_summary_path}")
    print(f"rows={len(clean)}")


if __name__ == "__main__":
    main()
