#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict


MODEL_DISPLAY_NAMES = {
    "deepseek-r1:14b": "DeepSeek-R1 14B",
    "gemma3:4b": "Gemma3 4B",
    "gpt-oss:20b": "GPT-OSS 20B",
    "llama2:7b": "Llama2 7B",
    "qwen3:8b": "Qwen3 8B",
}

DEVICE_LABELS = {
    "server": "4x RTX 4090 D server",
    "server": "4x RTX 4090 D server",
}


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def extract_server_suite_timestamp(run_id: str) -> int | None:
    parts = str(run_id).split("_")
    if len(parts) < 3 or parts[0] != "server" or parts[1] != "suite":
        return None
    try:
        return int(parts[2])
    except Exception:
        return None


def normalize_device(device: str) -> str:
    return str(device or "").strip().lower()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def numeric_series(items, key: str):
    vals = []
    for item in items:
        raw = str(item.get(key, "")).strip()
        if not raw or raw.upper() == "NA":
            continue
        try:
            vals.append(float(raw))
        except Exception:
            continue
    return vals


def display_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model.replace(":", " ").replace("-", " "))


def display_device_label(device: str) -> str:
    return DEVICE_LABELS.get(device, device.replace("_", " ").replace("-", " "))


def load_rows(matrix_path: pathlib.Path, device: str):
    rows = []
    target_device = normalize_device(device)
    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if normalize_device(r.get("device", "")) != target_device:
                continue
            rows.append(r)
    by_ts = defaultdict(list)
    for r in rows:
        ts = extract_server_suite_timestamp(r.get("run_id", ""))
        if ts is None:
            continue
        by_ts[ts].append(r)
    if by_ts:
        best_ts = max(
            by_ts,
            key=lambda ts: (
                len(
                    [
                        x
                        for x in by_ts[ts]
                        if parse_int(x.get("oom_or_fail", 0), 0) == 0
                        and parse_float(x.get("p50_latency_ms", 0.0), 0.0) > 0
                        and parse_float(x.get("tokens_per_sec", 0.0), 0.0) > 0
                    ]
                ),
                len(
                    {
                        x.get("model", "")
                        for x in by_ts[ts]
                        if parse_int(x.get("oom_or_fail", 0), 0) == 0
                        and parse_float(x.get("p50_latency_ms", 0.0), 0.0) > 0
                        and parse_float(x.get("tokens_per_sec", 0.0), 0.0) > 0
                    }
                ),
                len(by_ts[ts]),
                ts,
            ),
        )
        rows = by_ts[best_ts]
    return rows


def load_latest_summary(summary_path: pathlib.Path):
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def summary_is_usable(rows):
    if not rows:
        return False
    valid_rows = [
        r
        for r in rows
        if parse_int(r.get("oom_or_fail", 0), 0) == 0
        and parse_float(r.get("p50_latency_ms", 0.0), 0.0) > 0
        and parse_float(r.get("tokens_per_sec", 0.0), 0.0) > 0
    ]
    return len(valid_rows) > 0


def aggregate_summary_rows(rows):
    out = []
    for row in rows:
        out.append({
            "model": row.get("model", ""),
            "context": parse_int(row.get("context_len", "0")),
            "ttft_ms": parse_float(row.get("ttft_ms", 0.0)),
            "ttft_std": 0.0,
            "p50_latency_ms": parse_float(row.get("p50_latency_ms", 0.0)),
            "p50_std": 0.0,
            "p95_latency_ms": parse_float(row.get("p95_latency_ms", 0.0)),
            "p95_std": 0.0,
            "tokens_per_sec": parse_float(row.get("tokens_per_sec", 0.0)),
            "tokens_per_sec_std": 0.0,
            "quality_value": parse_float(row.get("quality_value", 0.0)),
            "quality_std": 0.0,
            "avg_power_w": parse_float(row.get("avg_power_w", 0.0)),
            "avg_power_std": 0.0,
            "energy_per_token_j": parse_float(row.get("energy_per_token_j", 0.0)),
            "energy_per_token_std": 0.0,
            "matrix_rows": 1,
        })
    return sorted(out, key=lambda x: (x["model"], x["context"]))


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("model", ""), parse_int(r.get("context_len", "0")))
        grouped[key].append(r)

    out = []
    for (model, ctx), items in grouped.items():
        valid = [x for x in items if parse_int(x.get("oom_or_fail", "0"), 0) == 0]
        source = valid if valid else items
        ttft_vals = numeric_series(source, "ttft_ms")
        p50_vals = numeric_series(source, "p50_latency_ms")
        p95_vals = numeric_series(source, "p95_latency_ms")
        tps_vals = numeric_series(source, "tokens_per_sec")
        q_vals = numeric_series(source, "quality_value")
        pwr_vals = numeric_series(source, "avg_power_w")
        ept_vals = numeric_series(source, "energy_per_token_j")
        out.append({
            "model": model,
            "context": ctx,
            "ttft_ms": mean(ttft_vals),
            "ttft_std": std(ttft_vals),
            "p50_latency_ms": mean(p50_vals),
            "p50_std": std(p50_vals),
            "p95_latency_ms": mean(p95_vals),
            "p95_std": std(p95_vals),
            "tokens_per_sec": mean(tps_vals),
            "tokens_per_sec_std": std(tps_vals),
            "quality_value": mean(q_vals),
            "quality_std": std(q_vals),
            "avg_power_w": mean(pwr_vals) if pwr_vals else 0.0,
            "avg_power_std": std(pwr_vals) if pwr_vals else 0.0,
            "energy_per_token_j": mean(ept_vals) if ept_vals else 0.0,
            "energy_per_token_std": std(ept_vals) if ept_vals else 0.0,
            "matrix_rows": len(source),
        })
    return sorted(out, key=lambda x: (x["model"], x["context"]))


def save_summary_table(agg_rows, out_csv: pathlib.Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "context",
        "ttft_ms",
        "ttft_std",
        "p50_latency_ms",
        "p50_std",
        "p95_latency_ms",
        "p95_std",
        "tokens_per_sec",
        "tokens_per_sec_std",
        "quality_value",
        "quality_std",
        "avg_power_w",
        "avg_power_std",
        "energy_per_token_j",
        "energy_per_token_std",
        "matrix_rows",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)


def plot_figures(agg_rows, figures_dir: pathlib.Path, device: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib import patheffects
        from matplotlib.lines import Line2D
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip3 install matplotlib\n"
            f"Original error: {e}"
        )

    figures_dir.mkdir(parents=True, exist_ok=True)

    contexts = sorted({r["context"] for r in agg_rows})
    models = sorted({r["model"] for r in agg_rows})
    device_label = display_device_label(device)
    model_colors = {
        model: color
        for model, color in zip(
            models,
            ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#7c3aed", "#14b8a6"],
        )
    }
    model_markers = {
        "deepseek-r1:14b": "o",
        "gemma3:4b": "s",
        "gpt-oss:20b": "D",
        "llama2:7b": "^",
        "qwen3:8b": "P",
    }
    context_colors = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#7c3aed", "#14b8a6"]
    context_markers = {512: "o", 1024: "s", 2048: "^", 4096: "D"}
    model_labels = [display_model_name(model) for model in models]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "axes.edgecolor": "#334155",
            "font.size": 13,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    def add_device_badge(ax):
        ax.text(
            0.02,
            0.98,
            f"Device: {device_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
        )

    # 1) Throughput by model/context
    fig1, ax1 = plt.subplots(figsize=(12.5, 7.2))
    width = 0.78 / max(1, len(contexts))
    x = list(range(len(models)))
    for i, ctx in enumerate(contexts):
        vals = []
        errs = []
        for m in models:
            row = next((r for r in agg_rows if r["model"] == m and r["context"] == ctx), None)
            vals.append(row["tokens_per_sec"] if row else 0.0)
            errs.append(row["tokens_per_sec_std"] if row else 0.0)
        x_shift = [xx - 0.39 + (i + 0.5) * width for xx in x]
        ax1.bar(
            x_shift,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            label=f"{ctx}",
            color=context_colors[i % len(context_colors)],
            edgecolor="white",
            linewidth=1.0,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=14, ha="right")
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_xlabel("Model")
    ax1.set_title("Throughput by model")
    add_device_badge(ax1)
    ax1.legend(title="Context tokens", ncol=2, loc="upper right", frameon=True)
    ax1.grid(axis="x", visible=False)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(8))
    fig1.tight_layout()
    fig1.savefig(str(figures_dir / f"{device}_throughput_by_model.png"), dpi=220, bbox_inches="tight")
    fig1.savefig(str(figures_dir / f"{device}_throughput_by_model.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # 2) Latency vs quality
    fig2, ax2 = plt.subplots(figsize=(11.8, 7.2))
    label_offsets = {
        "llama2:7b": (26, 18),
        "gemma3:4b": (18, 24),
        "gpt-oss:20b": (54, 30),
        "qwen3:8b": (-92, 18),
        "deepseek-r1:14b": (30, 20),
    }
    for model in models:
        model_rows = sorted([r for r in agg_rows if r["model"] == model], key=lambda item: item["context"])
        xs = [r["p50_latency_ms"] for r in model_rows]
        ys = [r["quality_value"] for r in model_rows]
        ax2.plot(xs, ys, color=model_colors[model], linewidth=2.4, alpha=0.88, zorder=2)
        for row in model_rows:
            ctx = row["context"]
            ax2.errorbar(
                row["p50_latency_ms"],
                row["quality_value"],
                xerr=row["p50_std"],
                yerr=row["quality_std"],
                fmt=context_markers.get(ctx, "o"),
                color=model_colors[model],
                ecolor=model_colors[model],
                markersize=13,
                capsize=4,
                elinewidth=1.2,
                markeredgecolor="white",
                markeredgewidth=1.3,
                alpha=0.96,
                zorder=3,
            )

        anchor = min(model_rows, key=lambda item: item["p50_latency_ms"])
        text = ax2.annotate(
            display_model_name(model),
            (anchor["p50_latency_ms"], anchor["quality_value"]),
            xytext=label_offsets.get(model, (8, 8)),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=model_colors[model],
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": model_colors[model], "alpha": 0.88},
            arrowprops={"arrowstyle": "->", "color": model_colors[model], "lw": 1.5, "alpha": 0.85},
            zorder=4,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

    context_handles = [
        Line2D(
            [0],
            [0],
            marker=context_markers.get(ctx, "o"),
            color="#334155",
            linestyle="None",
            markersize=9,
            label=f"{ctx}",
        )
        for ctx in contexts
    ]
    ax2.legend(handles=context_handles, title="Context tokens", loc="upper right", frameon=True)

    x_all = [r["p50_latency_ms"] for r in agg_rows if r.get("p50_latency_ms", 0) > 0]
    ax2.set_xscale("log")
    ax2.set_xlabel("Median latency (ms, log scale)")
    ax2.set_ylabel("Quality score")
    ax2.set_title("Latency-quality frontier")
    ax2.set_ylim(-0.05, 1.05)
    if x_all:
        xmin = min(x_all)
        xmax = max(x_all)
        ax2.set_xlim(max(1.0, xmin * 0.85), xmax * 1.22)
    ax2.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.grid(True, which="major", alpha=0.22)
    add_device_badge(ax2)
    fig2.tight_layout()
    fig2.savefig(str(figures_dir / f"{device}_latency_vs_quality.png"), dpi=220, bbox_inches="tight")
    fig2.savefig(str(figures_dir / f"{device}_latency_vs_quality.pdf"), bbox_inches="tight")
    plt.close(fig2)

    # 3) TTFT and P95 latency heatmaps
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(11.8, 8.8), sharex=True)
    ttft_matrix = []
    p95_matrix = []
    for model in models:
        ttft_row = []
        p95_row = []
        for ctx in contexts:
            row = next((r for r in agg_rows if r["model"] == model and r["context"] == ctx), None)
            ttft_row.append(row["ttft_ms"] if row else 0.0)
            p95_row.append(row["p95_latency_ms"] if row else 0.0)
        ttft_matrix.append(ttft_row)
        p95_matrix.append(p95_row)

    im1 = ax3a.imshow(ttft_matrix, aspect="auto", cmap="Blues")
    im2 = ax3b.imshow(p95_matrix, aspect="auto", cmap="Oranges")

    for ax, matrix, title in [(ax3a, ttft_matrix, "TTFT (ms)"), (ax3b, p95_matrix, "P95 latency (ms)")]:
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(model_labels)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        for i in range(len(models)):
            for j in range(len(contexts)):
                value = matrix[i][j]
                ax.text(
                    j,
                    i,
                    f"{value:.1f}" if value < 100 else f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white" if value > (max(max(row) for row in matrix) * 0.55) else "#0f172a",
                    fontsize=10,
                    fontweight="bold",
                )

    add_device_badge(ax3a)
    ax3b.set_xticks(range(len(contexts)))
    ax3b.set_xticklabels([str(ctx) for ctx in contexts])
    ax3b.set_xlabel("Context tokens")
    cbar1 = fig3.colorbar(im1, ax=ax3a, fraction=0.025, pad=0.02)
    cbar1.set_label("ms")
    cbar2 = fig3.colorbar(im2, ax=ax3b, fraction=0.025, pad=0.02)
    cbar2.set_label("ms")
    fig3.suptitle("Response startup and tail latency", fontsize=17, fontweight="bold", y=0.98)
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(str(figures_dir / f"{device}_ttft_p95.png"), dpi=220, bbox_inches="tight")
    fig3.savefig(str(figures_dir / f"{device}_ttft_p95.pdf"), bbox_inches="tight")
    plt.close(fig3)

    # 4) Quality heatmap (model x context)
    fig4, ax4 = plt.subplots(figsize=(10.8, 6.2))
    z = []
    for m in models:
        row = []
        for ctx in contexts:
            v = next((r["quality_value"] for r in agg_rows if r["model"] == m and r["context"] == ctx), 0.0)
            row.append(v)
        z.append(row)
    im = ax4.imshow(z, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax4.set_yticks(range(len(models)))
    ax4.set_yticklabels(model_labels)
    ax4.set_xticks(range(len(contexts)))
    ax4.set_xticklabels([str(c) for c in contexts])
    ax4.set_xlabel("Context tokens")
    ax4.set_ylabel("Model")
    ax4.set_title("Exact-match quality heatmap")
    add_device_badge(ax4)
    for i in range(len(models)):
        for j in range(len(contexts)):
            text_color = "white" if z[i][j] >= 0.6 else "#0f172a"
            ax4.text(j, i, f"{z[i][j]:.2f}", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")
    cbar = fig4.colorbar(im, ax=ax4)
    cbar.set_label("Quality score")
    fig4.tight_layout()
    fig4.savefig(str(figures_dir / f"{device}_quality_heatmap.png"), dpi=220, bbox_inches="tight")
    fig4.savefig(str(figures_dir / f"{device}_quality_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig4)

    # 5) Energy efficiency map
    fig5, ax5 = plt.subplots(figsize=(12.4, 7.4))
    energy_label_offsets = {
        "deepseek-r1:14b": (34, 18),
        "gemma3:4b": (24, 22),
        "gpt-oss:20b": (24, 18),
        "llama2:7b": (24, -26),
        "qwen3:8b": (24, 18),
    }
    for model in models:
        model_rows = sorted([r for r in agg_rows if r["model"] == model], key=lambda item: item["context"])
        xs = [r["avg_power_w"] for r in model_rows]
        ys = [r["energy_per_token_j"] for r in model_rows]
        ax5.plot(xs, ys, color=model_colors[model], linewidth=2.2, alpha=0.5, zorder=2)
        for row in model_rows:
            ctx = row["context"]
            ax5.errorbar(
                row["avg_power_w"],
                row["energy_per_token_j"],
                xerr=row["avg_power_std"],
                yerr=row["energy_per_token_std"],
                fmt=context_markers.get(ctx, "o"),
                color=model_colors[model],
                ecolor=model_colors[model],
                markersize=13,
                capsize=4,
                markeredgecolor="white",
                markeredgewidth=1.2,
                alpha=0.95,
                zorder=3,
            )
        anchor = min(model_rows, key=lambda item: item["energy_per_token_j"])
        text = ax5.annotate(
            display_model_name(model),
            (anchor["avg_power_w"], anchor["energy_per_token_j"]),
            xytext=energy_label_offsets.get(model, (8, 8)),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=model_colors[model],
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": model_colors[model], "alpha": 0.88},
            arrowprops={"arrowstyle": "->", "color": model_colors[model], "lw": 1.5, "alpha": 0.85},
            zorder=4,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

    ax5.annotate(
        "more efficient",
        xy=(0.12, 0.12),
        xytext=(0.28, 0.26),
        textcoords="axes fraction",
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#475569"},
        fontsize=11,
        color="#475569",
    )
    ax5.legend(handles=context_handles, title="Context tokens", loc="upper right", frameon=True)
    ax5.set_xlabel("Average GPU power (W)")
    ax5.set_ylabel("Energy/token (J)")
    ax5.set_title("Power-energy efficiency map")
    add_device_badge(ax5)
    fig5.tight_layout()
    fig5.savefig(str(figures_dir / f"{device}_energy_profile.png"), dpi=220, bbox_inches="tight")
    fig5.savefig(str(figures_dir / f"{device}_energy_profile.pdf"), bbox_inches="tight")
    plt.close(fig5)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures and aggregated table from experiment_matrix.csv")
    parser.add_argument("--matrix", default="experiments/experiment_matrix.csv")
    parser.add_argument("--summary-json", default="data/server_suite_summary_latest.json")
    parser.add_argument("--device", default="server")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--out-csv", default="analysis/server_suite_aggregated.csv")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    matrix_path = root / args.matrix
    summary_path = root / args.summary_json
    figures_dir = root / args.figures_dir
    out_csv = root / args.out_csv

    summary_rows = load_latest_summary(summary_path)
    if summary_is_usable(summary_rows):
        agg_rows = aggregate_summary_rows(summary_rows)
    else:
        rows = load_rows(matrix_path, args.device)
        if not rows:
            raise SystemExit(f"No rows found for device='{args.device}' in {matrix_path}")
        agg_rows = aggregate(rows)
    save_summary_table(agg_rows, out_csv)
    plot_figures(agg_rows, figures_dir, args.device)

    print(f"Aggregated rows: {len(agg_rows)}")
    print(f"Summary table: {out_csv}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
