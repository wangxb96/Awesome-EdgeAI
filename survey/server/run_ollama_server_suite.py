#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MATRIX_CSV = ROOT / "experiments" / "experiment_matrix.csv"

DEFAULT_TASKS = [
    {
        "id": "arith_add",
        "prompt": "Compute 27 + 15. Return only the integer.",
        "answer": "42",
    },
    {
        "id": "arith_div",
        "prompt": "Compute 144 / 12. Return only the integer.",
        "answer": "12",
    },
    {
        "id": "logic_set",
        "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude all roses fade quickly? Answer Yes or No only.",
        "answer": "No",
    },
    {
        "id": "count_chars",
        "prompt": "How many letters are in the word EDGE? Return only the integer.",
        "answer": "4",
    },
    {
        "id": "calendar",
        "prompt": "If tomorrow is Wednesday, what day is today? Return only the day name.",
        "answer": "Tuesday",
    },
    {
        "id": "step_cost",
        "prompt": "A pen costs 3 dollars. You buy 4 pens and use a 2 dollar coupon. Final cost? Return only the integer.",
        "answer": "10",
    },
    {
        "id": "reverse",
        "prompt": "Reverse this string: abcde. Return only the reversed string.",
        "answer": "edcba",
    },
    {
        "id": "compare",
        "prompt": "Which is larger, 0.125 or 0.5? Return only one value exactly as written.",
        "answer": "0.5",
    },
    {
        "id": "short_reason",
        "prompt": "A train travels 60 km in 1.5 hours. What is the average speed in km/h? Return only the integer.",
        "answer": "40",
    },
    {
        "id": "boolean",
        "prompt": "Is 11 a prime number? Answer Yes or No only.",
        "answer": "Yes",
    },
]


def normalized_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (len(s) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[lo])
    w = rank - lo
    return float(s[lo] * (1.0 - w) + s[hi] * w)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_gpu_power_w():
    """Return instantaneous total GPU power draw in watts via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    vals = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except Exception:
            continue
    if not vals:
        return None
    return float(sum(vals))


def make_padding_prompt(target_words: int) -> str:
    seed_words = "edge inference benchmark stability token".split()
    out = []
    while len(out) < target_words:
        out.extend(seed_words)
    return " ".join(out[:target_words])


def run_once(base_url: str, model: str, num_ctx: int, prompt: str, max_tokens: int, temperature: float, seed: int, timeout: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
        },
    }

    p_start = query_gpu_power_w()
    t0 = time.time()
    resp = None
    error = ""
    try:
        resp = post_json(base_url, payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        error = f"HTTP {e.code}"
    except urllib.error.URLError as e:
        error = f"URL error: {e.reason}"
    except Exception as e:
        error = f"Exception: {type(e).__name__}: {e}"

    t1 = time.time()
    p_end = query_gpu_power_w()
    power_samples = [x for x in (p_start, p_end) if isinstance(x, (int, float))]
    avg_power_w = float(statistics.mean(power_samples)) if power_samples else None
    wall_latency_ms = (t1 - t0) * 1000.0

    if error:
        return {
            "ok": False,
            "error": error,
            "wall_latency_ms": wall_latency_ms,
            "avg_power_w": avg_power_w,
            "energy_j": (avg_power_w * wall_latency_ms / 1000.0) if avg_power_w is not None else None,
        }

    # Keep type checkers happy: non-error path must have a parsed response dict.
    if resp is None:
        return {
            "ok": False,
            "error": "Empty response",
            "wall_latency_ms": wall_latency_ms,
            "avg_power_w": avg_power_w,
            "energy_j": (avg_power_w * wall_latency_ms / 1000.0) if avg_power_w is not None else None,
        }

    eval_count = resp.get("eval_count", 0)
    eval_duration = resp.get("eval_duration", 0)
    prompt_eval_count = resp.get("prompt_eval_count", 0)
    prompt_eval_duration = resp.get("prompt_eval_duration", 0)

    decode_tps = 0.0
    if eval_duration > 0 and eval_count > 0:
        decode_tps = eval_count / (eval_duration / 1e9)

    ttft_ms = 0.0
    if prompt_eval_duration > 0:
        ttft_ms = prompt_eval_duration / 1e6

    return {
        "ok": True,
        "error": "",
        "response": resp.get("response", ""),
        "wall_latency_ms": wall_latency_ms,
        "ttft_ms": ttft_ms,
        "decode_tokens": eval_count,
        "decode_tps": decode_tps,
        "prompt_tokens": prompt_eval_count,
        "avg_power_w": avg_power_w,
        "energy_j": (avg_power_w * wall_latency_ms / 1000.0) if avg_power_w is not None else None,
    }


def auto_models_from_ollama():
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []

    models = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts:
            name = parts[0].strip()
            if name and name != "NAME":
                models.append(name)
    return models


def choose_models(models_arg: str):
    if models_arg.strip().lower() == "auto":
        return auto_models_from_ollama()
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def append_matrix_rows(rows, device: str):
    with MATRIX_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([
                r["run_id"],
                r["date"],
                device,
                r["model"],
                r["quantization"],
                r["context_len"],
                r["prompt_tokens"],
                r["gen_tokens"],
                "default",
                "single-session",
                "local-only",
                f"{r['ttft_ms']:.2f}",
                f"{r['p50_latency_ms']:.2f}",
                f"{r['p95_latency_ms']:.2f}",
                f"{r['tokens_per_sec']:.2f}",
                f"{r['avg_power_w']:.2f}" if r.get("avg_power_w") is not None else "NA",
                f"{r['energy_per_token_j']:.6f}" if r.get("energy_per_token_j") is not None else "NA",
                "exact_match",
                f"{r['quality_value']:.3f}",
                str(r["oom_or_fail"]),
                r["notes"],
            ])


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-model Ollama benchmark suite for server experiments.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--models", default="auto", help="Comma-separated models or 'auto' from ollama list")
    parser.add_argument("--contexts", default="512,2048,4096")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--warmup-words", type=int, default=160)
    parser.add_argument("--device", default="server")
    return parser.parse_args()


def main():
    args = parse_args()
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    models = choose_models(args.models)

    if not models:
        raise SystemExit("No models found. Use --models or ensure 'ollama list' has entries.")

    ts = int(time.time())
    raw_csv_path = DATA_DIR / f"server_suite_raw_{ts}.csv"
    summary_json_path = DATA_DIR / f"server_suite_summary_{ts}.json"
    latest_json_path = DATA_DIR / "server_suite_summary_latest.json"

    all_raw = []
    summary_rows = []

    for model in models:
        quant = "runtime-default"
        for num_ctx in contexts:
            latencies = []
            ttfts = []
            tps_list = []
            gen_tokens = []
            prompt_tokens_list = []
            quality_hits = 0
            quality_total = 0
            fail_count = 0
            error_samples = []
            avg_power_samples = []
            total_energy_j = 0.0
            total_decode_tokens = 0

            warm_prompt = make_padding_prompt(args.warmup_words)
            _ = run_once(
                args.base_url,
                model,
                num_ctx,
                warm_prompt,
                args.max_tokens,
                args.temperature,
                args.seed,
                args.timeout,
            )

            for task in DEFAULT_TASKS:
                for rep in range(args.repeats):
                    run_seed = args.seed + rep
                    result = run_once(
                        args.base_url,
                        model,
                        num_ctx,
                        task["prompt"],
                        args.max_tokens,
                        args.temperature,
                        run_seed,
                        args.timeout,
                    )

                    if not result["ok"]:
                        fail_count += 1
                        if len(error_samples) < 3:
                            error_samples.append(result["error"])
                        if result.get("avg_power_w") is not None:
                            avg_power_samples.append(result["avg_power_w"])
                        if result.get("energy_j") is not None:
                            total_energy_j += result["energy_j"]
                        all_raw.append({
                            "model": model,
                            "context_len": num_ctx,
                            "task_id": task["id"],
                            "repeat": rep,
                            "wall_latency_ms": result["wall_latency_ms"],
                            "ttft_ms": 0.0,
                            "decode_tokens": 0,
                            "decode_tps": 0.0,
                            "prompt_tokens": 0,
                            "avg_power_w": result.get("avg_power_w") if result.get("avg_power_w") is not None else "",
                            "energy_j": result.get("energy_j") if result.get("energy_j") is not None else "",
                            "quality_hit": 0,
                            "prediction": "",
                            "answer": task["answer"],
                            "error": result["error"],
                        })
                        continue

                    pred = normalized_text(result["response"])
                    ans = normalized_text(task["answer"])
                    hit = 1 if ans in pred else 0
                    quality_hits += hit
                    quality_total += 1

                    latencies.append(result["wall_latency_ms"])
                    ttfts.append(result["ttft_ms"])
                    tps_list.append(result["decode_tps"])
                    gen_tokens.append(result["decode_tokens"])
                    prompt_tokens_list.append(result["prompt_tokens"])
                    total_decode_tokens += int(result["decode_tokens"])
                    if result.get("avg_power_w") is not None:
                        avg_power_samples.append(result["avg_power_w"])
                    if result.get("energy_j") is not None:
                        total_energy_j += result["energy_j"]

                    all_raw.append({
                        "model": model,
                        "context_len": num_ctx,
                        "task_id": task["id"],
                        "repeat": rep,
                        "wall_latency_ms": result["wall_latency_ms"],
                        "ttft_ms": result["ttft_ms"],
                        "decode_tokens": result["decode_tokens"],
                        "decode_tps": result["decode_tps"],
                        "prompt_tokens": result["prompt_tokens"],
                        "avg_power_w": result.get("avg_power_w") if result.get("avg_power_w") is not None else "",
                        "energy_j": result.get("energy_j") if result.get("energy_j") is not None else "",
                        "quality_hit": hit,
                        "prediction": result.get("response", "").strip().replace("\n", " "),
                        "answer": task["answer"],
                        "error": "",
                    })

            mean_ttft = statistics.mean(ttfts) if ttfts else 0.0
            p50 = statistics.median(latencies) if latencies else 0.0
            p95 = percentile(latencies, 0.95)
            mean_tps = statistics.mean(tps_list) if tps_list else 0.0
            mean_gen = statistics.mean(gen_tokens) if gen_tokens else 0.0
            mean_prompt = statistics.mean(prompt_tokens_list) if prompt_tokens_list else 0.0
            q = (quality_hits / quality_total) if quality_total else 0.0
            mean_power = statistics.mean(avg_power_samples) if avg_power_samples else None
            energy_per_token_j = (total_energy_j / total_decode_tokens) if total_decode_tokens > 0 else None
            total_trials = len(DEFAULT_TASKS) * args.repeats
            oom_or_fail = 1 if fail_count > 0 else 0

            notes = "server_suite via Ollama API"
            if fail_count > 0:
                notes += f"; failed={fail_count}/{total_trials}; sample_errors={'; '.join(error_samples)}"

            summary_rows.append({
                "run_id": f"server_suite_{ts}_{model.replace(':', '_')}_{num_ctx}",
                "date": time.strftime("%Y-%m-%d"),
                "model": model,
                "quantization": quant,
                "context_len": num_ctx,
                "prompt_tokens": int(mean_prompt),
                "gen_tokens": int(mean_gen),
                "ttft_ms": mean_ttft,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "tokens_per_sec": mean_tps,
                "avg_power_w": mean_power,
                "energy_per_token_j": energy_per_token_j,
                "quality_value": q,
                "oom_or_fail": oom_or_fail,
                "notes": notes,
            })

    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "context_len",
            "task_id",
            "repeat",
            "wall_latency_ms",
            "ttft_ms",
            "decode_tokens",
            "decode_tps",
            "prompt_tokens",
            "avg_power_w",
            "energy_j",
            "quality_hit",
            "prediction",
            "answer",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_raw:
            writer.writerow(r)

    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    latest_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    append_matrix_rows(summary_rows, args.device)

    print(f"Models tested: {len(models)}")
    print(f"Contexts tested: {contexts}")
    print(f"Raw results: {raw_csv_path}")
    print(f"Summary: {summary_json_path}")
    print(f"Latest summary: {latest_json_path}")
    print(f"Appended matrix: {MATRIX_CSV}")


if __name__ == "__main__":
    main()
