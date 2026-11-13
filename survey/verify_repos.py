#!/usr/bin/env python3
"""Reproducible artifact-repository verification for the Cognitive Edge survey.

Extracts every ``github.com/<owner>/<repo>`` link from the README, checks that
the repository resolves to a live Git HEAD via ``git ls-remote`` (no GitHub API
token), and emits ``survey/current_repos_manifest.json``.

Usage (from the repository root)::

    python3 survey/verify_repos.py                 # online HEAD check
    python3 survey/verify_repos.py --offline      # recompute stats only
    python3 survey/verify_repos.py --workers 12 --timeout 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from doc_sources import resource_documents, resource_lines
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "survey" / "current_repos_manifest.json"
FROZEN_CACHE = ROOT / "survey" / "repos_manifest.json"
REPO_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def extract() -> dict[str, list[str]]:
    repos: dict[str, list[str]] = defaultdict(list)
    for section, line in resource_lines():
        for owner_repo in REPO_RE.findall(line):
            key = owner_repo.rstrip(".,);")
            if section not in repos[key]:
                repos[key].append(section)
    return dict(repos)


def check(repo: str, timeout: int) -> tuple[str, str]:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        proc = subprocess.run(
            ["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD"],
            capture_output=True, timeout=timeout, env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return repo, "OK"
        return repo, "MISSING"
    except Exception:  # noqa: BLE001 - timeouts/reporting are intentional
        return repo, "ERR"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    repos = extract()
    names = sorted(repos, key=str.lower)
    print(f"Resource-guide GitHub repositories (unique): {len(names)}")

    if args.offline:
        status = {}
        for cache in [FROZEN_CACHE, args.out]:
            if cache.exists():status.update(json.loads(cache.read_text(encoding="utf-8")).get("status", {}))
        status = {name: status[name] for name in names if name in status}
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            status = dict(pool.map(lambda r: check(r, args.timeout), names))

    counts = Counter(status.get(r, "UNKNOWN") for r in names)
    sections = Counter(s for secs in repos.values() for s in secs)
    out = {
        "generated_by": "survey/verify_repos.py",
        "source_documents": [str(p.relative_to(ROOT)) for p in resource_documents()],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_mode": "cached HEAD status; no live revalidation" if args.offline else "live git ls-remote HEAD",
        "frozen_census_revision": "82b611089654cd2fef28a9cfb96b907d3c89dee5",
        "verification_method": "git ls-remote <repo>.git HEAD",
        "counts": {
            "unique_repositories": len(names),
            "ok": counts.get("OK", 0),
            "missing": counts.get("MISSING", 0),
            "error": counts.get("ERR", 0),
            "unknown": counts.get("UNKNOWN", 0),
        },
        "not_ok": [r for r in names if status.get(r) != "OK"],
        "section_distribution": dict(sorted(sections.items(), key=lambda kv: -kv[1])),
        "status": status,
        "repos": {r: {"sections": repos[r], "status": status.get(r, "UNKNOWN")}
                  for r in names},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(out["counts"])
    print("not OK:", out["not_ok"])
    print(f"manifest -> {args.out}")
    return 1 if out["not_ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
