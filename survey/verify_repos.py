#!/usr/bin/env python3
"""Reproducible artifact-repository verification for the Cognitive Edge survey.

Extracts every ``github.com/<owner>/<repo>`` link from README.md, checks that
the repository resolves to a live Git HEAD via ``git ls-remote`` (no GitHub API
token, no rate limit), and emits ``survey/repos_manifest.json``.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEFAULT_OUT = ROOT / "survey" / "repos_manifest.json"
REPO_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
SECTION_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$")


def extract() -> dict[str, list[str]]:
    repos: dict[str, list[str]] = defaultdict(list)
    section = "(preamble)"
    for line in README.read_text(encoding="utf-8").splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1).strip()
        for owner_repo in REPO_RE.findall(line):
            owner_repo = owner_repo.rstrip(".,);")
            # README also links to file-level anchors; keep the repo root only.
            key = "/".join(owner_repo.split("/")[:2])
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
    print(f"README GitHub repositories (unique): {len(names)}")

    if args.offline and args.out.exists():
        status = json.loads(args.out.read_text(encoding="utf-8")).get("status", {})
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            status = dict(pool.map(lambda r: check(r, args.timeout), names))

    counts = Counter(status.get(r, "UNKNOWN") for r in names)
    sections = Counter(s for secs in repos.values() for s in secs)
    out = {
        "generated_by": "survey/verify_repos.py",
        "source_document": "README.md",
        "verification_method": "git ls-remote <repo>.git HEAD",
        "counts": {
            "unique_repositories": len(names),
            "ok": counts.get("OK", 0),
            "missing": counts.get("MISSING", 0),
            "error": counts.get("ERR", 0),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
