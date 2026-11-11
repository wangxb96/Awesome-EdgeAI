#!/usr/bin/env python3
"""Reproducible literature-corpus verification for the Cognitive Edge Computing survey.

This script is the executable counterpart of ``survey/protocol.md`` and of
Sec. ``Survey Methodology and Evidence Grading'' in [submission identifier removed].

It extracts every arXiv identifier linked from the repository README, resolves
each identifier against the official arXiv Atom API, records the authoritative
title / submission date / withdrawal status, and emits a machine-readable
manifest plus the aggregate counts used in the manuscript.

Usage (from the repository root)::

    python3 survey/verify_corpus.py                 # online verification
    python3 survey/verify_corpus.py --offline      # reuse existing manifest
    python3 survey/verify_corpus.py --out /tmp/x.json

No third-party packages are required. Network access is needed only for the
online mode; the offline mode recomputes all statistics from the manifest.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEFAULT_OUT = ROOT / "survey" / "corpus_manifest.json"
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

LINK_RE = re.compile(
    r"\[([^\]]+)\]\(https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/"
    r"([0-9]{4}\.[0-9]{4,5})[^)]*\)"
)
ANY_RE = re.compile(
    r"https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})"
)
SECTION_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$")


def extract() -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Return sorted arXiv ids, {id: link_label}, {id: [sections]} from README.

    Both Markdown links and bare arXiv URLs are captured, so the manifest is a
    complete census of the corpus rather than a subset of one link style.
    """
    labels: dict[str, str] = {}
    ids: set[str] = set()
    sections: dict[str, list[str]] = defaultdict(list)
    section = "(preamble)"
    for line in README.read_text(encoding="utf-8").splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1).strip()
        line_ids = set(ANY_RE.findall(line))
        for label, aid in LINK_RE.findall(line):
            labels.setdefault(aid, label.strip())
            line_ids.add(aid)
        ids |= line_ids
        for aid in line_ids:
            if section not in sections[aid]:
                sections[aid].append(section)
    return sorted(ids), labels, dict(sections)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def fetch(ids: list[str]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for start in range(0, len(ids), 40):
        chunk = ids[start:start + 40]
        url = API + "?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}
        )
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Awesome-EdgeAI-corpus-verify/1.0"}
                )
                payload = urllib.request.urlopen(req, timeout=60).read()
                for entry in ET.fromstring(payload).findall("a:entry", NS):
                    aid = entry.find("a:id", NS).text.split("/abs/")[-1].split("v")[0]
                    title = " ".join(entry.find("a:title", NS).text.split())
                    published = entry.find("a:published", NS).text[:10]
                    comment_node = entry.find("arxiv:comment", NS)
                    comment = (
                        " ".join(comment_node.text.split())
                        if comment_node is not None and comment_node.text else ""
                    )
                    summary = " ".join(entry.find("a:summary", NS).text.split())
                    blob = (comment + " " + summary).lower()
                    records[aid] = {
                        "title": title,
                        "published": published,
                        "year": int(published[:4]),
                        "comment": comment,
                        "withdrawn": "withdraw" in blob,
                        "source": "arXiv Atom API",
                    }
                break
            except Exception as exc:  # noqa: BLE001 - network retries are intentional
                print(f"  retry {start}: {exc}", file=sys.stderr)
                time.sleep(4)
        time.sleep(1)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="reuse existing manifest")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ids, labels, sections = extract()
    print(f"README arXiv identifiers (unique, all URL styles): {len(ids)}")

    if args.offline and args.out.exists():
        manifest = json.loads(args.out.read_text(encoding="utf-8"))
        records = manifest.get("records", {})
    else:
        records = fetch(ids)

    missing = [i for i in ids if i not in records]
    withdrawn = sorted(i for i, r in records.items() if r.get("withdrawn"))
    mismatches = []
    for aid, label in labels.items():
        rec = records.get(aid)
        if not rec:
            continue
        ratio = difflib.SequenceMatcher(
            None, normalise(label), normalise(rec["title"])
        ).ratio()
        if ratio < 0.62:
            mismatches.append(
                {"id": aid, "similarity": round(ratio, 3),
                 "readme_label": label, "arxiv_title": rec["title"]}
            )

    years = Counter(r["year"] for r in records.values())
    by_section = Counter(s for secs in sections.values() for s in secs)
    by_link_section = Counter(sections[i][0] for i in ids if i in sections)

    out = {
        "generated_by": "survey/verify_corpus.py",
        "source_document": "README.md",
        "verification_source": "https://export.arxiv.org/api/query",
        "counts": {
            "unique_arxiv_ids": len(ids),
            "resolved_by_arxiv_api": len(records),
            "unresolved": len(missing),
            "withdrawn": len(withdrawn),
            "markdown_link_labels_checked": len(labels),
            "title_mismatches_below_0_62": len(mismatches),
        },
        "unresolved_ids": missing,
        "withdrawn_ids": withdrawn,
        "year_distribution": dict(sorted(years.items())),
        "section_distribution_of_links": dict(
            sorted(by_link_section.items(), key=lambda kv: -kv[1])
        ),
        "section_distribution_of_papers": dict(
            sorted(by_section.items(), key=lambda kv: -kv[1])
        ),
        "title_mismatches": sorted(mismatches, key=lambda x: x["similarity"]),
        "records": {
            aid: {**records.get(aid, {}),
                  "readme_label": labels.get(aid, "(bare URL: no link label)"),
                  "sections": sections.get(aid, [])}
            for aid in ids
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    c = out["counts"]
    print(f"resolved={c['resolved_by_arxiv_api']} unresolved={c['unresolved']} "
          f"withdrawn={c['withdrawn']} title-mismatches={c['title_mismatches_below_0_62']}")
    print("year distribution:", dict(sorted(years.items())))
    print(f"manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
