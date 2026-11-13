#!/usr/bin/env python3
"""Check public file closure; optionally verify anonymous access to the pinned edition."""
from pathlib import Path
from urllib.parse import unquote, urlsplit, quote
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MD_LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', action='store_true', help='HEAD every listed file without credentials')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'survey/public_artifacts.json').read_text())
    paths = manifest['files']
    failures = []
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    for rel in paths:
        p = (ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            failures.append('Missing or non-repository file: '+rel)
        if rel not in tracked:
            failures.append('Not in Git index: '+rel)
    if len(paths) != len(set(paths)):
        failures.append('Duplicate public-file entries')
    for doc in [ROOT/'README.md', *ROOT.glob('survey/**/*.md'), ROOT/'CONTRIBUTING.md']:
        text = doc.read_text()
        for target in MD_LINK.findall(text):
            u = urlsplit(target)
            if not u.scheme and not u.netloc and u.path:
                p = (doc.parent/unquote(u.path)).resolve()
                if not p.is_relative_to(ROOT) or not p.exists():
                    failures.append(f'{doc.relative_to(ROOT)}: missing local link {target}')
                elif p.is_file() and str(p.relative_to(ROOT)) not in paths:
                    failures.append(f'{doc.relative_to(ROOT)}: linked file omitted from inventory {target}')
        if re.search(r'/Users/|file://|\.codex/|localhost', text):
            failures.append(f'{doc.relative_to(ROOT)}: non-public local pointer')
    tex = (ROOT/'survey/supplement/[submission identifier removed]-supp.tex').read_text()
    for rel in re.findall(r'\\artifact\{([^}]+)\}', tex):
        if rel not in paths:
            failures.append('Supplement references unlisted file: '+rel)
    for figure in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
        if 'survey/figures/'+figure not in paths:
            failures.append('Supplement has unpublished figure dependency: '+figure)
    if re.search(r'/Users/|file://|\.codex/|localhost', tex):
        failures.append('Supplement contains a local-only pointer')
    remote = []
    if args.online and not failures:
        def check(rel):
            url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['edition']}/{quote(rel)}"
            try:
                with urlopen(Request(url, method='HEAD', headers={'User-Agent':'public-artifact-check'}), timeout=30) as r:
                    return {'file':rel, 'status':r.status, 'url':url}
            except Exception as exc:
                return {'file':rel, 'error':str(exc), 'url':url}
        with ThreadPoolExecutor(max_workers=6) as pool:
            remote = list(pool.map(check, paths))
        failures.extend('Anonymous access failed: '+x['file']+' '+x.get('error','')
                        for x in remote if x.get('status') != 200)
    print(json.dumps({'edition':manifest['edition'], 'listed_files':len(paths),
                      'anonymous_files_checked':len(remote), 'failures':failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
