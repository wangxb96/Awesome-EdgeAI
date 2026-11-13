#!/usr/bin/env python3
"""Check local navigation and canonical paper placement without network access."""
from pathlib import Path
from collections import defaultdict
from urllib.parse import unquote, urlsplit
import re, sys
from doc_sources import ROOT, resource_documents

LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')
ARXIV = re.compile(r'https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})')

def visible(text):
    return re.sub(r'^```[^\n]*\n.*?^```\s*$', '', text, flags=re.M|re.S)

def anchors(path):
    result=set();counts=defaultdict(int)
    for line in visible(path.read_text()).splitlines():
        if not line.startswith('#'):continue
        title=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',line.lstrip('#').strip()).lower()
        slug=re.sub(r'[^\w\- ]','',title).replace(' ','-')
        n=counts[slug];counts[slug]+=1
        result.add(slug+(f'-{n}' if n else ''))
    return result

def main():
    paths=[ROOT/'README.md',ROOT/'CONTRIBUTING.md',
           ROOT/'survey/protocol.md',ROOT/'survey/benchmark_protocol.md',ROOT/'survey/figures/README.md']
    failures=[];count=0
    for path in paths:
        source=visible(path.read_text())
        for target in LINK.findall(source):
            u=urlsplit(target)
            if u.scheme or u.netloc:continue
            local=(path.parent/unquote(u.path)).resolve() if u.path else path
            count+=1
            if not local.exists():failures.append(f'{path.relative_to(ROOT)}: missing {target}')
            elif u.fragment and local.suffix=='.md' and unquote(u.fragment) not in anchors(local):
                failures.append(f'{path.relative_to(ROOT)}: missing heading {target}')
        for stale in ['To be released','publisher link pending verification','Figures/','relevant']:
            if stale in source:failures.append(f'{path.relative_to(ROOT)}: obsolete pointer/placeholder {stale}')
    locations=defaultdict(set)
    for path in resource_documents():
        for aid in ARXIV.findall(visible(path.read_text())):locations[aid].add(str(path.relative_to(ROOT)))
    for aid,places in locations.items():
        if len(places)>1:failures.append(f'{aid}: repeated canonical placement {sorted(places)}')
    print(f'Checked {len(paths)} Markdown documents, {count} local targets, {len(locations)} unique arXiv IDs.')
    for item in failures:print('FAIL:',item)
    if failures:return 1
    print('PASS: local links, anchors, canonical placement and current internal pointers.')
    return 0

if __name__=='__main__':sys.exit(main())
