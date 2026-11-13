"""Canonical inputs shared by the literature and repository link audits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def resource_documents():
    return [ROOT / 'README.md']

def resource_lines():
    for path in resource_documents():
        section = '(preamble)'
        fenced = False
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.lstrip().startswith('```'):
                fenced = not fenced
                continue
            if fenced:
                continue
            if line.startswith('#'):
                section = line.lstrip('#').strip()
            yield str(path.relative_to(ROOT)) + ': ' + section, line
