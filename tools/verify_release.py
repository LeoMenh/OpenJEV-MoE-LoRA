#!/usr/bin/env python3
"""Verify published file hashes, pinned source snapshots and local Markdown links."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
    errors = []
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            errors.append('Export hash mismatch: ' + item['path'])
    contract = json.loads((ROOT / 'experiments/batch_nll_20261027/records/experiment_contract.json').read_text())
    source = ROOT / 'experiments/batch_nll_20261027/source'
    for relative, digest in contract['source_hashes'].items():
        path = source / relative
        if not path.is_file() or sha(path) != digest:
            errors.append('Pinned source mismatch: ' + relative)
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)\s]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            if not (path.parent / target.split('#')[0]).exists():
                errors.append(f'Broken link in {path.relative_to(ROOT)}: {target}')
    records = ROOT / 'experiments/batch_nll_20261027/records'
    def inherited_score(value):
        if isinstance(value, dict):
            return any(key in ('baseline_validation', 'best_validation_loss') or inherited_score(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(inherited_score(item) for item in value)
        return False
    for path in records.rglob('*.json'):
        if inherited_score(json.loads(path.read_text())):
            errors.append('Inherited previous-run metrics: ' + str(path.relative_to(ROOT)))
    experiments = [path.name for path in (ROOT / 'experiments').iterdir() if path.is_dir()]
    if experiments != ['batch_nll_20261027']:
        errors.append('Only the current experiment may be published')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"PASS: {len(manifest['files'])} exported files, {len(contract['source_hashes'])} pinned sources, documentation links, and current-experiment-only records")

if __name__ == '__main__':
    main()
