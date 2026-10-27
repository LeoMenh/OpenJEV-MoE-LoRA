#!/usr/bin/env python3
"""Build a new batch16 NLL oracle from verified immutable per-case route costs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    old, root = args.old_run, args.run_dir
    c = json.loads((old / 'config.json').read_text())
    contract = json.loads((old / 'reuse_contract.json').read_text())
    assert sha(root / 'reused/expert_and_head.pt') == contract['exported_expert_sha256']
    frame = pd.read_csv(c['split_manifest']['paths']['train'])
    n = len(frame)
    costs = torch.empty(n, 18, dtype=torch.float32)
    hashes = {}
    expected_start = 0
    files = sorted((old / 'oracle_train').glob('chunk_*.pt'))
    for index, path in enumerate(files):
        d = torch.load(path, map_location='cpu', weights_only=True)
        start, stop = d['start'], d['stop']
        assert start == expected_start and stop == min(start + 64, n)
        assert d['contract_sha256'] == c['contract_sha256']
        assert d['sample_ids'] == frame.iloc[start:stop].sample_id.astype(str).tolist()
        assert np.array_equal(d['labels'].numpy(), frame.iloc[start:stop].label.to_numpy())
        assert d['costs'].shape == (stop - start, 18) and torch.isfinite(d['costs']).all()
        costs[start:stop] = d['costs']
        hashes[path.name] = sha(path)
        expected_start = stop
        if index % 200 == 0:
            print(f'Verified cost chunks {index + 1}/{len(files)}', flush=True)
    assert expected_start == n == 212553
    routes = torch.tensor([[0, a, b, d] for a in range(2) for b in range(3) for d in range(3)])
    starts = torch.arange(0, n, 16)
    counts = torch.minimum(torch.full_like(starts, 16), n - starts)
    # Accumulate means in FP64, preserve the underlying per-case FP32 costs.
    means = torch.stack([costs[s:s + 16].double().mean(0) for s in starts.tolist()])
    best = means.argmin(1)  # stable lexicographic first route breaks exact ties
    gaps = means.sort(dim=1).values[:, 1] - means.min(1).values
    value = {'case_costs': costs, 'costs': means, 'routes': routes, 'starts': starts,
             'counts': counts, 'best_route_indices': best, 'best_routes': routes[best],
             'runner_up_gap': gaps, 'source_contract': c['contract_sha256'],
             'train_identity': c['split_manifest']['dataframe_sha256']['train'],
             'batch_size': 16}
    torch.save(value, root / 'oracle/train.pt')
    membership = frame[['sample_id']].copy()
    membership.insert(0, 'position', np.arange(n))
    membership['logical_batch'] = np.arange(n) // 16
    membership.to_csv(root / 'oracle/train_members.csv', index=False)
    manifest = {'status': 'complete', 'created_at': time.time(), 'train_cases': n,
                'logical_batches': len(starts), 'last_batch_cases': int(counts[-1]),
                'batch_size': 16, 'routes': routes.tolist(), 'cost_source_run': str(old),
                'cost_source_chunk_hashes': hashes,
                'expert_sha256': contract['exported_expert_sha256'],
                'split_identity': value['train_identity'],
                'oracle_sha256': sha(root / 'oracle/train.pt'),
                'membership_sha256': sha(root / 'oracle/train_members.csv'),
                'target': 'lexicographic argmin of batch mean true-label DRG NLL',
                'mean_runner_up_gap': float(gaps.mean()),
                'route_counts': torch.bincount(best, minlength=18).tolist(),
                'compatibility_labels_used': False,
                'previous_casewise_targets_or_features_used': False}
    (root / 'oracle/train_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'cost_source_chunk_hashes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
