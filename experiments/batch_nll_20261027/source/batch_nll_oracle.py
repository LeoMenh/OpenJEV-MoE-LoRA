#!/usr/bin/env python3
"""Independent true-label oracle collection; never supplies labels to a router."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'backbone'))

import numpy as np
import torch
from torch.nn import functional as F

from adaptive_blockwise.io_utils import atomic_torch_save, dataframe_identity, load_split_csv, write_json
from prefix_cache import _prefix_cache
from soft_openjev_runtime import ExpertPool, routes_for


@torch.no_grad()
def costs_for(pool, frame, routes):
    inputs = pool.inputs(frame)
    labels = torch.tensor(frame.label.to_numpy(), device=pool.device)
    values = []
    with _prefix_cache(pool.core) as active:
        for route in routes:
            active[:] = list(route)
            forced = torch.tensor(route, device=pool.device).expand(len(frame), -1)
            logits = pool(inputs, forced=forced)
            values.append(F.cross_entropy(logits, labels, reduction='none').cpu())
    return torch.stack(values, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['verify', 'validation'], required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=8)
    args = parser.parse_args()
    c = json.loads(args.config.read_text()); root = Path(c['run_dir'])
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    pool = ExpertPool(c, device)
    routes = routes_for(c['source']['num_groups'])
    if args.stage == 'verify':
        frame = load_split_csv(c['split_manifest']['paths']['train'])
        assert dataframe_identity(frame) == c['split_manifest']['dataframe_sha256']['train']
        oracle = torch.load(root / 'oracle/train.pt', map_location='cpu', weights_only=True)
        live = torch.cat([costs_for(pool, frame.iloc[i:i+8], routes) for i in [0, 8]])
        expected = oracle['case_costs'][:16]
        error = float((live - expected).abs().max())
        if not torch.equal(live, expected):
            raise RuntimeError(f'Reused training cost mismatch: {error}')
        mean = live.double().mean(0)
        assert torch.equal(mean, oracle['costs'][0])
        assert torch.equal(torch.tensor(routes)[mean.argmin()], oracle['best_routes'][0])
        result = {'passed': True, 'cases': 16, 'route_count': len(routes),
                  'max_reused_cost_error': error, 'mean_NLL_oracle_exact': True,
                  'train_batch_size': 16, 'compute_microbatch': 8, 'max_length': 1024,
                  'best_route': oracle['best_routes'][0].tolist(),
                  'finished_at': time.time()}
        write_json(root / 'oracle/verification.json', result)
        print(json.dumps(result), flush=True)
        return
    frame = load_split_csv(c['split_manifest']['paths']['validation'])
    assert dataframe_identity(frame) == c['split_manifest']['dataframe_sha256']['validation']
    out = root / 'oracle/validation'; out.mkdir(exist_ok=True)
    chunk = 64; total = math.ceil(len(frame) / chunk); completed = 0; started = time.time()
    for index in range(args.rank, total, args.world):
        start, stop = index * chunk, min((index + 1) * chunk, len(frame))
        path = out / f'chunk_{index:05d}.pt'
        if path.exists():
            row = torch.load(path, map_location='cpu', weights_only=True)
            assert row['start'] == start and row['stop'] == stop
            assert row['source_contract'] == c['contract_sha256']
        else:
            costs = torch.cat([costs_for(pool, frame.iloc[i:min(i+8, stop)], routes)
                               for i in range(start, stop, 8)])
            assert torch.isfinite(costs).all()
            row = {'start': start, 'stop': stop, 'case_costs': costs,
                   'sample_ids': frame.iloc[start:stop].sample_id.astype(str).tolist(),
                   'source_contract': c['contract_sha256']}
            atomic_torch_save(row, path)
        completed += 1
        progress = {'status': 'running', 'completed_chunks': completed,
                    'assigned_chunks': len(range(args.rank, total, args.world)),
                    'updated_at': time.time(), 'elapsed_seconds': time.time() - started}
        write_json(out / f'worker_{args.rank}.json', progress)
        print(f'Validation oracle rank={args.rank} completed={completed}/{progress["assigned_chunks"]}', flush=True)
    write_json(out / f'worker_{args.rank}.json', {**progress, 'status': 'complete'})


if __name__ == '__main__':
    main()
