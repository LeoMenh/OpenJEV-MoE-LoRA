#!/usr/bin/env python3
"""Matched casewise soft MLP/OpenJev routers supervised by fixed batch NLL labels.

The user explicitly chose copied batch targets and casewise soft execution.
The second phase therefore refreshes *soft* prefixes without relabeling; it
must not be described as the methods document's hard conditional R-B.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'backbone'))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_blockwise.io_utils import (atomic_torch_save, dataframe_identity,
    load_split_csv, seed_everything, write_json)
from adaptive_blockwise.joint_moe import SampleRouter
from soft_openjev_runtime import ExpertPool, SoftRouterBank


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def frame_for(c, split):
    if split not in ('train', 'validation'): raise ValueError('No test access')
    f = load_split_csv(c['split_manifest']['paths'][split])
    if dataframe_identity(f) != c['split_manifest']['dataframe_sha256'][split]:
        raise ValueError('Split changed')
    return f


class MLPBank(nn.Module):
    def __init__(self, c, device):
        super().__init__()
        self.features = nn.ModuleDict({str(b): SampleRouter(5120, c['source']['num_groups'][str(b)], c['mlp_width'])
                                      for b in (1, 2, 3)}).to(device)
        self.set_blocks((1, 2, 3))

    def forward(self, block, z): return self.features[str(block)](z)

    def set_blocks(self, blocks):
        self.requires_grad_(False)
        for b in blocks: self.features[str(b)].requires_grad_(True)

    def adapted_state(self): return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_adapted(self, state): self.load_state_dict(state, strict=True)

    def optimizer(self, c):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                lr=c['mlp_lr'], weight_decay=0)


def make_bank(c, method, device):
    seed_everything(c['seed'])
    return MLPBank(c, device) if method == 'mlp' else SoftRouterBank(c, 5120, device)


class PrefixCaptured(Exception): pass


@torch.no_grad()
def capture(pool, frame, block, bank=None, routes=None, all_blocks=False):
    # Existing entry hook first captures the correct hidden state, then stop
    # before this block's body. No full-dimensional token cache is retained.
    layer = pool.core.decoder_layers[10 * block]
    def stop(_module, _args): raise PrefixCaptured()
    handle = layer.register_forward_pre_hook(stop)
    try:
        forced = torch.tensor(routes, device=pool.device) if routes is not None else None
        try: pool(pool.inputs(frame), bank=bank, forced=forced, capture=True)
        except PrefixCaptured: pass
        else: raise RuntimeError('Capture hook not reached')
        blocks = (1, 2, 3) if all_blocks else (block,)
        return {str(b): pool.features[b].to(torch.bfloat16) for b in blocks}
    finally: handle.remove()


def feature_dir(root, phase, method=None, block=None):
    if phase == 'soft' and block == 1:
        # Block 1 always follows the single fixed expert in block 0.
        return root / 'features/optimal'
    return root / 'features' / ('optimal' if phase == 'optimal' else f'soft_{method}_block_{block}')


def extract(c, device, rank, world, phase, method=None, block=None):
    root = Path(c['run_dir']); frame = frame_for(c, 'train')
    oracle = torch.load(root / 'oracle/train.pt', map_location='cpu', weights_only=True)
    target_routes = oracle['best_routes'].repeat_interleave(16, dim=0)[:len(frame)]
    pool = ExpertPool(c, device)
    bank = None; upstream = None
    if phase == 'soft':
        bank = make_bank(c, method, device)
        chosen = root / method / 'working.pt'
        bank.load_adapted(torch.load(chosen, map_location='cpu', weights_only=True)['state'])
        bank.eval(); bank.requires_grad_(False); upstream = sha(chosen)
    out = feature_dir(root, phase, method, block); out.mkdir(parents=True, exist_ok=True)
    total = math.ceil(len(frame) / 64); done = 0; started = time.time()
    for index in range(rank, total, world):
        path = out / f'chunk_{index:05d}.pt'; start = index * 64; stop = min(start + 64, len(frame))
        if path.exists():
            row = torch.load(path, map_location='cpu', weights_only=True)
            assert row['contract'] == c['experiment_contract'] and row['upstream'] == upstream
            assert (row['start'], row['stop']) == (start, stop)
        else:
            pieces = []
            for i in range(start, stop, 8):
                end = min(i + 8, stop)
                pieces.append(capture(pool, frame.iloc[i:end], 3 if phase == 'optimal' else block,
                    bank=bank, routes=target_routes[i:end].numpy() if phase == 'optimal' else None,
                    all_blocks=phase == 'optimal'))
            row = {'start': start, 'stop': stop, 'contract': c['experiment_contract'],
                   'upstream': upstream, 'targets': target_routes[start:stop],
                   'features': {b: torch.cat([p[b] for p in pieces]) for b in pieces[0]}}
            atomic_torch_save(row, path)
        done += 1
        write_json(out / f'worker_{rank}.json', {'status': 'running', 'completed_chunks': done,
            'assigned_chunks': len(range(rank, total, world)), 'updated_at': time.time(),
            'elapsed_seconds': time.time() - started, 'upstream': upstream})
        if done % 25 == 1: print(f'Features {phase} {method} block={block} rank={rank} {done}', flush=True)
    write_json(out / f'worker_{rank}.json', {'status': 'complete', 'completed_chunks': done,
                                          'upstream': upstream, 'updated_at': time.time()})


def checkpoint(bank, path, **meta):
    atomic_torch_save({'state': bank.adapted_state(), 'meta': meta}, path)


def train(c, method, device, phase, only_block=None):
    root = Path(c['run_dir']); out = root / method; out.mkdir(exist_ok=True)
    bank = make_bank(c, method, device)
    working = out / 'working.pt'
    if working.exists(): bank.load_adapted(torch.load(working, map_location='cpu', weights_only=True)['state'])
    n = c['split_manifest']['counts']['train']; batch_count = math.ceil(n / 16)
    blocks = (1, 2, 3) if phase == 'optimal' else (only_block,)
    epochs = c['optimal_epochs'] if phase == 'optimal' else c['soft_prefix_epochs']
    for b in blocks:
        stage = f'{phase}_block_{b}'; final = out / f'{stage}.pt'
        if final.exists():
            bank.load_adapted(torch.load(final, map_location='cpu', weights_only=True)['state'])
            # Recover even if interrupted between final.pt and working.pt writes.
            checkpoint(bank, working, phase=phase, last_completed_block=b)
            continue
        source = feature_dir(root, phase, method, b)
        if len(list(source.glob('chunk_*.pt'))) != math.ceil(n / 64): raise ValueError('Features incomplete')
        bank.set_blocks((b,)); bank.train(); optimizer = bank.optimizer(c)
        resume = out / f'{stage}_resume.pt'; first_epoch = first_step = 0
        if resume.exists():
            d = torch.load(resume, map_location='cpu', weights_only=False)
            bank.load_adapted(d['state']); optimizer.load_state_dict(d['optimizer'])
            first_epoch, first_step = d['epoch'], d['next_step']
            torch.set_rng_state(d['rng']); torch.cuda.set_rng_state(d['cuda_rng'])
            if first_step >= batch_count:
                first_epoch += 1; first_step = 0
        for epoch in range(first_epoch, epochs):
            order = torch.randperm(batch_count, generator=torch.Generator().manual_seed(c['seed'] + b * 100 + epoch)).tolist()
            total_loss = total_correct = count = 0; cached_index = None; cached = None
            for step in range(first_step if epoch == first_epoch else 0, batch_count):
                batch_id = order[step]; start = batch_id * 16; stop = min(start + 16, n)
                index = start // 64
                if index != cached_index:
                    cached = torch.load(source / f'chunk_{index:05d}.pt', map_location='cpu', weights_only=True)
                    if cached['contract'] != c['experiment_contract']: raise ValueError('Stale features')
                    cached_index = index
                offset = start - cached['start']; size = stop - start
                z = cached['features'][str(b)][offset:offset + size].to(device).float()
                targets = cached['targets'][offset:offset + size, b].to(device)
                assert targets.unique().numel() == 1  # copied batch oracle, never casewise argmin
                optimizer.zero_grad(set_to_none=True); logits = bank(b, z)
                # Uniform logical batch sampling: exactly Eq.35 with m_S=1.
                # The tail has nine cases and is weighted by its true count.
                ce_sum = F.cross_entropy(logits, targets, reduction='sum')
                loss = ce_sum * (batch_count / n)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite oracle CE')
                loss.backward()
                params = [p for p in bank.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise RuntimeError('Missing/nonfinite router gradients')
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
                total_loss += float(ce_sum.detach()); total_correct += int((logits.argmax(-1) == targets).sum()); count += size
                if (step + 1) % 200 == 0 or step + 1 == batch_count:
                    atomic_torch_save({'state': bank.adapted_state(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'next_step': step + 1, 'rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state()}, resume)
                    write_json(out / 'progress.json', {'status': 'training', 'phase': phase, 'block': b,
                        'epoch': epoch + 1, 'step': step + 1, 'steps_per_epoch': batch_count,
                        'mean_oracle_CE_since_resume': total_loss / count,
                        'oracle_accuracy_since_resume': total_correct / count, 'updated_at': time.time()})
                    print(f'{method} {stage} epoch={epoch+1} step={step+1}/{batch_count} CE={total_loss/count:.6f} acc={total_correct/count:.4f}', flush=True)
            record = {'method': method, 'phase': phase, 'block': b, 'epoch': epoch + 1,
                      'mean_CE': total_loss / max(count, 1), 'accuracy': total_correct / max(count, 1),
                      'logged_cases': count, 'updated_at': time.time()}
            with (out / 'training_history.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            checkpoint(bank, out / f'{stage}_epoch_{epoch+1}.pt', **record)
        checkpoint(bank, final, phase=phase, block=b, epochs=epochs)
        checkpoint(bank, working, phase=phase, last_completed_block=b)
    if phase == 'optimal': checkpoint(bank, out / 'optimal.pt', phase=phase)
    if phase == 'soft' and only_block == 3: checkpoint(bank, out / 'soft.pt', phase=phase)


def preflight(c, device):
    root = Path(c['run_dir']); frame = frame_for(c, 'train').iloc[:16]
    oracle = torch.load(root / 'oracle/train.pt', map_location='cpu', weights_only=True)
    pool = ExpertPool(c, device)
    # Keep the same eight-case compute microbatches used by the verified costs.
    pieces = [capture(pool, frame.iloc[i:i+8], 3,
                      routes=oracle['best_routes'][0].expand(8, -1).numpy(), all_blocks=True)
              for i in (0, 8)]
    features = {b: torch.cat([p[b] for p in pieces]).to(device).float() for b in ('1','2','3')}
    expert_before = {k: v.detach().cpu().clone() for k, v in pool.core.state_dict().items()
                     if any(s in k for s in ('shared_', 'private_', '.score.'))}
    result = {'passed': False, 'casewise_soft': True, 'copied_batch_hard_targets': True,
              'experts_frozen': not any(p.requires_grad for p in pool.parameters()), 'methods': {}}
    for method in ('mlp', 'openjev'):
        bank = make_bank(c, method, device); bank.eval()
        checks = {}
        for b in (1,2,3):
            z = features[str(b)]
            with torch.no_grad():
                first = bank(b, z).softmax(-1)[0].clone()
                changed = z.clone(); changed[1:] = torch.randn_like(changed[1:])
                peer_error = float((first - bank(b, changed).softmax(-1)[0]).abs().max())
                if peer_error > 2e-6: raise ValueError('Cross-case routing dependence')
            bank.set_blocks((b,)); bank.train(); optimizer = bank.optimizer(c); losses = []
            before = {k: v.clone() for k, v in bank.adapted_state().items()}
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                target = oracle['best_routes'][0, b].expand(len(z)).to(device)
                loss = F.cross_entropy(bank(b, z), target)
                loss.backward()
                params = [p for p in bank.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise RuntimeError('Preflight missing/nonfinite gradients')
                grad_norm = sum(float(p.grad.square().sum()) for p in params) ** .5
                if grad_norm <= 0: raise RuntimeError('No router gradient')
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
            if all(torch.equal(v, before[k]) for k, v in bank.adapted_state().items()):
                raise RuntimeError('No optimizer update')
            bank.eval()
            checks[str(b)] = {'losses': losses, 'grad_norm': grad_norm,
                              'fixed_shape_peer_error': peer_error}
        result['methods'][method] = {'blocks': checks,
            'all_adapted_parameters': sum(v.numel() for v in bank.adapted_state().values())}
        del bank, optimizer, params
        import gc
        gc.collect(); torch.cuda.empty_cache()
    for k, v in expert_before.items():
        if not torch.equal(v, pool.core.state_dict()[k].cpu()): raise RuntimeError('Frozen expert changed')
    result.update(passed=True, updates_discarded=True, experts_and_head_unchanged=True,
                  finished_at=time.time())
    write_json(root / 'preflight.json', result); print(json.dumps(result), flush=True)


@torch.no_grad()
def evaluate(c, device, rank, world, method, phase):
    root = Path(c['run_dir']); frame = frame_for(c, 'validation'); pool = ExpertPool(c, device)
    bank = make_bank(c, method, device)
    chosen = root / method / f'{phase}.pt'
    bank.load_adapted(torch.load(chosen, map_location='cpu', weights_only=True)['state'])
    bank.eval(); bank.requires_grad_(False)
    checkpoint_hash = sha(chosen)
    out = root / 'evaluation' / method / phase; out.mkdir(parents=True, exist_ok=True)
    positions = []; scores = []; gates = {b: [] for b in (1,2,3)}
    total = math.ceil(len(frame)/8); completed = 0
    for index in range(rank, total, world):
        start = index*8; stop = min(start+8, len(frame)); sub = frame.iloc[start:stop]
        logits = pool(pool.inputs(sub), bank=bank)
        scores.append(logits.cpu().numpy()); positions.extend(range(start,stop))
        for b in gates: gates[b].append(pool.probabilities[b].cpu().numpy())
        completed += 1
        if completed % 50 == 0 or completed == 1:
            write_json(out/f'worker_{rank}.json', {'status':'running','completed_batches':completed,
                'assigned_batches':len(range(rank,total,world)),'updated_at':time.time()})
            print(f'Evaluation {method} {phase} rank={rank}: {completed}/{len(range(rank,total,world))}',flush=True)
    np.savez(out/f'rank_{rank}.npz', positions=np.asarray(positions),scores=np.concatenate(scores),
             **{f'block_{b}':np.concatenate(gates[b]) for b in gates})
    write_json(out/f'worker_{rank}.json', {'status':'complete','cases':len(positions),
        'checkpoint_sha256':checkpoint_hash,'updated_at':time.time()})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=['extract', 'train', 'evaluate', 'preflight'], required=True)
    p.add_argument('--phase', choices=['optimal', 'soft'], default='optimal')
    p.add_argument('--method', choices=['mlp', 'openjev']); p.add_argument('--block', type=int)
    p.add_argument('--rank', type=int, default=0); p.add_argument('--world', type=int, default=8)
    a = p.parse_args(); c = json.loads(a.config.read_text()); torch.set_num_threads(1)
    if c['routing_granularity'] != 'casewise_soft_copied_batch_target': raise ValueError('Unconfirmed policy')
    device = torch.device('cuda:0')
    if a.stage == 'extract': extract(c, device, a.rank, a.world, a.phase, a.method, a.block)
    elif a.stage == 'train': train(c, a.method, device, a.phase, a.block)
    elif a.stage == 'evaluate': evaluate(c, device, a.rank, a.world, a.method, a.phase)
    else: preflight(c, device)


if __name__ == '__main__': main()
