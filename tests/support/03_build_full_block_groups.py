#!/usr/bin/env python3
import argparse
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.io_utils import (
    atomic_torch_save,
    load_split_csv,
    parse_str_list,
    read_json,
    write_json,
)
from adaptive_blockwise.formulas import metric_kernel
from adaptive_blockwise.experiment_contract import block_groups, validate_partition_groups
from adaptive_blockwise.signals import (
    ExactSignalComputer,
    compute_directional_matrix,
    extract_signal_cache,
    merge_signal_caches,
    load_signal_part,
    load_tokenizer,
    make_dataset,
    symmetrize_and_normalize_memmap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build exact full-data block groups and compatibility statistics.")
    parser.add_argument("--config", type=Path, default=ROOT / "config_openjev.json")
    parser.add_argument("--compat_device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument('--parallel-gpus')
    parser.add_argument('--block-id', type=int, choices=range(4))
    parser.add_argument('--signature-only', action='store_true')
    parser.add_argument('--groups-only', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    return parser.parse_args()


def num_labels(config):
    return int(pd.read_csv(config["id2label_csv"])["label"].nunique())


def exact_spectral(compatibility, groups, seed):
    if groups == 1:
        return np.zeros(compatibility.shape[0], dtype=np.int64)
    affinity = np.asarray(compatibility, dtype=np.float32).copy()
    minimum = float(affinity.min())
    if minimum < 0:
        affinity -= minimum
    maximum = float(affinity.max())
    if maximum > 0:
        affinity /= maximum
    np.fill_diagonal(affinity, 1.0)
    labels = SpectralClustering(
        n_clusters=groups,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=20,
        random_state=seed,
    ).fit_predict(affinity)
    if len(np.unique(labels)) != groups:
        raise RuntimeError("Full-data spectral clustering produced an empty expert group")
    return labels.astype(np.int64)


def accumulate_block_layer(normalized_path, output_path, weight, first_layer, chunk_rows=256):
    matrix = np.load(normalized_path, mmap_mode="r")
    mode = "w+" if first_layer else "r+"
    output = np.lib.format.open_memmap(output_path, mode=mode, dtype=np.float32, shape=matrix.shape)
    for start in range(0, matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, matrix.shape[0])
        value = np.asarray(matrix[start:end], dtype=np.float32)
        if first_layer:
            output[start:end] = weight * value
        else:
            output[start:end] += weight * value
    output.flush()
    del output
    del matrix


def build_batch_manifest(train_df, batch_size):
    rows = []
    for batch_id, start in enumerate(range(0, len(train_df), batch_size)):
        end = min(start + batch_size, len(train_df))
        rows.append(
            {
                "batch_id": batch_id,
                "start_index": start,
                "end_index_exclusive": end,
                "num_examples": end - start,
                "first_sample_id": train_df.iloc[start]["sample_id"],
                "last_sample_id": train_df.iloc[end - 1]["sample_id"],
            }
        )
    return pd.DataFrame(rows)


def average_group_statistics(cache_dir, layer, labels, num_groups):
    manifest = read_json(Path(cache_dir) / "manifest.json")
    names = manifest["layer_modules"][str(layer)]
    fields = (
        "mean_gradient",
        "mean_gradient_gram",
        "mean_weighted_gradient",
        "mean_kernel",
    )
    sums = {
        group: {name: {field: None for field in fields} for name in names}
        for group in range(num_groups)
    }
    counts = {group: 0 for group in range(num_groups)}
    for record in manifest["parts"]:
        part = load_signal_part(cache_dir, record, names)
        part_labels = torch.tensor(
            [int(labels[batch_id]) for batch_id in part["batch_ids"]],
            dtype=torch.long,
        )
        for group in range(num_groups):
            selected = part_labels == group
            selected_count = int(selected.sum().item())
            if selected_count == 0:
                continue
            counts[group] += selected_count
            for name in names:
                gradients = part["grads"][name][selected].cpu()
                metrics = part["metrics"][name][selected].cpu()
                inverses = part["metric_inverses"][name][selected].cpu()
                kernels = metric_kernel(metrics, inverses)
                values = {
                    "mean_gradient": gradients.sum(dim=0),
                    "mean_gradient_gram": torch.einsum(
                        "bor,bot->rt", gradients, gradients
                    ),
                    "mean_weighted_gradient": torch.einsum(
                        "bor,brs->os", gradients, kernels
                    ),
                    "mean_kernel": kernels.sum(dim=0),
                }
                for field, value in values.items():
                    previous = sums[group][name][field]
                    sums[group][name][field] = (
                        value.clone() if previous is None else previous + value
                    )
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Empty compatibility-statistics group at layer {layer}: {counts}")
    statistics = {
        group: {
            name: {
                field: value / float(counts[group])
                for field, value in field_sums.items()
            }
            for name, field_sums in module_sums.items()
        }
        for group, module_sums in sums.items()
    }
    return statistics, counts, names


def main():
    args = parse_args()
    config = read_json(args.config)
    run_dir = Path(config["run_dir"])
    split_manifest = read_json(run_dir / "00_splits" / "manifest.json")
    partition = read_json(run_dir / "02_partition" / "partition.json")
    validate_partition_groups(config, partition)
    output_dir = run_dir / "03_full_groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "_block_signal_cache"
    matrix_root = output_dir / "_working_matrices"
    statistics_root = output_dir / "prototypes"
    cache_root.mkdir(exist_ok=True)
    matrix_root.mkdir(exist_ok=True)
    statistics_root.mkdir(exist_ok=True)

    tokenizer = load_tokenizer(config["model_name_or_path"])
    token_cache = output_dir / '_tokenized_train'
    token_contract = {'train_identity_sha256': split_manifest['dataframe_sha256']['train'],
                      'model': config['model_name_or_path'], 'max_length': config['max_length']}
    from datasets import load_from_disk
    if (token_cache / 'dataset_info.json').exists():
        if read_json(output_dir / 'tokenized_contract.json') != token_contract:
            raise ValueError('Tokenized training data contract changed')
        train_ds = load_from_disk(str(token_cache))
        # datasets.load_from_disk reapplies formatting and hashes that operation,
        # even for the default format. Preserve the saved tokenization identity
        # so caches computed before disk serialization can resume unchanged.
        train_ds._fingerprint = read_json(token_cache / 'state.json')['_fingerprint']
        train_df = pd.read_csv(split_manifest['paths']['train'], usecols=['sample_id'])
    else:
        train_df = load_split_csv(split_manifest['paths']['train'])
        train_ds = make_dataset(train_df, tokenizer, int(config['max_length']))
        if args.parallel_gpus:
            train_ds.save_to_disk(str(token_cache))
            write_json(output_dir / 'tokenized_contract.json', token_contract)
    batch_size = int(config["full_signature_batch_size"])
    batch_manifest = build_batch_manifest(train_df, batch_size)
    if args.block_id is None:
        batch_manifest.to_csv(output_dir / "full_batch_manifest.csv", index=False)

    if args.parallel_gpus:
        gpu_ids = args.parallel_gpus.split(',')
        if len(gpu_ids) not in (4, 8) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError('Parallel signatures require four or eight distinct GPU IDs')
        shards = len(gpu_ids)//4
        jobs = []
        def launch(block_id, gpu, extra, suffix):
            log = (output_dir / f'block_{block_id}{suffix}_worker.log').open('a', buffering=1)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--config', str(args.config.resolve()), '--block-id', str(block_id),
                '--compat_device', args.compat_device, *extra], env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((block_id, process, log))
        def finish_jobs():
            while any(process.poll() is None for _, process, _ in jobs):
                for block_id, process, _ in jobs:
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f'Signature/group worker block {block_id} failed; see worker log')
                time.sleep(5)
            if any(process.returncode for _, process, _ in jobs):
                raise RuntimeError('A signature/group worker failed')
            for _, _, log in jobs: log.close()
            jobs.clear()
        try:
            if shards == 1:
                for b, gpu in enumerate(gpu_ids): launch(b, gpu, [], '')
                finish_jobs()
            else:
                for b in range(4):
                    if (output_dir/f'block_{b}_complete.json').exists(): continue
                    for shard in range(shards):
                        launch(b, gpu_ids[b*shards+shard], ['--signature-only', '--shard-index', str(shard),
                            '--num-shards', str(shards)], f'_shard_{shard}')
                finish_jobs()
                for b in range(4):
                    if (output_dir/f'block_{b}_complete.json').exists(): continue
                    merge_signal_caches(cache_root/f'block_{b}_all_layers',
                        [cache_root/f'block_{b}_shards/shard_{i}' for i in range(shards)])
                    launch(b, gpu_ids[b*shards], ['--groups-only'], '_groups')
                finish_jobs()
        finally:
            for _, process, log in jobs:
                if process.poll() is None:
                    process.terminate(); process.wait()
                log.close()
        rows = batch_manifest.copy(); normalization = {}; statistics = {}
        for block_id in range(4):
            done = read_json(output_dir / f'block_{block_id}_complete.json')
            rows[f'block_{block_id}_group'] = done['labels']
            normalization.update(done['normalization']); statistics[str(block_id)] = done['statistics_paths']
        rows.to_csv(output_dir / 'full_batch_groups.csv', index=False)
        write_json(output_dir / 'manifest.json', {
            'format': 'openjev_full_train16_groups_v1', 'num_groups': config['num_groups'],
            'train_identity_sha256': split_manifest['dataframe_sha256']['train'],
            'signature_batch_size': batch_size, 'num_train_samples': len(train_df),
            'num_signature_batches': len(batch_manifest), 'layer_normalization': normalization,
            'block_layers': {str(b['block_id']): b['layers'] for b in partition['blocks']},
            'group_statistics_paths': statistics, 'parallel_workers': len(gpu_ids),
            'signature_shards_per_block': shards,
            'compatibility_formula': 'equal mean of all block layers normalized symmetric compatibility'})
        print(f'All train16 block signature workers completed across {len(gpu_ids)} GPUs', flush=True)
        return

    computer = ExactSignalComputer(
        config["model_name_or_path"],
        num_labels(config),
        tokenizer,
        run_dir / "01_calibration" / "probe_coordinate.pt",
        int(config["seed"]),
        rank=int(config["probe_lora_rank"]),
        alpha=int(config["probe_lora_alpha"]),
        ridge=float(config["ridge"]),
        target_modules=parse_str_list(config["target_modules"]),
        no_4bit=bool(config.get("no_4bit", False)),
    )
    computer.microbatch_size = int(config.get('signature_microbatch', 4))
    normalization = {}
    flattened_layers = [int(layer) for block in partition["blocks"] for layer in block["layers"]]
    expected_layers = list(range(int(config["num_layers"])))
    if sorted(flattened_layers) != expected_layers:
        raise ValueError(
            "Partition blocks must cover every configured layer exactly once; "
            f"expected {expected_layers}, got {flattened_layers}"
        )
    statistics_manifest = {}
    group_rows = batch_manifest.copy()
    for block in partition["blocks"]:
        block_id = int(block["block_id"])
        if args.block_id is not None and block_id != args.block_id:
            continue
        layers = [int(layer) for layer in block["layers"]]
        if not layers:
            raise ValueError(f"Block {block_id} contains no layers")
        completed_path = output_dir / f"block_{block_id}_complete.json"
        if completed_path.exists():
            completed = read_json(completed_path)
            if completed["signature_batch_size"] != batch_size or completed["num_groups"] != block_groups(config["num_groups"], block_id):
                raise ValueError("Completed block does not match current batch/K")
            if len(completed["labels"]) != len(batch_manifest) or any(
                not Path(path).exists() for path in completed["statistics_paths"].values()
            ):
                raise ValueError("Completed block artifacts are incomplete")
            group_rows[f"block_{block_id}_group"] = completed["labels"]
            normalization.update(completed["normalization"])
            statistics_manifest[str(block_id)] = completed["statistics_paths"]
            print(f"Reusing completed full-data block {block_id}", flush=True)
            continue
        block_cache = cache_root / f"block_{block_id}_all_layers"
        # At B=0, computing all block-layer gradients in one
        # backward gives exactly the same partial derivatives as separate
        # backwards, while avoiding repeated full Qwen3 passes.
        if args.signature_only:
            if not 0 <= args.shard_index < args.num_shards: raise ValueError('Invalid signature shard')
            count = len(batch_manifest)
            shard_cache = cache_root/f'block_{block_id}_shards/shard_{args.shard_index}'
            extract_signal_cache(computer, train_ds, tokenizer, layers, shard_cache,
                batch_size=batch_size, cache_batches=int(config['signature_cache_batches']),
                batch_start=count*args.shard_index//args.num_shards,
                batch_stop=count*(args.shard_index+1)//args.num_shards)
            computer.close()
            return
        if not args.groups_only:
            extract_signal_cache(
            computer,
            train_ds,
            tokenizer,
            layers,
            block_cache,
            batch_size=batch_size,
            cache_batches=int(config["signature_cache_batches"]),
            )
        elif not read_json(block_cache/'manifest.json')['complete']:
            raise ValueError('Grouping requires complete merged signature cache')
        compatibility_path = matrix_root / f"block_{block_id}_compatibility.npy"
        layer_weight = 1.0 / float(len(layers))
        for layer_index, layer in enumerate(layers):
            directional_path = matrix_root / f"layer_{layer:02d}_directional.npy"
            normalized_path = matrix_root / f"layer_{layer:02d}_normalized.npy"
            compute_directional_matrix(
                block_cache,
                block_cache,
                layer,
                directional_path,
                args.compat_device,
                score_block_batches=int(config["compatibility_score_block_batches"]),
                preload=bool(config["compatibility_preload"]),
            )
            normalized, mean, std = symmetrize_and_normalize_memmap(directional_path, normalized_path)
            del normalized
            directional_path.unlink()
            normalization[str(layer)] = {"off_diagonal_mean": mean, "off_diagonal_std": std}
            accumulate_block_layer(
                normalized_path,
                compatibility_path,
                layer_weight,
                first_layer=layer_index == 0,
            )
            normalized_path.unlink()
        compatibility = np.load(compatibility_path, mmap_mode="r")
        labels = exact_spectral(compatibility, block_groups(config["num_groups"], block_id), int(config["seed"]) + block_id)
        del compatibility
        compatibility_path.unlink()
        group_rows[f"block_{block_id}_group"] = labels
        statistics_manifest[str(block_id)] = {}
        for layer in layers:
            statistics, counts, module_names = average_group_statistics(
                block_cache,
                layer,
                labels,
                block_groups(config["num_groups"], block_id),
            )
            path = statistics_root / f"block_{block_id}_layer_{layer:02d}.pt"
            atomic_torch_save(
                {
                    "format": "block_compatibility_statistics_v3",
                    "block_id": block_id,
                    "layer": layer,
                    "module_names": module_names,
                    "counts": counts,
                    "statistics": statistics,
                },
                path,
            )
            statistics_manifest[str(block_id)][str(layer)] = str(path)
        write_json(completed_path, {
            "signature_batch_size": batch_size,
            "num_groups": block_groups(config["num_groups"], block_id),
            "labels": labels.tolist(),
            "normalization": {str(layer): normalization[str(layer)] for layer in layers},
            "statistics_paths": statistics_manifest[str(block_id)],
        })
        shutil.rmtree(block_cache)
        shard_cache_root = cache_root/f'block_{block_id}_shards'
        if shard_cache_root.exists(): shutil.rmtree(shard_cache_root)
        if args.block_id is None:
            group_rows.to_csv(output_dir / "full_batch_groups.partial.csv", index=False)
        print(f"completed full compatibility and group statistics for block {block_id}", flush=True)
    if args.block_id is not None:
        computer.close()
        return
    group_rows.to_csv(output_dir / "full_batch_groups.csv", index=False)
    (output_dir / "full_batch_groups.partial.csv").unlink(missing_ok=True)
    computer.close()
    if cache_root.exists() and not any(cache_root.iterdir()):
        cache_root.rmdir()
    if matrix_root.exists() and not any(matrix_root.iterdir()):
        matrix_root.rmdir()

    write_json(
        output_dir / "manifest.json",
        {
            "format": "adaptive_blockwise_full_groups_v3",
            "train_identity_sha256": split_manifest["dataframe_sha256"]["train"],
            "num_train_samples": len(train_df),
            "signature_batch_size": batch_size,
            "num_signature_batches": len(batch_manifest),
            "num_groups": config["num_groups"],
            "block_layers": {str(block["block_id"]): [int(layer) for layer in block["layers"]] for block in partition["blocks"]},
            "layer_normalization": normalization,
            "group_statistics_paths": statistics_manifest,
            "compatibility_formula": "equal mean of every block layer's normalized exact symmetric compatibility",
            "group_statistics_formula": "per-group means of gradient, gradient Gram, metric-weighted gradient, and metric kernel; sufficient for exact mean symmetric sample-to-group compatibility",
            "storage_policy": "one block's all-layer signals are cached at a time; scratch and fused compatibility matrices are deleted immediately after clustering, and signals are deleted after group statistics",
            "peak_signature_policy": {
                "calibration": "all configured layer matrices only; raw signatures deleted after each layer matrix",
                "full_data": "one block's all-layer cache plus one block accumulator and one layer scratch matrix",
                "signature_cache_batches": int(config["signature_cache_batches"]),
                "compatibility_score_block_batches": int(config["compatibility_score_block_batches"]),
                "compatibility_preload": bool(config["compatibility_preload"]),
            },
        },
    )
    print(f"Wrote full block groups and compatibility statistics to {output_dir}")


if __name__ == "__main__":
    main()
