#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.cluster import SpectralClustering

ROOT = Path(__file__).resolve().parents[1]
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
    extract_signal_cache_distributed,
    initialize_signal_distributed,
    load_signal_part,
    load_tokenizer,
    make_dataset,
    symmetrize_and_normalize_memmap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build exact full-data block groups and compatibility statistics.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--compat_device", choices=["auto", "cuda", "cpu"], default="auto")
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
    rank, world, local_rank = initialize_signal_distributed()
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

    train_df = load_split_csv(split_manifest["paths"]["train"])
    tokenizer = load_tokenizer(config["model_name_or_path"])
    train_ds = make_dataset(train_df, tokenizer, int(config["max_length"]))
    batch_size = int(config["full_signature_batch_size"])
    batch_manifest = build_batch_manifest(train_df, batch_size)
    if rank == 0:
        batch_manifest.to_csv(output_dir / "full_batch_manifest.csv", index=False)

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
        device_map={"": local_rank} if world > 1 else "auto",
    )
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
        extractor = extract_signal_cache_distributed if world > 1 else extract_signal_cache
        extractor(
            computer,
            train_ds,
            tokenizer,
            layers,
            block_cache,
            batch_size=batch_size,
            cache_batches=int(config["signature_cache_batches"]),
        )
        if rank != 0:
            dist.barrier()
            continue
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
        group_rows.to_csv(output_dir / "full_batch_groups.partial.csv", index=False)
        print(f"completed full compatibility and group statistics for block {block_id}", flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
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
            "signature_world_size": world,
            "signature_local_batch_size": batch_size // world,
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
