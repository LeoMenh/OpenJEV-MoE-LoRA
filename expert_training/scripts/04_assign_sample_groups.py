#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.formulas import soft_assignment, symmetric_group_mean_scores
from adaptive_blockwise.io_utils import atomic_torch_save, dataframe_identity, load_split_csv, parse_str_list, read_json, write_json
from adaptive_blockwise.signals import (ExactSignalComputer, load_tokenizer, make_dataset,
                                       compute_distributed_on_batch, initialize_signal_distributed)
from adaptive_blockwise.experiment_contract import block_groups, validate_partition_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize batch-level block groups for every split.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def num_labels(config):
    return int(pd.read_csv(config["id2label_csv"])["label"].nunique())


def load_group_statistics(full_dir, partition):
    blocks = {}
    for block in partition["blocks"]:
        block_id = int(block["block_id"])
        blocks[block_id] = {}
        for layer in block["layers"]:
            path = full_dir / "prototypes" / f"block_{block_id}_layer_{int(layer):02d}.pt"
            payload = torch.load(path, map_location="cpu")
            if payload.get("format") != "block_compatibility_statistics_v3":
                raise RuntimeError(f"Outdated Stage 03 group statistics: {path}")
            blocks[block_id][int(layer)] = payload["statistics"]
    return blocks


def stack_group_statistics(statistics, device, num_groups):
    stacked = {}
    for block, layer_map in statistics.items():
        stacked[block] = {}
        for layer, group_map in layer_map.items():
            module_names = group_map[0].keys()
            stacked[block][layer] = {
                name: {
                    field: torch.stack(
                        [group_map[group][name][field] for group in range(block_groups(num_groups, block))]
                    ).to(device)
                    for field in group_map[0][name]
                }
                for name in module_names
            }
    return stacked


def batch_lengths(num_samples, batch_size):
    return [
        min(batch_size, num_samples - start)
        for start in range(0, num_samples, batch_size)
    ]


def training_batch_assignments(full_dir, train_frame, partition, batch_size, num_groups):
    groups = pd.read_csv(full_dir / "full_batch_groups.csv")
    lengths = batch_lengths(len(train_frame), batch_size)
    if len(groups) != len(lengths):
        raise ValueError(
            f"Stage 03 batch count {len(groups)} does not match train split batch count {len(lengths)}"
        )
    if "num_examples" in groups and groups["num_examples"].astype(int).tolist() != lengths:
        raise ValueError("Stage 03 batch boundaries do not match the immutable train split")

    assignments = {}
    for block in partition["blocks"]:
        block_id = int(block["block_id"])
        column = f"block_{block_id}_group"
        if column not in groups:
            raise ValueError(f"Stage 03 group table is missing {column}")
        labels = groups[column].astype(int).to_numpy(copy=True)
        groups_in_block = block_groups(num_groups, block_id)
        if labels.min() < 0 or labels.max() >= groups_in_block:
            raise ValueError(f"Invalid Stage 03 labels in {column}")
        assignments[block_id] = {
            "labels": labels,
            "confidence": np.ones(len(labels), dtype=np.float32),
            "probabilities": np.eye(groups_in_block, dtype=np.float32)[labels],
            "scores": None,
        }
    return assignments


def score_out_of_sample_batches(
    computer,
    frame,
    tokenizer,
    group_statistics,
    normalization,
    partition,
    config,
    split_name,
):
    batch_size = int(config["full_signature_batch_size"])
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    dataset = make_dataset(frame, tokenizer, int(config["max_length"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    all_layers = sorted({int(layer) for block in partition["blocks"] for layer in block["layers"]})
    computer.set_active_layers(all_layers)
    device = next(computer.model.parameters()).device
    device_statistics = stack_group_statistics(
        group_statistics, device, config["num_groups"]
    )
    score_rows = {int(block["block_id"]): [] for block in partition["blocks"]}
    checkpoint_path = Path(config["run_dir"]) / "04_sample_groups" / f"{split_name}_scores.pt"
    identity = dataframe_identity(frame)
    completed = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["batch_size"] != batch_size or saved["num_groups"] != config["num_groups"]:
            raise ValueError("Assignment checkpoint contract changed")
        score_rows = saved["score_rows"]
        completed = saved["completed"]
        if any(len(rows) != completed for rows in score_rows.values()):
            raise ValueError("Assignment checkpoint has inconsistent block lengths")
        print(f"Resuming {split_name} assignment at {completed}/{len(loader)} batches", flush=True)

    for step, batch in enumerate(loader, start=1):
        if step <= completed:
            continue
        compute = compute_distributed_on_batch if distributed else lambda computer, batch: computer.compute_on_batch(batch)
        gradients, metrics, inverses, loss = compute(computer, batch)
        if rank != 0:
            continue
        device_gradients = {name: value.unsqueeze(0).to(device) for name, value in gradients.items()}
        device_metrics = {name: value.unsqueeze(0).to(device) for name, value in metrics.items()}
        device_inverses = {name: value.unsqueeze(0).to(device) for name, value in inverses.items()}

        for block in partition["blocks"]:
            block_id = int(block["block_id"])
            scores = torch.zeros(
                (1, block_groups(config["num_groups"], block_id)),
                device=device,
                dtype=torch.float32,
            )
            layers = [int(layer) for layer in block["layers"]]
            layer_weight = 1.0 / float(len(layers))
            for layer in layers:
                layer_scores = torch.zeros_like(scores)
                for name in computer.layer_modules[layer]:
                    statistics = device_statistics[block_id][layer][name]
                    layer_scores += symmetric_group_mean_scores(
                        device_gradients[name],
                        device_metrics[name],
                        device_inverses[name],
                        statistics["mean_gradient"],
                        statistics["mean_gradient_gram"],
                        statistics["mean_weighted_gradient"],
                        statistics["mean_kernel"],
                    )
                layer_normalization = normalization[str(layer)]
                layer_scores = (
                    layer_scores - float(layer_normalization["off_diagonal_mean"])
                ) / (float(layer_normalization["off_diagonal_std"]) + 1e-8)
                scores += layer_weight * layer_scores
            if not torch.isfinite(scores).all():
                raise FloatingPointError(
                    f"Non-finite batch compatibility scores in {split_name}, block {block_id}, batch {step - 1}"
                )
            score_rows[block_id].append(scores[0].detach().cpu())

        if step == 1 or step % 25 == 0 or step == len(loader):
            atomic_torch_save({
                "identity": identity, "batch_size": batch_size,
                "num_groups": config["num_groups"], "completed": step, "score_rows": score_rows,
            }, checkpoint_path)
            print(
                f"batch assignment {split_name}: {step}/{len(loader)} batches; loss={loss:.6f}",
                flush=True,
            )

    if distributed:
        dist.barrier()
    if rank != 0:
        return None
    assignments = {}
    for block_id, rows in score_rows.items():
        scores = torch.stack(rows)
        probabilities, labels, confidence = soft_assignment(
            scores, float(config["temperature"])
        )
        assignments[block_id] = {
            "labels": labels.numpy(),
            "confidence": confidence.numpy(),
            "probabilities": probabilities.numpy(),
            "scores": scores.numpy(),
        }
    return assignments


def quantiles(values):
    return {
        name: float(value)
        for name, value in zip(
            ("p01", "p05", "p50", "p95", "p99"),
            np.quantile(values, (0.01, 0.05, 0.5, 0.95, 0.99)),
        )
    }


def materialize_batch_assignments(
    frame,
    assignments,
    partition,
    batch_size,
    num_groups,
    output_path,
    assignment_source,
):
    lengths = batch_lengths(len(frame), batch_size)
    result = frame.copy()
    empty_groups = {}
    diagnostic_blocks = {}

    for block in partition["blocks"]:
        block_id = int(block["block_id"])
        block_assignment = assignments[block_id]
        labels = np.asarray(block_assignment["labels"], dtype=np.int64)
        confidence = np.asarray(block_assignment["confidence"], dtype=np.float32)
        probabilities = np.asarray(block_assignment["probabilities"], dtype=np.float32)
        groups_in_block = block_groups(num_groups, block_id)
        if labels.shape != (len(lengths),) or probabilities.shape != (len(lengths), groups_in_block):
            raise RuntimeError(f"Batch assignment shape mismatch for block {block_id}")

        probability_strings = np.array(
            [" ".join(f"{float(value):.8f}" for value in row) for row in probabilities],
            dtype=object,
        )
        result[f"block_{block_id}_group"] = np.repeat(labels, lengths)
        result[f"block_{block_id}_confidence"] = np.repeat(confidence, lengths)
        result[f"block_{block_id}_probabilities"] = np.repeat(probability_strings, lengths)

        batch_counts = np.bincount(labels, minlength=groups_in_block)
        sample_counts = np.bincount(
            result[f"block_{block_id}_group"].to_numpy(dtype=np.int64),
            minlength=groups_in_block,
        )
        missing = np.flatnonzero(sample_counts == 0).astype(int).tolist()
        if missing:
            empty_groups[str(block_id)] = missing
            print(
                f"WARNING: {output_path.stem}: block {block_id} has empty groups {missing}; "
                "assignments are retained and downstream stages will continue",
                flush=True,
            )
        print(
            f"{output_path.stem}: block {block_id} batch counts {batch_counts.tolist()}; "
            f"sample counts {sample_counts.tolist()}",
            flush=True,
        )

        block_diagnostics = {
            "num_batches": int(len(labels)),
            "num_samples": int(len(frame)),
            "batch_group_counts": [int(value) for value in batch_counts],
            "sample_group_counts": [int(value) for value in sample_counts],
            "confidence": quantiles(confidence),
        }
        scores = block_assignment["scores"]
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32)
            ordered = np.sort(scores, axis=1)
            block_diagnostics["per_group_scores"] = {
                str(group): {
                    "mean": float(scores[:, group].mean()),
                    "std": float(scores[:, group].std()),
                    "min": float(scores[:, group].min()),
                    "max": float(scores[:, group].max()),
                }
                for group in range(groups_in_block)
            }
            block_diagnostics["raw_top1_top2_margin"] = (
                quantiles(ordered[:, -1] - ordered[:, -2]) if groups_in_block > 1 else None
            )
        diagnostic_blocks[str(block_id)] = block_diagnostics

    result.to_csv(output_path, index=False)
    diagnostics_path = output_path.with_name(f"{output_path.stem}_score_diagnostics.json")
    write_json(
        diagnostics_path,
        {
            "format": "batch_group_assignment_diagnostics_v2",
            "split": output_path.stem.removesuffix("_with_block_groups"),
            "assignment_unit": "signature_batch",
            "signature_batch_size": batch_size,
            "assignment_source": assignment_source,
            "blocks": diagnostic_blocks,
        },
    )
    return empty_groups


def main():
    args = parse_args()
    config = read_json(args.config)
    rank, world, local_rank = initialize_signal_distributed()
    run_dir = Path(config["run_dir"])
    split_manifest = read_json(run_dir / "00_splits" / "manifest.json")
    partition = read_json(run_dir / "02_partition" / "partition.json")
    validate_partition_groups(config, partition)
    full_dir = run_dir / "03_full_groups"
    full_manifest = read_json(full_dir / "manifest.json")
    if full_manifest.get("format") != "adaptive_blockwise_full_groups_v3":
        raise RuntimeError(
            "Batch-level Stage 04 requires symmetric all-layer Stage 03 artifacts "
            "(adaptive_blockwise_full_groups_v3). Delete Stage 03-07 outputs and rerun from START_STAGE=3."
        )

    output_dir = run_dir / "04_sample_groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and any(output_dir.glob("*_with_block_groups.csv")) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains assignments; pass --overwrite")

    batch_size = int(config["full_signature_batch_size"])
    groups = config["num_groups"]
    if full_manifest["signature_batch_size"] != batch_size or full_manifest.get("num_groups", groups) != groups:
        raise ValueError("Stage 03 signature batch/K do not match the assignment configuration")
    train_frame = load_split_csv(split_manifest["paths"]["train"])
    train_assignments = training_batch_assignments(
        full_dir, train_frame, partition, batch_size, groups
    )
    empty_groups_by_split = {
        "train": materialize_batch_assignments(
            train_frame,
            train_assignments,
            partition,
            batch_size,
            groups,
            output_dir / "train_with_block_groups.csv",
            "direct Stage-03 spectral batch labels; one-hot probability and confidence=1",
        )
    } if rank == 0 else {}

    tokenizer = load_tokenizer(config["model_name_or_path"])
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
    group_statistics = load_group_statistics(full_dir, partition)
    normalization = full_manifest["layer_normalization"]
    for split in ("validation", "test"):
        frame = load_split_csv(split_manifest["paths"][split])
        assignments = score_out_of_sample_batches(
            computer,
            frame,
            tokenizer,
            group_statistics,
            normalization,
            partition,
            config,
            split,
        )
        if rank != 0:
            continue
        empty_groups_by_split[split] = materialize_batch_assignments(
            frame,
            assignments,
            partition,
            batch_size,
            groups,
            output_dir / f"{split}_with_block_groups.csv",
            "batch-level mean symmetric compatibility to Stage-03 train spectral groups",
        )
    computer.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return

    write_json(
        output_dir / "manifest.json",
        {
            "format": "batch_level_block_assignment_v4",
            "assignment_unit": "signature_batch",
            "signature_batch_size": batch_size,
            "train_assignment": "direct copy of Stage-03 spectral batch labels to every sample in the batch",
            "num_groups": groups,
            "validation_test_assignment": "argmax of all-layer normalized mean symmetric batch compatibility to Stage-03 train groups",
            "train_confidence": "1.0 with one-hot probabilities because spectral labels are used directly",
            "temperature": float(config["temperature"]),
            "oracle_test_note": "test batch assignment requires true DRG labels for gradient signatures and is evaluation-only",
            "split_identity_sha256": {
                split: split_manifest["dataframe_sha256"][split]
                for split in ("train", "validation", "test")
            },
            "paths": {
                split: str(output_dir / f"{split}_with_block_groups.csv")
                for split in ("train", "validation", "test")
            },
            "score_diagnostic_paths": {
                split: str(output_dir / f"{split}_with_block_groups_score_diagnostics.json")
                for split in ("train", "validation", "test")
            },
            "empty_groups_by_split": empty_groups_by_split,
            "empty_group_policy": "retain assignments, create no synthetic batches, and continue downstream with zero class weight for absent train groups",
        },
    )
    print(f"Wrote batch-level assignments to {output_dir}")


if __name__ == "__main__":
    main()
