#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.formulas import (
    block_heterogeneity,
    bootstrap_layer_distances,
    solve_contiguous_partition,
)
from adaptive_blockwise.io_utils import read_json, write_json
from adaptive_blockwise.experiment_contract import validate_partition_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Select adaptive contiguous blocks with the exact Version-B objective.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    return parser.parse_args()


def spectral_labels(compatibility, groups, seed):
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
        random_state=seed,
        n_init=20,
    ).fit_predict(affinity)
    if len(np.unique(labels)) != groups:
        raise RuntimeError("Spectral clustering returned an empty calibration group")
    return labels.astype(np.int64)


def assign_validation(cross_compatibility, train_labels, groups):
    scores = np.stack(
        [cross_compatibility[train_labels == group].mean(axis=0) for group in range(groups)],
        axis=1,
    )
    return scores.argmax(axis=1).astype(np.int64)


def probe_cross_entropy(train_hidden, train_labels, val_hidden, val_labels, seed, config=None):
    if config is not None and "probe_train_batch_size" in config:
        generator = torch.Generator().manual_seed(seed)
        x = torch.as_tensor(train_hidden, dtype=torch.float32)
        y = torch.as_tensor(train_labels, dtype=torch.long)
        vx = torch.as_tensor(val_hidden, dtype=torch.float32)
        vy = torch.as_tensor(val_labels, dtype=torch.long)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            probe = torch.nn.Linear(x.shape[1], int(config.get("calibration_num_groups", 3)))
        optimizer = torch.optim.Adam(probe.parameters(), lr=float(config["probe_learning_rate"]))
        batch_size = int(config["probe_train_batch_size"])
        for _ in range(int(config["probe_epochs"])):
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(x), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(probe(x[indices]), y[indices])
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            loss_sum = 0.0
            for start in range(0, len(vx), batch_size):
                loss_sum += float(torch.nn.functional.cross_entropy(
                    probe(vx[start:start + batch_size]), vy[start:start + batch_size], reduction="sum"
                ))
        return loss_sum / len(vx)
    groups = sorted(np.unique(train_labels).tolist())
    if groups != [0, 1, 2] or sorted(np.unique(val_labels).tolist()) != [0, 1, 2]:
        return 50.0
    probe = LogisticRegression(
        penalty=None,
        max_iter=300,
        solver="lbfgs",
        random_state=seed,
    )
    probe.fit(train_hidden, train_labels)
    probabilities = probe.predict_proba(val_hidden)
    return float(log_loss(val_labels, probabilities, labels=[0, 1, 2]))


def block_key(start, end):
    return f"{start}-{end}"


def main():
    args = parse_args()
    config = read_json(args.config)
    run_dir = Path(config["run_dir"])
    calibration_dir = run_dir / "01_calibration"
    output_dir = run_dir / "02_partition"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_layers = int(config["num_layers"])
    groups = int(config.get("calibration_num_groups", config["num_groups"]))
    if groups != 3:
        raise ValueError("Version B fixes K=3")

    train_matrices = np.stack(
        [np.load(calibration_dir / f"layer_{layer:02d}" / "train_normalized_compatibility.npy") for layer in range(num_layers)]
    )
    cross_matrices = np.stack(
        [
            np.load(calibration_dir / f"layer_{layer:02d}" / "train_validation_normalized_compatibility.npy")
            for layer in range(num_layers)
        ]
    )
    mean_distance, variance_distance, bootstrap_runs = bootstrap_layer_distances(
        train_matrices,
        repeats=int(config["bootstrap_repeats"]),
        seed=int(config["seed"]),
    )
    np.save(output_dir / "layer_distance_mean.npy", mean_distance.astype(np.float32))
    np.save(output_dir / "layer_distance_variance.npy", variance_distance.astype(np.float32))
    np.save(output_dir / "layer_distance_bootstrap_runs.npy", bootstrap_runs.astype(np.float32))

    train_hiddens = torch.load(calibration_dir / "calibration_train_entry_hiddens.pt", map_location="cpu").float().numpy()
    val_hiddens = torch.load(calibration_dir / "calibration_validation_entry_hiddens.pt", map_location="cpu").float().numpy()
    if train_hiddens.shape[1] != num_layers + 1 or val_hiddens.shape[1] != num_layers + 1:
        raise ValueError("Hidden-state artifact does not contain embedding entry plus every layer output")
    if train_hiddens.shape[0] != train_matrices.shape[1] or val_hiddens.shape[0] != cross_matrices.shape[2]:
        raise ValueError("Calibration hidden rows must correspond one-to-one with signature batches")

    candidate_costs = {}
    candidate_payload = {}
    candidate_rows = []
    min_length = int(config["block_min_length"])
    max_length = int(config["block_max_length"])
    for start in range(num_layers):
        for end in range(start + min_length - 1, min(num_layers, start + max_length)):
            block_train = train_matrices[start : end + 1].mean(axis=0)
            labels_train = spectral_labels(block_train, groups, int(config["seed"]) + 1009 * start + end)
            block_cross = cross_matrices[start : end + 1].mean(axis=0)
            labels_val = assign_validation(block_cross, labels_train, groups)
            learnability = probe_cross_entropy(
                train_hiddens[:, start, :],
                labels_train,
                val_hiddens[:, start, :],
                labels_val,
                int(config["seed"]) + start,
                config,
            )
            heterogeneity = block_heterogeneity(mean_distance, start, end)
            cost = heterogeneity + float(config["beta"]) * learnability
            candidate_costs[(start, end)] = cost
            key = block_key(start, end)
            candidate_payload[key] = {
                "train_labels": labels_train.tolist(),
                "validation_labels": labels_val.tolist(),
            }
            candidate_rows.append(
                {
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "heterogeneity": heterogeneity,
                    "probe_cross_entropy": learnability,
                    "cost_without_router_penalty": cost,
                }
            )
            print(f"candidate [{start},{end}] H={heterogeneity:.6f} E={learnability:.6f}", flush=True)

    blocks, objective = solve_contiguous_partition(
        candidate_costs,
        num_layers,
        min_length=min_length,
        max_length=max_length,
        min_blocks=int(config["min_blocks"]),
        max_blocks=int(config["max_blocks"]),
        router_penalty=float(config["lambda"]),
    )
    block_records = []
    calibration_groups = pd.DataFrame()
    validation_groups = pd.DataFrame()
    for block_id, (start, end) in enumerate(blocks):
        labels = candidate_payload[block_key(start, end)]
        calibration_groups[f"block_{block_id}_group"] = labels["train_labels"]
        validation_groups[f"block_{block_id}_group"] = labels["validation_labels"]
        block_records.append(
            {
                "block_id": block_id,
                "start": start,
                "end": end,
                "layers": list(range(start, end + 1)),
                "heterogeneity": block_heterogeneity(mean_distance, start, end),
                "probe_cross_entropy": next(
                    row["probe_cross_entropy"] for row in candidate_rows if row["start"] == start and row["end"] == end
                ),
            }
        )

    bootstrap_partitions = []
    for repeat, distance in enumerate(bootstrap_runs):
        repeat_costs = {
            key: block_heterogeneity(distance, *key)
            + float(config["beta"])
            * next(row["probe_cross_entropy"] for row in candidate_rows if (row["start"], row["end"]) == key)
            for key in candidate_costs
        }
        repeat_blocks, repeat_objective = solve_contiguous_partition(
            repeat_costs,
            num_layers,
            min_length=min_length,
            max_length=max_length,
            min_blocks=int(config["min_blocks"]),
            max_blocks=int(config["max_blocks"]),
            router_penalty=float(config["lambda"]),
        )
        bootstrap_partitions.append({"repeat": repeat, "blocks": repeat_blocks, "objective": repeat_objective})

    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_blocks.csv", index=False)
    calibration_groups.to_csv(output_dir / "selected_calibration_train_groups.csv", index=False)
    validation_groups.to_csv(output_dir / "selected_calibration_validation_groups.csv", index=False)
    write_json(
        output_dir / "partition.json",
        {
            "format": "adaptive_contiguous_partition_v2",
            "objective_formula": "sum_B[H(B)+beta*E_probe(B)]+lambda*G",
            "full_data_layer_policy": "all layers in each selected block, equally weighted",
            "objective": objective,
            "beta": float(config["beta"]),
            "lambda": float(config["lambda"]),
            "constraints": {
                "min_block_length": min_length,
                "max_block_length": max_length,
                "min_blocks": int(config["min_blocks"]),
                "max_blocks": int(config["max_blocks"]),
            },
            "blocks": block_records,
            "bootstrap_partitions": bootstrap_partitions,
        },
    )
    print(f"Selected blocks: {blocks}; objective={objective:.6f}")
    validate_partition_groups(config, {"blocks": block_records})


if __name__ == "__main__":
    main()
