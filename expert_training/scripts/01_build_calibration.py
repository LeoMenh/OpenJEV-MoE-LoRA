#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.io_utils import load_split_csv, parse_str_list, read_json, seed_everything, write_json
from adaptive_blockwise.signals import (
    ExactSignalComputer,
    compute_directional_matrix,
    extract_signal_cache,
    extract_signal_cache_distributed,
    initialize_signal_distributed,
    load_tokenizer,
    make_dataset,
    symmetrize_and_normalize_memmap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build all-layer calibration compatibility matrices and entry hiddens.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--keep_signal_cache", action="store_true")
    parser.add_argument("--compat_device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def num_labels(config):
    import pandas as pd

    return int(pd.read_csv(config["id2label_csv"])["label"].nunique())


def extract_hiddens(computer, dataset, tokenizer, output_path, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
    rows = []
    for step, batch in enumerate(loader, start=1):
        pooled = computer.pooled_hidden_on_batch(batch)
        # One probe input per signature: equal sample mean of token-pooled entries.
        rows.append(pooled.float().mean(dim=0).to(torch.float16))
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f"hidden extraction: {step}/{len(loader)}", flush=True)
    hidden = torch.stack(rows)
    torch.save(hidden, output_path)
    return list(hidden.shape)


def extract_hiddens_distributed(computer, dataset, tokenizer, output_path, batch_size):
    rank, world = dist.get_rank(), dist.get_world_size()
    collator = DataCollatorWithPadding(tokenizer)
    rows = []
    device = next(computer.model.parameters()).device
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        count = end - start
        global_batch = collator([dataset[index] for index in range(start, end)])
        lo, hi = count * rank // world, count * (rank + 1) // world
        if hi > lo:
            batch = {key: value[lo:hi] for key, value in global_batch.items()}
            pooled = computer.pooled_hidden_on_batch(batch).float().sum(dim=0).to(device)
        else:
            config = computer.model.config
            pooled = torch.zeros((config.num_hidden_layers + 1, config.hidden_size), device=device)
        dist.all_reduce(pooled, op=dist.ReduceOp.SUM)
        if rank == 0:
            rows.append((pooled / count).cpu().to(torch.float16))
    if rank == 0:
        hidden = torch.stack(rows)
        torch.save(hidden, output_path)
        return list(hidden.shape)
    return None


def normalize_cross(matrix, mean, std):
    return ((np.asarray(matrix, dtype=np.float32) - float(mean)) / (float(std) + 1e-8)).astype(np.float32)


def main():
    args = parse_args()
    config = read_json(args.config)
    rank, world, local_rank = initialize_signal_distributed()
    train_batch_size = int(config.get("calibration_batch_size", 1))
    validation_batch_size = int(config.get("calibration_validation_batch_size", train_batch_size))
    if train_batch_size < 1 or validation_batch_size < 1:
        raise ValueError("calibration batch sizes must be positive")
    seed_everything(int(config["seed"]))
    run_dir = Path(config["run_dir"])
    split_manifest = read_json(run_dir / "00_splits" / "manifest.json")
    output_dir = run_dir / "01_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_train = output_dir / "_signals_train"
    cache_val = output_dir / "_signals_validation"

    train_df = load_split_csv(split_manifest["paths"]["calibration_train"])
    val_df = load_split_csv(split_manifest["paths"]["calibration_validation"])
    tokenizer = load_tokenizer(config["model_name_or_path"])
    train_ds = make_dataset(train_df, tokenizer, int(config["max_length"]))
    val_ds = make_dataset(val_df, tokenizer, int(config["max_length"]))
    layers = list(range(int(config["num_layers"])))
    computer = ExactSignalComputer(
        config["model_name_or_path"],
        num_labels(config),
        tokenizer,
        output_dir / "probe_coordinate.pt",
        int(config["seed"]),
        rank=int(config["probe_lora_rank"]),
        alpha=int(config["probe_lora_alpha"]),
        ridge=float(config["ridge"]),
        target_modules=parse_str_list(config["target_modules"]),
        no_4bit=bool(config.get("no_4bit", False)),
        device_map={"": local_rank} if world > 1 else "auto",
    )
    if sorted(computer.layer_modules) != layers:
        raise ValueError(
            f"Theory requires exactly layers {layers}; backbone exposes {sorted(computer.layer_modules)}"
        )

    extractor = extract_signal_cache_distributed if world > 1 else extract_signal_cache
    extractor(
        computer,
        train_ds,
        tokenizer,
        layers,
        cache_train,
        batch_size=train_batch_size,
        cache_batches=int(config["signature_cache_batches"]),
    )
    extractor(
        computer,
        val_ds,
        tokenizer,
        layers,
        cache_val,
        batch_size=validation_batch_size,
        cache_batches=int(config["signature_cache_batches"]),
    )
    hidden_extractor = extract_hiddens_distributed if world > 1 else extract_hiddens
    train_hidden_shape = hidden_extractor(computer, train_ds, tokenizer, output_dir / "calibration_train_entry_hiddens.pt", train_batch_size)
    val_hidden_shape = hidden_extractor(computer, val_ds, tokenizer, output_dir / "calibration_validation_entry_hiddens.pt", validation_batch_size)
    computer.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return

    normalization = {}
    for layer in layers:
        layer_dir = output_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        raw_train = layer_dir / "train_directional.npy"
        raw_train_val = layer_dir / "train_to_val_directional.npy"
        raw_val_train = layer_dir / "val_to_train_directional.npy"
        compute_directional_matrix(
            cache_train,
            cache_train,
            layer,
            raw_train,
            args.compat_device,
            score_block_batches=int(config["compatibility_score_block_batches"]),
            preload=bool(config["compatibility_preload"]),
        )
        _, mean, std = symmetrize_and_normalize_memmap(
            raw_train,
            layer_dir / "train_normalized_compatibility.npy",
        )
        train_to_val = compute_directional_matrix(
            cache_train,
            cache_val,
            layer,
            raw_train_val,
            args.compat_device,
            score_block_batches=int(config["compatibility_score_block_batches"]),
            preload=bool(config["compatibility_preload"]),
        )
        val_to_train = compute_directional_matrix(
            cache_val,
            cache_train,
            layer,
            raw_val_train,
            args.compat_device,
            score_block_batches=int(config["compatibility_score_block_batches"]),
            preload=bool(config["compatibility_preload"]),
        )
        cross = 0.5 * (np.asarray(train_to_val) + np.asarray(val_to_train).T)
        np.save(layer_dir / "train_validation_normalized_compatibility.npy", normalize_cross(cross, mean, std))
        normalization[str(layer)] = {"off_diagonal_mean": mean, "off_diagonal_std": std}
        raw_train.unlink()
        raw_train_val.unlink()
        raw_val_train.unlink()
        print(f"completed calibration layer {layer}", flush=True)

    if not args.keep_signal_cache:
        shutil.rmtree(cache_train)
        shutil.rmtree(cache_val)
    manifest = {
        "format": "adaptive_blockwise_calibration_v1",
        "formula": "C_ij_l=0.5*(S_i_to_j_l+S_j_to_i_l), per-layer off-diagonal z-normalization",
        "num_layers": len(layers),
        "num_calibration_train": len(train_df),
        "num_calibration_validation": len(val_df),
        "calibration_batch_size": train_batch_size,
        "calibration_validation_batch_size": validation_batch_size,
        "signature_world_size": world,
        "signature_local_batch_size": train_batch_size // world,
        "num_train_signatures": train_hidden_shape[0],
        "num_validation_signatures": val_hidden_shape[0],
        "hidden_pooling": "masked token mean per sample, then equal sample mean per signature batch",
        "train_identity_sha256": split_manifest["dataframe_sha256"]["calibration_train"],
        "validation_identity_sha256": split_manifest["dataframe_sha256"]["calibration_validation"],
        "normalization": normalization,
        "train_hidden_shape": train_hidden_shape,
        "validation_hidden_shape": val_hidden_shape,
        "signal_cache_retained": bool(args.keep_signal_cache),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote calibration artifacts to {output_dir}")


if __name__ == "__main__":
    main()
