#!/usr/bin/env python3
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.io_utils import dataframe_identity, read_json, sha256_file, write_json


IDENTITY_CANDIDATES = (
    "subject_id",
    "patient_id",
    "hadm_id",
    "admission_id",
    "signature_batch_id",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create full-new_train training and new_test validation manifests before signatures.")
    parser.add_argument("--config", type=Path, default=ROOT / "config_openjev.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_sample_id(split, source_row, text, label):
    payload = f"{split}\0{source_row}\0{int(label)}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def validate_source(df, path):
    missing = sorted({"text", "label"} - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{path} contains null text or label")
    df = df.copy().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["source_row"] = np.arange(len(df), dtype=np.int64)
    return df


def resolve_identity_column(config, train_df, test_df):
    requested = config.get("identity_column", "auto")
    if requested != "auto":
        if requested not in train_df or requested not in test_df:
            raise ValueError(f"identity_column={requested!r} must exist in both source CSVs")
        return requested
    for candidate in IDENTITY_CANDIDATES:
        if candidate in train_df and candidate in test_df:
            return candidate
    return None


def assert_source_isolation(train_df, test_df, identity_column):
    if identity_column is not None:
        overlap = set(train_df[identity_column].dropna()) & set(test_df[identity_column].dropna())
        if overlap:
            raise ValueError(f"Source train/test leak {len(overlap)} identities through {identity_column}")
    duplicate_text = set(train_df["text"].astype(str)) & set(test_df["text"].astype(str))
    if duplicate_text:
        raise ValueError(f"Source train/test contain {len(duplicate_text)} identical texts")


def split_train_validation(df, fraction, seed, identity_column):
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if identity_column is None or df[identity_column].nunique(dropna=False) == len(df):
        indices = np.arange(len(df))
        train_index, val_index = train_test_split(
            indices,
            test_size=fraction,
            random_state=seed,
            stratify=df["label"],
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        train_index, val_index = next(splitter.split(df, y=df["label"], groups=df[identity_column]))
    return df.iloc[train_index].copy(), df.iloc[val_index].copy()


def finalize(df, source_split):
    df = df.copy().sort_values("source_row").reset_index(drop=True)
    df.insert(
        0,
        "sample_id",
        [stable_sample_id(source_split, row.source_row, row.text, row.label) for row in df.itertuples()],
    )
    return df


def stratified_calibration(df, size, seed):
    if size <= 0 or size >= len(df):
        return df.copy().reset_index(drop=True)
    # A stratified holdout must contain at least one item per class. The DRG
    # data has 738 labels, so the requested smoke value 512 is raised to the
    # smallest feasible calibration size instead of failing in sklearn.
    target_size = min(max(int(size), int(df["label"].nunique())), len(df))
    rng = np.random.default_rng(seed)
    selected_indices = []
    for _, group in df.groupby("label", sort=True):
        selected_indices.append(int(rng.choice(group.index.to_numpy())))
    remaining = df.drop(index=selected_indices)
    remaining_needed = target_size - len(selected_indices)
    if remaining_needed > 0:
        extra_indices = rng.choice(remaining.index.to_numpy(), size=remaining_needed, replace=False)
        selected_indices.extend(int(index) for index in extra_indices)
    selected = df.loc[selected_indices]
    return selected.sort_values("sample_id").reset_index(drop=True)


def main():
    args = parse_args()
    config = read_json(args.config)
    run_dir = Path(config["run_dir"])
    split_dir = run_dir / "00_splits"
    if split_dir.exists() and any(split_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{split_dir} is not empty; pass --overwrite to replace split files")
    split_dir.mkdir(parents=True, exist_ok=True)

    source_train_path = Path(config["source_train_csv"])
    source_test_path = Path(config["source_test_csv"])
    source_train = validate_source(pd.read_csv(source_train_path), source_train_path)
    source_test = validate_source(pd.read_csv(source_test_path), source_test_path)
    identity_column = resolve_identity_column(config, source_train, source_test)
    assert_source_isolation(source_train, source_test, identity_column)

    protocol = config.get("split_protocol", "new_train_full_new_test_validation")
    if protocol != "new_train_full_new_test_validation":
        raise ValueError("This experiment uses full new_train and new_test as validation")
    train, validation = source_train.copy(), source_test.copy()
    train = finalize(train, "source_train")
    validation = finalize(validation, "source_test")

    if identity_column is not None:
        train_ids = set(train[identity_column])
        val_ids = set(validation[identity_column])
        if train_ids & val_ids:
            raise RuntimeError("Identity isolation failed after split")

    calibration_train = stratified_calibration(train, int(config["calibration_train_size"]), int(config["seed"]))
    calibration_val = stratified_calibration(validation, int(config["calibration_val_size"]), int(config["seed"]) + 1)
    outputs = {
        "train": train,
        "validation": validation,
        "calibration_train": calibration_train,
        "calibration_validation": calibration_val,
    }
    paths = {}
    identities = {}
    for name, frame in outputs.items():
        path = split_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
        identities[name] = dataframe_identity(frame)

    manifest = {
        "format": "adaptive_blockwise_split_manifest_v1",
        "seed": int(config["seed"]),
        "split_protocol": protocol,
        "independent_test_available": False,
        "identity_column": identity_column,
        "isolation_basis": identity_column or "one-row-per-admission source contract plus duplicate-text audit",
        "source_train_csv": str(source_train_path),
        "source_train_sha256": sha256_file(source_train_path),
        "source_test_csv": str(source_test_path),
        "source_test_sha256": sha256_file(source_test_path),
        "paths": paths,
        "dataframe_sha256": identities,
        "counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "label_counts": {
            name: {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
            for name, frame in outputs.items()
        },
    }
    write_json(split_dir / "manifest.json", manifest)
    from openjev_drg.protocol import make_members
    for name, size in (("train", 16), ("validation", 8)):
        write_json(split_dir / f"{name}_batches.json", make_members(outputs[name], size))
    print(f"Wrote immutable split manifest to {split_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
