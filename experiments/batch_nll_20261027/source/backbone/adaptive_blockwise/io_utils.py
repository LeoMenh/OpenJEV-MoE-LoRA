import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SERVER_ROOT = Path("/home/devbox/drg-llama/new_layerwise")
SERVER_DATA = Path("/home/devbox/drg-llama/data")
DEFAULT_MODEL = Path("/home/devbox/drg-llama/models/Qwen3-14B")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path, chunk_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_identity(df):
    required = ["sample_id", "text", "label"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Cannot identify dataframe; missing columns: {missing}")
    digest = hashlib.sha256()
    for row in df[required].itertuples(index=False, name=None):
        digest.update(str(row[0]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row[1]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row[2])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_split_csv(path, require_groups=False, num_blocks=None):
    path = Path(path)
    df = pd.read_csv(path).reset_index(drop=True)
    required = {"sample_id", "text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values")
    df["label"] = df["label"].astype(int)
    if require_groups:
        if num_blocks is None:
            raise ValueError("num_blocks is required when require_groups=True")
        for block in range(num_blocks):
            for suffix in ("group", "confidence"):
                name = f"block_{block}_{suffix}"
                if name not in df.columns:
                    raise ValueError(f"{path} is missing {name}")
    return df


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_unlink(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
