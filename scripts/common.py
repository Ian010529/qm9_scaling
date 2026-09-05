"""Shared paths, ID alignment and provenance. Run scripts from any directory."""
import hashlib
import gzip
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["mu", "homo", "lumo", "G"]


def config():
    return json.loads((ROOT / "config/experiment.json").read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv_gz(frame, path):
    """Close and verify a complete gzip member before exposing its final path."""
    path = Path(path)
    payload = gzip.compress(frame.to_csv(index=False).encode("utf-8"), mtime=0)
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    if gzip.decompress(temp.read_bytes()).count(b"\n") != len(frame) + 1:
        raise ValueError("CSV write failed integrity check")
    temp.replace(path)


def versions():
    result = {}
    for name in ["numpy", "pandas", "rdkit", "xgboost", "torch", "transformers", "chemprop"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def dataset():
    return pd.read_csv(ROOT / "data/processed/qm9.csv.gz")


def load_split(seed, size):
    manifest = json.loads((ROOT / "splits/manifest.json").read_text())
    if manifest["dataset_sha256"] != sha256(ROOT / "data/processed/qm9.csv.gz"):
        raise ValueError("Data changed after splitting; refusing to use stale indices")
    seq = np.load(ROOT / f"splits/sequence_{seed}.npy")
    if size not in manifest["train_sizes"] or size > len(seq):
        raise ValueError("Unsupported training size")
    with np.load(ROOT / "splits/fixed.npz") as fixed:
        return {"train": seq[:size], "val": fixed["val"], "test": fixed["test"]}


def row_indices(frame, ids):
    pos = pd.Index(frame.mol_id).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError("Molecule IDs are missing from data")
    return pos
