"""Content-based provenance and strict atomic records."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON number {value}: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temp.open('w', encoding='utf-8') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def check_file(path, expected):
    with Path(path).open('rb') as f:
        if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Git LFS pointer instead of data: {path}; run git lfs pull')
    if sha256(path) != expected:
        raise ValueError(f'Content hash mismatch: {path}')


def code_manifest():
    files = [*ROOT.glob('scaling/*.py'), ROOT/'config/scaling_law_v1.json', ROOT/'requirements-scaling.txt', ROOT/'requirements-base.txt', ROOT/'requirements-deep.txt', ROOT/'docs/SCALING_LAW_PROTOCOL.md', ROOT/'docs/RUNBOOK.md']
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files) if p.is_file()}


def versions():
    result = {}
    for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'rdkit', 'xgboost', 'torch', 'chemprop']:
        try:
            result[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            result[n] = None
    return result


def partition_check(ids, parts):
    universe = set(map(str, ids))
    if len(universe) != len(ids):
        raise ValueError('Duplicate molecule IDs')
    seen = set()
    for name, arr in parts.items():
        values = set(map(str, arr))
        if len(values) != len(arr) or values & seen or not values <= universe:
            raise ValueError(f'Invalid/overlapping split: {name}')
        seen |= values
    if seen != universe:
        raise ValueError('Partition does not cover the dataset')
