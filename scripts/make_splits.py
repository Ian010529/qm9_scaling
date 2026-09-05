"""Freeze the master split and store one full nested ordering per subset seed."""
import numpy as np
import pandas as pd

from common import ROOT, config, dataset, sha256, write_json, write_csv_gz


def make_arrays(ids, cfg):
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("IDs must be unique")
    order = np.random.default_rng(cfg["split_seed"]).permutation(ids)
    ntrain = int(len(ids) * cfg["train_fraction"])
    ntest = int(len(ids) * cfg["test_fraction"])
    fixed = {"test": order[:ntest], "val": order[ntest:len(ids)-ntrain], "pool": order[len(ids)-ntrain:]}
    if max(cfg["train_sizes"]) > ntrain:
        raise ValueError("Cleaned training pool is smaller than largest requested training size")
    seqs = {seed: np.random.default_rng(seed).permutation(fixed["pool"]) for seed in cfg["subset_seeds"]}
    return fixed, seqs


def main():
    out = ROOT / "splits"
    if (out / "manifest.json").exists():
        raise FileExistsError("Frozen splits already exist; refusing to overwrite")
    cfg, frame = config(), dataset()
    fixed, sequences = make_arrays(frame.mol_id.to_numpy(dtype=str), cfg)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fixed.npz", **fixed)
    mapping = []
    for split, ids in fixed.items():
        mapping.extend({"mol_id": mol_id, "split": split} for mol_id in ids)
    pd.DataFrame(mapping).to_csv(out / "fixed_assignment.csv", index=False)
    hashes = {"fixed.npz": sha256(out / "fixed.npz")}
    for seed, seq in sequences.items():
        path = out / f"sequence_{seed}.npy"
        np.save(path, seq, allow_pickle=False)
        hashes[path.name] = sha256(path)
        # Rank makes every nested subset visible without 40 redundant files.
        write_csv_gz(pd.DataFrame({"rank": np.arange(1, len(seq)+1), "mol_id": seq}),
                     out / f"sequence_{seed}.csv.gz")
    write_json(out / "manifest.json", {
        "dataset_sha256": sha256(ROOT / "data/processed/qm9.csv.gz"),
        "split_seed": cfg["split_seed"], "subset_seeds": cfg["subset_seeds"],
        "training_seeds": cfg["training_seeds"], "train_sizes": cfg["train_sizes"],
        "counts": {k: len(v) for k, v in fixed.items()}, "files_sha256": hashes,
        "convention": "IDs, not row offsets. First N IDs of each sequence are the N-sample training set."
    })
    print({k: len(v) for k, v in fixed.items()})


if __name__ == "__main__":
    main()
