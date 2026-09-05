"""Independent integrity checks of real data, frozen splits and Morgan alignment."""
import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from common import ROOT, TARGETS, config, dataset, sha256, write_json
from download import exclusion_ids


def main():
    cfg, frame = config(), dataset()
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    check("unique_molecule_ids", frame.mol_id.is_unique)
    check("unique_nonisomeric_smiles", frame.smiles.is_unique)
    check("finite_four_targets", np.isfinite(frame[TARGETS].to_numpy()).all())
    excluded = exclusion_ids(ROOT / "data/raw/uncharacterized.txt")
    check("excluded_ids_absent", not set(frame.qm9_index) & excluded)
    reference = pd.read_csv(ROOT / "data/raw/qm9_smiles.csv").set_index("mol_id", verify_integrity=True)
    check("reference_id_coverage", set(frame.mol_id) <= set(reference.index))
    matched = reference.loc[frame.mol_id]
    for target, source in [("mu", "mu"), ("homo", "homo"), ("lumo", "lumo"), ("G", "g298")]:
        check(f"reference_labels_{target}", np.array_equal(frame[target].to_numpy(), matched[source].to_numpy()))
    expected_smiles = [Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(s)), canonical=True, isomericSmiles=False)
                       for s in matched.smiles]
    check("all_structures_match_reference", np.array_equal(frame.smiles.to_numpy(), expected_smiles))
    methane = frame.loc[frame.mol_id == "gdb_1"].iloc[0]
    check("methane_source_label_and_units", methane.smiles == "C" and np.allclose(
        methane[TARGETS].to_numpy(dtype=float), [0., -0.3877, 0.1171, -40.498597], rtol=0, atol=1e-9))
    manifest = json.loads((ROOT / "splits/manifest.json").read_text())
    check("dataset_hash_matches", manifest["dataset_sha256"] == sha256(ROOT / "data/processed/qm9.csv.gz"))
    for file, digest in manifest["files_sha256"].items():
        check(f"hash_{file}", sha256(ROOT / "splits" / file) == digest)
    with np.load(ROOT / "splits/fixed.npz") as f:
        fixed = {k: f[k] for k in f.files}
    pool, val, test = (set(fixed[k]) for k in ["pool", "val", "test"])
    check("pairwise_disjoint", not (pool & val or pool & test or val & test))
    check("complete_partition", pool | val | test == set(frame.mol_id))
    for seed in cfg["subset_seeds"]:
        seq = np.load(ROOT / f"splits/sequence_{seed}.npy")
        check(f"sequence_{seed}_unique_complete", len(seq) == len(pool) and set(seq) == pool)
        check(f"sequence_{seed}_reproducible", np.array_equal(seq, np.random.default_rng(seed).permutation(fixed["pool"])))
        prev = set()
        for size in cfg["train_sizes"]:
            current = set(seq[:size])
            check(f"sequence_{seed}_n{size}", len(current) == size and prev <= current and not current & (val | test))
            prev = current
    if (ROOT / "features/morgan.npz").exists():
        meta = json.loads((ROOT / "features/morgan.json").read_text())
        check("feature_hash", meta["features_sha256"] == sha256(ROOT / "features/morgan.npz"))
        check("feature_data_hash", meta["dataset_sha256"] == manifest["dataset_sha256"])
        generator = rdFingerprintGenerator.GetMorganGenerator(**cfg["morgan"])
        with np.load(ROOT / "features/morgan.npz") as f:
            check("feature_ids_aligned", np.array_equal(f["mol_id"], frame.mol_id.to_numpy(dtype=str)))
            packed = f["packed"]
            # Full recomputation: checks every retained feature against reference SMILES.
            mismatches = 0
            for i, s in enumerate(expected_smiles):
                actual = np.unpackbits(packed[i], bitorder="little", count=cfg["morgan"]["fpSize"])
                expected = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
                mismatches += not np.array_equal(actual, expected)
            check("all_fingerprints_match_reference", mismatches == 0)
    report = {"all_passed": all(checks.values()), "checks": checks, "n_checks": len(checks),
              "dataset_rows": len(frame), "split_counts": manifest["counts"]}
    write_json(ROOT / "reports/verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
