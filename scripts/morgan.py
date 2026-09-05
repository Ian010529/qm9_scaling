"""Calculate immutable Morgan bit features, stored packed (8 bits/byte)."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from common import ROOT, config, dataset, sha256, versions, write_json


def main():
    out = ROOT / "features"
    out.mkdir(exist_ok=True)
    if (out / "morgan.npz").exists():
        raise FileExistsError("Features already exist")
    cfg, frame = config()["morgan"], dataset()
    generator = rdFingerprintGenerator.GetMorganGenerator(**cfg)
    packed = np.empty((len(frame), (cfg["fpSize"] + 7)//8), dtype=np.uint8)
    for i, smiles in enumerate(frame.smiles):
        fp = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
        packed[i] = np.packbits(fp, bitorder="little")
        if (i+1) % 20000 == 0:
            print(f"Fingerprinted {i+1:,}", flush=True)
    np.savez_compressed(out / "morgan.npz", packed=packed, mol_id=frame.mol_id.to_numpy(dtype=str))
    n_unique = len(np.unique(packed, axis=0))
    write_json(out / "morgan.json", {"parameters": cfg, "bitorder": "little", "rows": len(frame),
               "unique_fingerprints": n_unique, "duplicate_fingerprint_rows": len(frame)-n_unique,
               "dataset_sha256": sha256(ROOT / "data/processed/qm9.csv.gz"),
               "features_sha256": sha256(out / "morgan.npz"), "versions": versions()})
    print(f"Saved {len(frame):,} fingerprints; {n_unique:,} distinct bit vectors")


if __name__ == "__main__":
    main()
