"""Prepare explicit Chemprop splits for one run. Does not train a model."""
import argparse

import pandas as pd

from common import ROOT, TARGETS, dataset, load_split, row_indices, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--subset-seed", type=int, default=11)
    p.add_argument("--target", choices=TARGETS, default="homo")
    p.add_argument("--training-seed", type=int, default=101)
    p.add_argument("--include-test", action="store_true", help="Only enable for finalized benchmark evaluation")
    args = p.parse_args()
    frame = dataset()
    splits = load_split(args.subset_seed, args.size)
    selected = ["train", "val"] + (["test"] if args.include_test else [])
    out = ROOT / f"exports/chemprop_{args.target}_n{args.size}_s{args.subset_seed}"
    out.mkdir(parents=True, exist_ok=False)
    parts = []
    for name in selected:
        part = frame.iloc[row_indices(frame, splits[name])][["mol_id", "smiles", args.target]].copy()
        part["split"] = name
        parts.append(part)
    pd.concat(parts, ignore_index=True).to_csv(out / "data.csv", index=False)
    # One explicit splits column avoids Chemprop making its own random split.
    command = (f"chemprop train --data-path {out.relative_to(ROOT)}/data.csv "
               f"--smiles-columns smiles --target-columns {args.target} --splits-column split "
               f"--task-type regression --metrics mae rmse --epochs 100 --patience 20 --pytorch-seed {args.training_seed} "
               f"--output-dir runs/dmpnn_{args.target}_n{args.size}_s{args.subset_seed}")
    (out / "command.txt").write_text(command + "\n")
    write_json(out / "manifest.json", {"arguments": vars(args), "dataset_sha256": sha256(ROOT / "data/processed/qm9.csv.gz"),
               "status": "Input exported; training not executed. Fixed model seed and pilot early-stopping settings are included."})
    print(command)


if __name__ == "__main__":
    main()
