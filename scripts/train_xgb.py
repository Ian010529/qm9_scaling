"""One single-target run. Pilot reports validation only unless --evaluate-test."""
import argparse
import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common import ROOT, TARGETS, config, dataset, load_split, row_indices, sha256, versions, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=TARGETS, default="homo")
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--subset-seed", type=int, default=11)
    p.add_argument("--training-seed", type=int, default=101)
    p.add_argument("--evaluate-test", action="store_true")
    args = p.parse_args()
    cfg, frame = config(), dataset()
    splits = load_split(args.subset_seed, args.size)
    meta = json.loads((ROOT / "features/morgan.json").read_text())
    if meta["dataset_sha256"] != sha256(ROOT / "data/processed/qm9.csv.gz"):
        raise ValueError("Features refer to a different dataset")
    with np.load(ROOT / "features/morgan.npz") as f:
        if not np.array_equal(f["mol_id"], frame.mol_id.to_numpy(dtype=str)):
            raise ValueError("Feature/label ID ordering mismatch")
        packed = f["packed"]
    out = ROOT / f"runs/xgb_{args.target}_n{args.size}_s{args.subset_seed}_t{args.training_seed}"
    out.mkdir(parents=True, exist_ok=False)
    pos = {s: row_indices(frame, ids) for s, ids in splits.items()}
    y = frame[args.target].to_numpy(dtype=np.float64)
    # Fit normalization on this training subset only. Useful for tiny energy labels.
    mean, scale = float(y[pos["train"]].mean()), float(y[pos["train"]].std())
    scale = scale if scale > 0 else 1.0
    def features(split):
        return np.unpackbits(packed[pos[split]], axis=1, count=cfg["morgan"]["fpSize"], bitorder="little")
    model = XGBRegressor(**cfg["xgboost_pilot"], random_state=args.training_seed, objective="reg:squarederror")
    started = time.perf_counter()
    model.fit(features("train"), (y[pos["train"]]-mean)/scale,
              eval_set=[(features("val"), (y[pos["val"]]-mean)/scale)], verbose=False)
    fit_seconds = time.perf_counter()-started
    report = {"arguments": vars(args), "parameters": model.get_params(), "versions": versions(),
              "dataset_sha256": meta["dataset_sha256"], "features_sha256": meta["features_sha256"],
              "split_manifest_sha256": sha256(ROOT / "splits/manifest.json"),
              "unit": cfg["units"][args.target], "target_normalization": {"mean": mean, "std": scale},
              "fit_seconds": fit_seconds, "best_iteration": model.best_iteration,
              "role": "pilot; validation-selected early stopping; hyperparameters not tuned", "metrics": {}}
    for split in ["val"] + (["test"] if args.evaluate_test else []):
        pred = model.predict(features(split)).astype(float)*scale + mean
        true = y[pos[split]]
        err = pred-true
        report["metrics"][split] = {"n": len(true), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt(np.mean(err**2))),
                                      "training_mean_baseline_MAE": float(np.abs(true-mean).mean())}
        pd.DataFrame({"mol_id": splits[split], "target": args.target, "unit": report["unit"],
                      "y_true": true, "y_pred": pred}).to_csv(out / f"{split}_predictions.csv", index=False)
    model.save_model(out / "model.ubj")
    write_json(out / "learning_history.json", model.evals_result())
    write_json(out / "metrics.json", report)
    print(json.dumps({"fit_seconds": fit_seconds, "best_iteration": model.best_iteration, "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
