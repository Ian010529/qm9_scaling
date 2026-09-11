#!/usr/bin/env python3
"""Render research-report figures from existing v1.3 results; no fitting or training."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scaling.laws import predict_v13

STUDY = ROOT / "studies/scaling-law-v1.3-exec-v4"
OUT = ROOT / "reports/assets/v1_3_exec_v4"
ROUTES = {"morgan_xgb": ("Morgan-XGBoost", "#d97706"),
          "dmpnn": ("D-MPNN", "#2563eb"),
          "molformer_mlp": ("Frozen MoLFormer + MLP", "#059669")}
TASKS = ["mu", "homo", "lumo", "G"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [STUDY / "analysis" / n for n in
               ["run_registry.csv", "scaling_fit.json", "scaling_uncertainty.json", "data_gains.csv"]]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    frame = pd.read_csv(sources[0])
    fits = json.loads(sources[1].read_text())
    gains = pd.read_csv(sources[3])
    assert len(frame) == 650 and frame.id.nunique() == 650
    assert set(frame[frame.route != "g_composition"].repeat.unique()) == {1, 2, 3, 4, 5}
    assert np.isfinite(frame[["MAE", "RMSE"]]).all().all()
    chosen = {(r["route"], r["task"]): r["fits"][r["selected_form"]]
              for r in fits if r["metric"] == "MAE" and r["range_min"] == 100}
    assert len(chosen) == 12
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    residual_fig, residual_axes = plt.subplots(2, 2, figsize=(13, 8))
    gain_fig, gain_axes = plt.subplots(2, 2, figsize=(13, 8))
    for task, ax, rx, gx in zip(TASKS, axes.flat, residual_axes.flat, gain_axes.flat):
        for route, (label, color) in ROUTES.items():
            pivot = frame[(frame.route == route) & (frame.task == task)].pivot(
                index="repeat", columns="n", values="MAE").sort_index(axis=1)
            assert pivot.shape == (5, 10) and not pivot.isna().any().any()
            n = pivot.columns.to_numpy(dtype=float)
            mean = pivot.mean().to_numpy()
            fit = chosen[route, task]
            for row in pivot.to_numpy():
                ax.plot(n, row, color=color, alpha=.16, linewidth=.8)
            ax.plot(n, mean, "o-", color=color, label=label, markersize=4)
            grid = np.geomspace(n.min(), n.max(), 250)
            ax.plot(grid, predict_v13(grid, fit), "--", color=color, linewidth=1.6)
            residuals = np.log(predict_v13(n, fit)) - np.log(mean)
            assert np.allclose(residuals, fit["residuals_log"], rtol=1e-8, atol=1e-10)
            rx.plot(n, residuals, "o-", color=color, label=label, markersize=4)
            rows = gains[(gains.route == route) & (gains.task == task) & (gains.metric == "MAE")]
            ag = rows.groupby(["n1", "n2"]).local_alpha.agg(["mean", "std"])
            assert len(ag) == 9
            mids = np.sqrt(np.asarray([a*b for a,b in ag.index], dtype=float))
            gx.errorbar(mids, ag["mean"], yerr=ag["std"], fmt="o-", color=color,
                        label=label, markersize=3, linewidth=1, capsize=2)
        if task == "G":
            base = frame[frame.route == "g_composition"].pivot(index="repeat", columns="n", values="MAE").sort_index(axis=1)
            for row in base.to_numpy():
                ax.plot(base.columns, row, color="#666666", alpha=.18, linewidth=.8)
            ax.plot(base.columns, base.mean(), ":s", color="#555555", markersize=3, label="Composition baseline")
        ax.set(xscale="log", yscale="log", xlabel="Training molecules N",
               ylabel="MAE (Debye)" if task == "mu" else "MAE (Hartree)", title=task)
        ax.legend(fontsize=8)
        for axis in [rx, gx]:
            axis.axhline(0, color="#666666", linewidth=.7)
            axis.set_xscale("log")
            axis.set_title(task)
            axis.legend(fontsize=8)
        rx.set(xlabel="Training molecules N", ylabel="log(fitted MAE / observed mean MAE)")
        gx.set(xlabel="Geometric midpoint of adjacent N values", ylabel="Local exponent (mean +/- sample SD)")
    fig.suptitle("QM9 v1.3: five repetitions, observed means, and selected MAE fits\nThin: repetitions; solid: mean; dashed: fit over N=100-100000", fontsize=12)
    residual_fig.suptitle("Selected full-range fit residuals (not held-out prediction errors)", fontsize=12)
    gain_fig.suptitle("Paired adjacent-size local exponents; error bars show repetition SD, not confidence intervals", fontsize=12)
    for figure, name in [(fig, "scaling_curves.png"), (residual_fig, "fit_residuals.png"), (gain_fig, "local_exponents.png")]:
        figure.tight_layout(rect=(0, 0, 1, .95))
        figure.savefig(OUT / name, dpi=170)
        plt.close(figure)
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in sources}
    record = {"source_files_sha256": before, "script_sha256": sha(Path(__file__)),
              "training_performed": False, "refitting_performed": False,
              "figures_sha256": {p.name: sha(p) for p in sorted(OUT.glob("*.png"))}}
    (OUT / "figure_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Rendered 3 figures; source hashes unchanged; no training or refitting.")


if __name__ == "__main__":
    main()
