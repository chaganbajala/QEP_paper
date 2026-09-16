#!/usr/bin/env python
"""Aggregate the focused extra-seed check of the N=9 budget-curve dip."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, ttest_ind

from AFM_searching.shot_study import config as C
from AFM_searching.shot_study.verify_budget_noise import (
    BUDGET, EXTRA_SEEDS, N, NG_VALUES, OUTPUT_DIR,
)


def original_values(ng):
    values = []
    for seed in range(C.BUDGET_SEEDS):
        path = C.DATA_DIR / "budgets" / f"N{N}_Ng{ng}_seed{seed}.npz"
        data = np.load(path, allow_pickle=False)
        i = int(np.flatnonzero(data["budgets"] == BUDGET)[0])
        values.append(float(data["checkpoint_exact_AM2"][i]))
    return np.asarray(values)


def extra_values(ng):
    values = []
    for seed in EXTRA_SEEDS:
        path = OUTPUT_DIR / f"N{N}_budget{BUDGET}_Ng{ng}_seed{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        values.append(float(np.load(path, allow_pickle=False)["exact_AM2"]))
    return np.asarray(values)


def describe(values):
    n = len(values)
    std = float(values.std(ddof=1))
    sem = std / np.sqrt(n)
    half_width = float(t.ppf(0.975, n - 1) * sem)
    return {
        "n": n,
        "mean": float(values.mean()),
        "std": std,
        "sem": float(sem),
        "ci95": [float(values.mean() - half_width),
                 float(values.mean() + half_width)],
    }


def main():
    original = {ng: original_values(ng) for ng in NG_VALUES}
    extra = {ng: extra_values(ng) for ng in NG_VALUES}
    combined = {ng: np.concatenate((original[ng], extra[ng])) for ng in NG_VALUES}

    summary = {
        "N": N,
        "budget": BUDGET,
        "original_5_seeds": {str(ng): describe(original[ng]) for ng in NG_VALUES},
        "new_20_seeds": {str(ng): describe(extra[ng]) for ng in NG_VALUES},
        "combined_25_seeds": {str(ng): describe(combined[ng]) for ng in NG_VALUES},
        "welch_tests_combined": {},
    }
    for neighbor in (20, 100, 200):
        test = ttest_ind(combined[50], combined[neighbor], equal_var=False)
        summary["welch_tests_combined"][f"Ng50_vs_Ng{neighbor}"] = {
            "mean_difference": float(combined[50].mean() - combined[neighbor].mean()),
            "t_statistic": float(test.statistic),
            "two_sided_p": float(test.pvalue),
        }

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.asarray(NG_VALUES)
    old_mean = np.asarray([original[ng].mean() for ng in NG_VALUES])
    mean = np.asarray([combined[ng].mean() for ng in NG_VALUES])
    low = np.asarray([summary["combined_25_seeds"][str(ng)]["ci95"][0]
                      for ng in NG_VALUES])
    high = np.asarray([summary["combined_25_seeds"][str(ng)]["ci95"][1]
                       for ng in NG_VALUES])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
    ax.plot(x, old_mean, "o--", color="0.55", label="original 5-seed mean")
    ax.errorbar(x, mean, yerr=np.vstack((mean - low, high - mean)), fmt="o-",
                capsize=3, label="combined 25-seed mean and 95% CI")
    ax.set_xscale("log")
    ax.set_xlabel(r"shots per phase and epoch $N_g$")
    ax.set_ylabel(r"exact $\langle AFM^2\rangle$")
    ax.set_title(rf"Noise check: $N={N}$, budget $={BUDGET:,}$")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 180} if suffix == "png" else {}
        fig.savefig(C.FIG_DIR / f"time_budget_N9_noise_check.{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)

    out = C.DATA_DIR / "budget_noise_check_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
