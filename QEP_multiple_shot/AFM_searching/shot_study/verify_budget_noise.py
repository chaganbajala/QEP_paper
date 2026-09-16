#!/usr/bin/env python
"""Focused extra-seed check of the N=9, budget=50,000, N_g=50 dip."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from AFM_searching.afm_shots import common
from AFM_searching.shot_study import config as C
from AFM_searching.shot_study.run import atomic_save, metadata, train_case


N = 9
BUDGET = 50_000
NG_VALUES = (20, 50, 100, 200)
EXTRA_SEEDS = tuple(range(5, 25))
OUTPUT_DIR = C.DATA_DIR / "budget_noise_check"


def cases():
    return [(ng, seed) for ng in NG_VALUES for seed in EXTRA_SEEDS]


def run_case(task_id: int, smoke: bool = False):
    ng, seed = cases()[task_id]
    epochs = 2 if smoke else BUDGET // (2 * ng)
    nn, _, result = train_case(
        N, C.TYPICAL_POINT, ng, epochs, seed=seed,
        verbose_every=max(1, epochs // 10),
    )
    final_params = nn.set_params(float(result["Omega"][-1]),
                                 float(result["Delta"][-1]))
    exact_am2 = common.exact_am2(nn, final_params)
    suffix = "_smoke" if smoke else ""
    path = OUTPUT_DIR / f"N{N}_budget{BUDGET}_Ng{ng}_seed{seed}{suffix}.npz"
    atomic_save(
        path,
        metadata=metadata(
            task="budget_noise_check", N=N, point=C.TYPICAL_POINT,
            N_g=ng, seed_id=seed, epochs=epochs, budget=BUDGET, smoke=smoke,
        ),
        budget=np.asarray(BUDGET),
        exact_AM2=np.asarray(exact_am2),
        **result,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.task_id < len(cases()):
        raise SystemExit(f"task_id must be in [0, {len(cases()) - 1}]")
    run_case(args.task_id, smoke=args.smoke)


if __name__ == "__main__":
    main()
