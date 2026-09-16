#!/usr/bin/env python
"""Map compact Slurm-array IDs onto production experiment cases."""
import argparse

from . import config as C
from . import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("gradient", "trajectory", "budget", "stats"))
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()
    i = args.task_id

    if args.family == "gradient":
        cases = [(N, scan) for N in C.N_LIST for scan in ("ng", "beta")]
        N, scan = cases[i]
        run.run_gradient(N, scan, C.GRADIENT_REPEATS)
    elif args.family == "trajectory":
        # One array task handles all five seeds for a fixed (N, init, N_g),
        # amortising import/JIT overhead without making the long budget jobs larger.
        cases = [(N, init_id, N_g) for N in C.N_LIST
                 for init_id in range(len(C.INITIAL_POINTS)) for N_g in C.TRAJECTORY_NG]
        N, init_id, N_g = cases[i]
        all_cases = run.trajectory_cases()
        for case_id, case in enumerate(all_cases):
            if case[0] == N and case[1] == init_id and case[3] == N_g:
                run.run_trajectory(case_id)
    elif args.family == "budget":
        run.run_budget(i)
    else:
        run.run_stats(C.N_LIST[i])


if __name__ == "__main__":
    main()
