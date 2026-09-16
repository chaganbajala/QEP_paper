"""Repeat the four green-start figures for purple trajectory init5 (25 seeds).

250 fresh training cases; same Ng grid, budgets, Ny, and numerical method.
The purple coordinates are verified against the saved init5 trajectories.
"""
import argparse

from . import config as C
from .figure11b_resampling import run_case
from .green_start_budget import summarize

POINT = (1.16771310144114, 1.0853474197535722)
INIT_ID = 5
SEED_OFFSET = 5000
OUT = C.DATA_DIR/'budget_purple_init5_N5_2026-09-15'
SUMMARY = C.DATA_DIR/'time_budget_purple_init5_N5_25seeds.npz'
EXPECTATIONS = C.DATA_DIR/'purple_trajectory_expectations_N5_25seeds.npz'


def finish():
    from .green_trajectory_expectations import process
    summarize(point=POINT, out=OUT, summary=SUMMARY, init_id=INIT_ID,
              seed_offset=SEED_OFFSET)
    process(point=POINT, out=OUT, summary=SUMMARY, seed_offset=SEED_OFFSET,
            output=EXPECTATIONS)
    from .purple_plot_figures import main
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'finish'))
    parser.add_argument('--case-id', type=int, choices=range(250))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':
        if args.case_id is None: parser.error('run requires --case-id')
        run_case(args.case_id, args.smoke, point=POINT, output_dir=OUT,
                 seeds=tuple(range(25)), seed_offset=SEED_OFFSET,
                 task='purple_init5_budget')
    else:
        finish()
