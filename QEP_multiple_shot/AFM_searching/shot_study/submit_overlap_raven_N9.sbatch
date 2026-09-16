#!/bin/bash -l
#SBATCH -J afm_overlap_N9
#SBATCH -D /raven/u/qinwang/notebooks/QEP_overlap_20260914/QEP_multiple_shot
#SBATCH -p gpu
#SBATCH -o ./AFM_searching/logs/overlap_raven_N9.%A_%a.out
#SBATCH -e ./AFM_searching/logs/overlap_raven_N9.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --array=0-24%8

set -euo pipefail
conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
# N=9 analogue of submit_overlap_raven.sbatch (2026-09-14 N=5 campaign):
# same 25-group partition of the 10-start x 5-seed x 4-Ng ensemble, same
# --groups mechanism, filtered to --N 9 instead of --N 5. Concurrency is
# capped at 8 (vs the N5 campaign's 16) per the conservative-concurrency
# instruction for this task, given unrelated jobs active on both clusters
# and the N5 campaign's own QOSMaxJobsPerUserLimit experience at 16.
# Per-group new-epoch totals were computed offline from the real cases()/
# EPOCHS tables (see GPT_relay.md, N=9 section, 2026-09-15): 25 groups of
# 8 cases each carry 11,200-11,400 new epochs, i.e. ~126-128 minutes at the
# measured 0.66 s/epoch N=9 steady-state rate; --time=4:00:00 gives ~1.9x
# margin over that estimate.
srun python -u -m AFM_searching.shot_study.overlap_training "${SLURM_ARRAY_TASK_ID}" --groups 25 --N 9 "$@"
