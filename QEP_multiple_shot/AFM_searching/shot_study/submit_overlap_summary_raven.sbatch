#!/bin/bash -l
#SBATCH -J afm_overlap_sum
#SBATCH -D /raven/u/qinwang/notebooks/QEP_overlap_20260914/QEP_multiple_shot
#SBATCH -p general
#SBATCH -o ./AFM_searching/logs/overlap_summary.%j.out
#SBATCH -e ./AFM_searching/logs/overlap_summary.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00

set -euo pipefail
conda activate qep-cuda
export JAX_PLATFORM_NAME=cpu
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.summarize_overlap --N 5
