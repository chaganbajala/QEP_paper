#!/bin/bash -l
#SBATCH -J afm_green25_plot
#SBATCH -D /zeropoint/u/qinwang/notebooks/QEP_fixed_shot_verify_20260914/QEP_multiple_shot
#SBATCH -p standard
#SBATCH -o ./AFM_searching/logs/green25_summary.%j.out
#SBATCH -e ./AFM_searching/logs/green25_summary.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00

set -eo pipefail
conda activate qep-cuda
set -u
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/qinwang-green25-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.green_start_budget finish
