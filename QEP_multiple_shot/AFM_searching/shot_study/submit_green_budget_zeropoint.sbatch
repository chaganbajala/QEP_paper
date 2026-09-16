#!/bin/bash -l
#SBATCH -J afm_green25
#SBATCH -D /zeropoint/u/qinwang/notebooks/QEP_fixed_shot_verify_20260914/QEP_multiple_shot
#SBATCH -p highfreq
#SBATCH -o ./AFM_searching/logs/green25.%A_%a.out
#SBATCH -e ./AFM_searching/logs/green25.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=01:15:00
#SBATCH --array=0-249%40

set -eo pipefail
conda activate qep-cuda
set -u
export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/qinwang-green25-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.green_start_budget run --case-id "${SLURM_ARRAY_TASK_ID}"
