#!/bin/bash -l
#SBATCH -J afm_overlap
#SBATCH -D /zeropoint/u/qinwang/notebooks/QEP_fixed_shot_verify_20260914/QEP_multiple_shot
#SBATCH -p gpu
#SBATCH -o ./AFM_searching/logs/overlap_zp.%A_%a.out
#SBATCH -e ./AFM_searching/logs/overlap_zp.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
#SBATCH --time=23:50:00
#SBATCH --exclude=zpg002
#SBATCH --array=0-24%8

set -euo pipefail
conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.overlap_training "${SLURM_ARRAY_TASK_ID}" --groups 25 --N 5 "$@"
