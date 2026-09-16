#!/bin/bash -l
#SBATCH -J afm_overlap_view
#SBATCH -D /raven/u/qinwang/notebooks/QEP_overlap_20260914/QEP_multiple_shot
#SBATCH -p gpu
#SBATCH -o ./AFM_searching/logs/overlap_preview.%A_%a.out
#SBATCH -e ./AFM_searching/logs/overlap_preview.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=12GB
#SBATCH --time=00:15:00
#SBATCH --array=120-158:2,200-238:2,320-358:2%16

set -euo pipefail
conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
# Real production prefixes, not smoke data: full jobs resume these later.
srun python -u -m AFM_searching.shot_study.overlap_training "${SLURM_ARRAY_TASK_ID}" --N 5 --max-new-epochs 200
