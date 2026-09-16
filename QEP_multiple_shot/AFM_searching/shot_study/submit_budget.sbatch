#!/bin/bash -l
#SBATCH -J afm_shot_budget
#SBATCH -D ./QEP_multiple_shot
#SBATCH -o ./AFM_searching/logs/budget.%A_%a.out
#SBATCH -e ./AFM_searching/logs/budget.%A_%a.err
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=12GB
#SBATCH --time=08:00:00
#SBATCH --array=0-99%20

conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.cluster_dispatch budget "${SLURM_ARRAY_TASK_ID}"
