#!/bin/bash -l
#SBATCH -J afm_budget_noise
#SBATCH -D ./QEP_multiple_shot
#SBATCH -o ./AFM_searching/logs/budget_noise.%A_%a.out
#SBATCH -e ./AFM_searching/logs/budget_noise.%A_%a.err
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=12GB
#SBATCH --time=01:00:00
#SBATCH --array=0-79%20

conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.verify_budget_noise "${SLURM_ARRAY_TASK_ID}"
