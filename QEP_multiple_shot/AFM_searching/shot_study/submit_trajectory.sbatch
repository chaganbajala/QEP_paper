#!/bin/bash -l
#SBATCH -J afm_shot_traj
#SBATCH -D ./QEP_multiple_shot
#SBATCH -o ./AFM_searching/logs/trajectory.%A_%a.out
#SBATCH -e ./AFM_searching/logs/trajectory.%A_%a.err
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=12GB
#SBATCH --time=06:00:00
#SBATCH --array=0-23%12

conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
srun python -u -m AFM_searching.shot_study.cluster_dispatch trajectory "${SLURM_ARRAY_TASK_ID}"
