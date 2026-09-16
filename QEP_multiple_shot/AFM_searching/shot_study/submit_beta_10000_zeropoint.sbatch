#!/bin/bash -l
#SBATCH -J afm_beta10000
#SBATCH -D /zeropoint/u/qinwang/notebooks/QEP_fixed_shot_verify_20260914/QEP_multiple_shot
#SBATCH -p gpu
#SBATCH -o ./AFM_searching/logs/beta10000_zp.%A_%a.out
#SBATCH -e ./AFM_searching/logs/beta10000_zp.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
#SBATCH --time=00:30:00
#SBATCH --array=0-1

set -e
conda activate qep-cuda
export MPLCONFIGDIR=/tmp/qinwang-mpl-${SLURM_JOB_ID}
sizes=(5 9)
srun python -u -m AFM_searching.shot_study.beta_shot_scan --N "${sizes[$SLURM_ARRAY_TASK_ID]}" --Ng 10000
