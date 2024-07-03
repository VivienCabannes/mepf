#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=mepf
#SBATCH --output=/checkpoint/vivc/mepf/exp-%a.out
#SBATCH --error=/checkpoint/vivc/mepf/exp-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=vivc@meta.com

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --array=1-9070


python /private/home/vivc/code/mepf/src/mepf/experiments/grid_script.py --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --save-dir /checkpoint/vivc/mepf
