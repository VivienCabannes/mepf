#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=bandit
#SBATCH --output=/checkpoint/vivc/mepf/exp-%a.out
#SBATCH --error=/checkpoint/vivc/mepf/exp-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=vivc@meta.com

# Job specification
#SBATCH --partition=devlab
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --array=1-500


python /private/home/vivc/code/entropicsearch/src/mepf/experiments/grid_script.py --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --save-dir /checkpoint/vivc/bandits
