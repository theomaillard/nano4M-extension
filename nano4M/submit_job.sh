#!/bin/bash
#SBATCH --job-name=extnanofm         # Change as needed
#SBATCH --time=07:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out    # Output log file
#SBATCH --error=interactive_job.err     # Error log file
#SBATCH --partition=l40s

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

export MASTER_PORT=$((10000 + RANDOM % 50000))
export WANDB_API_KEY=$WANDB && OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_training.py --config $CONFIG_FILE