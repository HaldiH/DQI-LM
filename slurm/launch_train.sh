#!/bin/bash
#SBATCH --job-name=dqi-finetune
#SBATCH --output=logs/dqi-%j.out
#SBATCH --error=logs/dqi-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9"
#SBATCH --partition=shared-gpu
#SBATCH --time=10:00:00
#SBATCH --mem=128G

module purge

mkdir -p logs outputs

if [ $# -gt 0 ]; then
  CONFIG=$1
else
  CONFIG="configs/config.yaml"
fi

echo "Starting training..."
make train CONFIG=$CONFIG

echo "Job finished."
