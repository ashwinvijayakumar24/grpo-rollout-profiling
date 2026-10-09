#!/bin/bash
# One-minute probe: record the GPU model, memory, driver, and CUDA versions on the
# node, so the software pins in DECISIONS.md can be checked against the real driver.
#SBATCH --job-name=grpo-probe
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | head -4
