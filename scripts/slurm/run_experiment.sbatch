#!/bin/bash
# Run one experiment spec (all arms x reps) in a single allocation, so every arm is
# measured on the same GPU. inferno is not preemptible; one GPU type is pinned so
# reps never mix hardware. Resumable: rerunning skips completed runs.
#
#   sbatch scripts/slurm/run_experiment.sbatch configs/experiments/E1_breakdown.yaml
#   sbatch --time=03:00:00 scripts/slurm/run_experiment.sbatch configs/experiments/E4_sync_freq.yaml
#SBATCH --job-name=grpo-exp
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
SPEC=${1:?usage: sbatch run_experiment.sbatch SPEC.yaml [extra args for rlstudy.experiment]}
shift

STORE=$HOME/ps-simpliearn-0
REPO=${REPO:-$STORE/grpo-rollout-profiling}
PY=${PY:-$STORE/envs/grpo/bin/python}
export HF_HOME=${HF_HOME:-$STORE/.hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# vLLM compiles kernels and caches them; keep that cache off the small $HOME quota.
export VLLM_CACHE_ROOT=$STORE/.vllm_cache
export TRITON_CACHE_DIR=$STORE/.triton_cache

module load cuda/12.9.1

cd "$REPO"
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "git sha: $(git rev-parse HEAD) dirty: $(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv

"$PY" -m rlstudy.experiment "$SPEC" "$@"
echo EXPERIMENT_DONE
