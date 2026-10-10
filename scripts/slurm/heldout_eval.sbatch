#!/bin/bash
# E6: held-out accuracy before vs after GRPO training, on the frozen 200-problem eval
# split (from GSM8K test). One allocation, so every model is scored on the same GPU.
#   base model                          -> results/E6_heldout/base
#   E1 baseline config, 50 steps, seeds 0-2 -> results/E6_heldout/trained_50/rep{0,1,2}
#   same config, 125 steps (1 epoch), seeds 0-2 -> results/E6_heldout/trained_125/rep{0,1,2}
#   (seed 0 first ran in job 13943559; seeds 1-2 added after its lenient gain needed confirming)
# Resumable: a model whose eval.json exists is skipped. Checkpoints go to project
# storage, not git.
#SBATCH --job-name=grpo-E6_heldout
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
STORE=$HOME/ps-simpliearn-0
REPO=${REPO:-$STORE/grpo-rollout-profiling}
PY=${PY:-$STORE/envs/grpo/bin/python}
CKPT=$STORE/grpo_ckpts/E6
export HF_HOME=${HF_HOME:-$STORE/.hf_cache} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=$STORE/.vllm_cache TRITON_CACHE_DIR=$STORE/.triton_cache
module load cuda/12.9.1
# Put the env's tools (ninja, used by FlashInfer's JIT kernel builds) on PATH; the env's
# python is called by absolute path, so conda activation never did this (E6 job 13943303).
export PATH="$(dirname "$PY"):$PATH"
cd "$REPO"
OUT=results/E6_heldout
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is), git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

evaluate() {  # model out label
    if [[ -f "$2/eval.json" ]]; then echo "[skip] $2"; return; fi
    "$PY" -m rlstudy.evaluate --model "$1" --out "$2" --label "$3"
}
train() {  # steps seed run_dir ckpt_dir
    if [[ -f "$4/model.safetensors" ]]; then echo "[skip] train $4"; return; fi
    rm -rf "$3"
    "$PY" -m rlstudy.train --config configs/baseline.yaml --out "$3" \
        --set experiment='"E6_heldout"' --set seed="$2" --set grpo.max_steps="$1" \
        --set timing.nvml_interval_s=null --set save_final_to="\"$4\"" > "$3.log" 2>&1
}

evaluate Qwen/Qwen2.5-1.5B-Instruct "$OUT/base" base
for s in 0 1 2; do
    mkdir -p "$OUT/train_runs"
    train 50 "$s" "$OUT/train_runs/steps50_rep$s" "$CKPT/steps50_rep$s"
    evaluate "$CKPT/steps50_rep$s" "$OUT/trained_50/rep$s" "trained_50_seed$s"
done
for s in 0 1 2; do
    train 125 "$s" "$OUT/train_runs/steps125_rep$s" "$CKPT/steps125_rep$s"
    evaluate "$CKPT/steps125_rep$s" "$OUT/trained_125/rep$s" "trained_125_seed$s"
done
echo E6_DONE
