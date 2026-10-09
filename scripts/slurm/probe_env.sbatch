#!/bin/bash
# Five-minute probe: record the GPU model, memory, driver, and CUDA versions on the
# node, and, if the grpo env exists, check that its torch sees the GPU and vLLM
# imports. Override the QOS/partition on the command line, e.g.
#   sbatch --qos=inferno --partition=gpu-h100 scripts/slurm/probe_env.sbatch
#SBATCH --job-name=grpo-probe
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is), qos=$SLURM_JOB_QOS partition=$SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | head -4

PY=$HOME/ps-simpliearn-0/envs/grpo/bin/python
if [[ -x "$PY" ]]; then
    HF_HUB_OFFLINE=1 "$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", repr(e))
PYEOF
else
    echo "grpo env not built yet"
fi
