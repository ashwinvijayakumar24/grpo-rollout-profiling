#!/bin/bash
# Build the `grpo` conda env on the PACE *login* node (compute nodes have no internet),
# then pre-download the model so jobs can run with HF_HUB_OFFLINE=1.
#
#   nohup bash scripts/setup_pace_env.sh > logs/setup_env.log 2>&1 &
#
# The env lives in project storage, not $HOME (small quota), and is separate from the
# `llm` env the LoRA project's jobs use. Versions are the DECISIONS.md pin set.
# Ends with SETUP_DONE on success.
set -euo pipefail

STORE=$HOME/ps-simpliearn-0
ENV_PREFIX=${ENV_PREFIX:-$STORE/envs/grpo}
export HF_HOME=${HF_HOME:-$STORE/.hf_cache}
export PIP_CACHE_DIR=$STORE/.pip_cache
VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
MODELS=${MODELS:-"Qwen/Qwen2.5-1.5B-Instruct"}

cd "$(dirname "$0")/.."
module load anaconda3

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create -y -p "$ENV_PREFIX" python=3.12
fi
PY="$ENV_PREFIX/bin/python"
PIP=("$PY" -m pip install --no-input)

"${PIP[@]}" --upgrade pip
# torch first, from the cu129 index, so vLLM does not pull a cu130 build from PyPI.
"${PIP[@]}" torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129
"${PIP[@]}" "$VLLM_WHEEL" --extra-index-url https://download.pytorch.org/whl/cu129
"${PIP[@]}" trl==1.14.2 transformers==5.17.0 accelerate==1.15.0 peft==0.21.2 datasets==5.0.1 \
    nvidia-ml-py pyyaml matplotlib pytest
"${PIP[@]}" -e . --no-deps

"$PY" -m pip check || echo "WARNING: pip check reported conflicts (see above)"
"$PY" - <<'EOF'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
assert torch.version.cuda.startswith("12.9"), "torch is not the cu129 build"
for lib in ("vllm", "trl", "transformers", "accelerate", "peft", "datasets", "flashinfer-python"):
    try:
        print(lib, md.version(lib))
    except md.PackageNotFoundError:
        print(lib, "MISSING")
EOF

for m in $MODELS; do
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken']))"
done

echo SETUP_DONE
