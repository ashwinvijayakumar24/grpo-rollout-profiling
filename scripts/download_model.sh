#!/bin/bash
# Download model weights into the shared HF cache on the PACE login node.
# Xet transfers are disabled: on 2026-10-09 the Xet path was killed / stalled on the
# login node (see PROGRESS.md); plain HTTP with one worker is slower but reliable.
#   setsid nohup bash scripts/download_model.sh [MODEL_ID] > logs/download.log 2>&1 < /dev/null &
set -euo pipefail
MODEL=${1:-Qwen/Qwen2.5-1.5B-Instruct}
STORE=$HOME/ps-simpliearn-0
export HF_HOME=${HF_HOME:-$STORE/.hf_cache}
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
find "$HF_HOME/hub/models--${MODEL/\//--}" -name "*.incomplete" -delete 2>/dev/null || true
"$STORE/envs/grpo/bin/python" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=1))
PY
echo DOWNLOAD_DONE
