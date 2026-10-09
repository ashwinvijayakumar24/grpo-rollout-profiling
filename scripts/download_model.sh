#!/bin/bash
# Download model weights into the shared HF cache on the PACE login node.
# NOTE: on 2026-10-09 every attempt to fetch the 3.1 GB weight file on the login node
# was killed partway (with and without Xet), most likely by login-node process limits.
# What worked: download on a laptop, then
#   rsync -a --copy-unsafe-links <HF_HOME>/hub/models--Qwen--Qwen2.5-1.5B-Instruct \
#         pace:ps-simpliearn-0/.hf_cache/hub/
# and verify the blob's sha256 equals its file name. Kept for small models.
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
