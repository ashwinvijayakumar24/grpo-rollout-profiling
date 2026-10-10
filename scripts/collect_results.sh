#!/bin/bash
# Copy results from PACE to this checkout, then rebuild summaries and figures.
#   bash scripts/collect_results.sh [EXPERIMENT ...]     (default: all)
# Large profiler traces are copied too but stay out of git (.gitignore); per-run
# logs are copied next to their runs so failures can be diagnosed offline.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=pace:ps-simpliearn-0/grpo-rollout-profiling/results
EXPS=("$@")
[[ ${#EXPS[@]} -eq 0 ]] && EXPS=(E0_overhead E1_breakdown E2_group_size E3_gen_length E4_sync_freq)
PY=${PY:-.venv/bin/python}
for e in "${EXPS[@]}"; do
    rsync -a "$REMOTE/$e" results/ 2>/dev/null && echo "copied $e" || echo "no results yet for $e"
done
"$PY" -m rlstudy.analyze $(for e in "${EXPS[@]}"; do [[ -d results/$e ]] && echo results/$e; done)
"$PY" -m rlstudy.figures
