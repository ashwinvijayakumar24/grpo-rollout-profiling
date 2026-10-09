# Blockers

Things that stop GPU work, what was built anyway, and what unblocks each one.

| ID | Status | Needs | Blocks | Built anyway |
|---|---|---|---|---|
| B1 | **Resolved** 2026-10-09 | GPU driver version | — | Probe 13908530 (inferno, gpu-h100): H100 80GB HBM3, driver 615.71.09 (`results/env/probe-13908530.txt`) |
| B2 | Proceeding | Ashwin's OK on TRL over veRL (DECISIONS.md) | Nothing: Ashwin asked for autonomous progress; switching later would replace only `rlstudy/instrumented.py` | Everything else is framework-independent |
