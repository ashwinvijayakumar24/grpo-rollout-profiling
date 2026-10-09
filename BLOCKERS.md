# Blockers

Things that stop GPU work, what was built anyway, and what unblocks each one.

| ID | Status | Needs | Blocks | Built anyway |
|---|---|---|---|---|
| B1 | Open | Probe job 13907139 to report the PACE GPU driver version | Building the `grpo` env with the pinned vLLM/torch wheels (needs driver ≥ 575.51.03 for cu129) | All local code and tests |
| B2 | Open | Ashwin's OK on TRL over veRL (DECISIONS.md) | Nothing yet; local work is framework-light until the trainer subclass | Reward, data, timing, trace, analysis |
