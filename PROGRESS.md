# Progress log

A running record of what was done, what was measured, and where each result lives.
Newest entries at the bottom. Rules:

- Every number here links to the file in `results/` that it came from, plus the git
  SHA and Slurm job ID of the run that produced it.
- Local (Mac) runs are smoke tests. Their timings are never quoted as results.
- If a run is redone, the old entry stays and gets a note saying why it was superseded.

## 2026-10-08

- **Plan written** (PLAN.md). Repo created at
  github.com/ashwinvijayakumar24/grpo-rollout-profiling.
- **Phase 0 reading done** (NOTES.md). Covers GRPO as data flow, TRL's architecture,
  how TRL's vLLM weight sync works, and where to hook each timing bucket.
- **Framework decision: TRL `GRPOTrainer` + vLLM colocate** (DECISIONS.md). Chosen
  over veRL for install fit on PACE (CUDA 12.9 wheels), single-process timing, and an
  easy hook for the E4 sync-frequency experiment. SGLang is dropped from E5.
- **PACE access checked.** SSH from the Mac works. Submitted probe job 13907139
  (`scripts/slurm/probe_env.sbatch`) to record GPU model, memory, and driver version.
  Result: pending.
