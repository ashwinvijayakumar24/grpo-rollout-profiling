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
- **Local build (Mac, no GPU), all tested** (`make test`; 94 fast + 7 tiny-model tests):
  - Frozen GSM8K subset: 1,000 train (from GSM8K train) / 200 eval (from GSM8K test),
    seed 0, dataset revision `740312add88f781978c0658806c59bc2815b9866`
    (`data/gsm8k_subset/manifest.json`).
  - Rewards: strict exact match after the last `####` (Decimal comparison) + 0.1
    format bonus.
  - Step timer with GPU syncs at span boundaries and exclusive-time nesting; trace
    writer (`run.json` with env capture, flushed `trace.jsonl`).
  - `ProfiledGRPOTrainer` hooks for all six buckets plus sub-buckets (old-log-prob,
    loss forward, backward); owns the vLLM sync schedule for E4.
  - NVML sampler + torch.profiler window reduced to per-stage GPU busy fraction.
  - Analysis (warm-up and profiled steps dropped; mean ± stdev over reps), charts,
    experiment runner (interleaved, paired seeds, resumable, ledger), Slurm wrapper.
  - The pinned stack (torch 2.13.0, trl 1.14.2, transformers 5.17.0, peft 0.21.2,
    datasets 5.0.1) installs on the Mac. vLLM is CUDA-only, so the vLLM path is
    untested until the first GPU job.
- **Issue found and worked around:** transformers 5.17.0's threaded weight loader
  segfaults inside a training process on macOS/MPS. Local runs set
  `HF_DEACTIVATE_ASYNC_LOAD=1` (Mac only; `rlstudy/train.py`).
- **PACE:** `grpo` env install started on the login node
  (`scripts/setup_pace_env.sh`, env at `~/ps-simpliearn-0/envs/grpo`). The vLLM
  0.30.0 cu129 wheel URL was confirmed to exist (545 MB). The `embers` probe
  13907139 sat in the queue for over an hour; a second probe 13908530 was
  submitted on `inferno`/`gpu-h100`.

## 2026-10-09

- **GPU probe result** (job 13908530, `inferno`/`gpu-h100`, node atl1-1-03-008-32-0):
  NVIDIA H100 80GB HBM3, 81,559 MiB, driver 615.71.09; torch 2.13.0+cu129 sees the GPU.
  Saved at `results/env/probe-13908530.txt`. Blocker B1 resolved; the `embers` probe
  13907139 was cancelled as redundant.
