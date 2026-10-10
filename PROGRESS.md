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
- **`grpo` env built on PACE** (`~/ps-simpliearn-0/envs/grpo`, `logs/setup_env.log`
  on PACE): torch 2.13.0+cu129, vllm 0.30.0+cu129, trl 1.14.2, transformers 5.17.0,
  accelerate 1.15.0, peft 0.21.2, datasets 5.0.1, flashinfer-python 0.6.18.post1;
  `pip check` reports no broken requirements. The vLLM wheel pulled transformers
  5.19.0, which the next install step replaced with the 5.17.0 pin.
- The model pre-download step was killed on the login node (likely a per-process
  limit on login nodes). Retrying with a single download worker.
- **Model weights on PACE.** Two downloads of Qwen2.5-1.5B-Instruct on the login node
  were killed partway. Downloaded on the Mac instead (revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) and copied with rsync in about 3 minutes;
  the `model.safetensors` sha256 on PACE equals the Hub's LFS hash
  (`dd924a11…c6d3ee`, 3,087,467,144 bytes).
- **Smoke job 13909990 submitted** (`configs/experiments/smoke.yaml` on
  `inferno`/`gpu-h100`): 8 vLLM steps with steps 5–6 profiled, plus 4 HF-generate steps.
- **Smoke job 13909990 failed: out of GPU memory** (node atl1-1-03-008-27-0, H100
  80GB HBM3, driver 615.71.09, max SM clock 1980 MHz). Both arms hit
  `torch.OutOfMemoryError` in the loss forward (`_compute_loss` →
  `_full_logits_logps`), with 77.97 GiB already allocated by PyTorch. Cause: all 64
  completions went through the training forward at once, so activations and
  64 × (prompt + 256) × 151k-vocab logits were alive together. The HF arm failed too,
  so vLLM's 0.3 memory share was not the cause.
  Fix (commit after `62b009d`): micro-batches of 16 completions with gradient
  accumulation, still one generation round per optimizer step. Logs:
  `results/smoke/*/rep0.log` on PACE.
- **Smoke job 13918411** (git `fdde4eb`, node atl1-1-03-008-27-0, H100 80GB):
  - `hf_generate` arm **completed**: first end-to-end instrumented run on the GPU.
    Peak `torch.cuda.max_memory_allocated` 57,258,060,800 bytes
    (`results/smoke/hf_generate/rep0/run.json` on PACE). Its 4-step timing is a
    smoke check, not a result.
  - `vllm` arm failed again with out-of-memory in the loss forward, with vLLM holding
    its 0.3 share. Fix: `gpu_memory_utilization` 0.15 and `micro_batch` 8.
- **Smoke job 13926413 completed, both arms** (git `57aea3f`, node atl1-1-03-008-27-0).
  Artifacts in `results/smoke/`. The 29.5 MB profiler trace is gitignored; its only copy is on the Mac (`results/smoke/vllm/rep0/profile_trace.json.gz`) because the PACE copy was deleted when the smoke directory was cleared before the pull.
  This is a single 8-step run, so these are plumbing checks, not results:
  - vLLM path works end to end: weight sync, generation, old-log-prob pass, micro-
    batched loss (8 micro-batches of 8), profiler window, NVML sampler. Peak
    allocated 54.7 GB with vLLM at 0.15.
  - Early signals worth testing properly: step time is about 2.4 s; rollout is about
    a third of it, not the majority; the GPU is busy only about 2% of the weight-sync
    window (`results/smoke/vllm/rep0/profile_busy.json`).
- **Budget projection** (`python -m rlstudy.budget results/smoke/vllm/rep0`, a
  planning estimate): about 102 GPU-minutes for E0–E4 plus about 1 minute of process
  start-up per run; longest single run about 4.5 minutes. Nothing near the 30-minute
  flag. Submitting all five experiments.
- **Experiments submitted** (git `bd0d33a`, `inferno`/`gpu-h100`): E0 13930285,
  E1 13930286, E2 13930287, E3 13930288, E4 13930289. One job per experiment, so
  every arm of an experiment shares one GPU.
- **E0 job 13930285 failed** (timing_on arm, all 3 reps): `EADDRINUSE` on port 29500.
  E0 and E1 ran on the same node (atl1-1-03-008-27-0) at the same time, and every
  run's vLLM process group used the default rendezvous port 29500. The timing_off
  reps completed but are not used: E0's two arms must be measured interleaved in one
  job. Fix in `c7605be` (each run claims a free port). Failed attempt kept on PACE at
  `results/E0_overhead.failed-13930285`; E0 resubmitted in full.
- **E1 timing reps completed** (job 13930286, all 3 reps; the profiled arm was still
  running at the time of this entry).
- **E1 baseline results** (job 13930286, git `bd0d33a`, H100 80GB HBM3 on
  atl1-1-03-008-27-0, 3 reps × 50 steps, first 2 steps dropped). Source:
  `results/E1_breakdown/baseline/arm_summary.json`. Mean ± stdev across reps:
  - Step time 2.266 ± 0.014 s.
  - Shares: rollout_gen 33.2 ± 0.4 %, advantage_loss 57.1 ± 0.5 %, weight_sync
    7.8 ± 0.1 %, optimizer_step 0.8 %, other 1.1 %, reward < 0.1 %.
  - Inside advantage_loss per step: backward 0.653 s, loss forward 0.301 s,
    old-log-prob pass 0.298 s.
  - Weight sync 0.178 ± 0.001 s per sync (every step).
  - Rollout 11,383 ± 296 generated tokens/s; mean completion 134 tokens; 9.8 % of
    completions hit the 256-token cap.
  - Training-batch correctness rose from 0.309 (first 10 steps) to 0.588 (last 10).
    This is the per-step batch, not a held-out evaluation.
  - Open question: NVML reports about 80 % GPU utilization inside weight-sync spans,
    while the smoke run's profiler measured about 2 % busy. NVML averages over up to
    1 s, longer than the 0.18 s sync, so it likely includes the backward pass before
    it. The E1 profiled arm decides this.
- **E1 profiled arm** (same job, steps 20–21 under torch.profiler;
  `results/E1_breakdown/baseline_profiled/rep0/profile_busy.json`). Fraction of each
  span with any GPU kernel/memcpy/memset running: weight_sync **2.0 %** (7.1 ms of
  kernels across two syncs totalling 0.362 s), rollout_gen 65.7 %, old_logprob 83.4 %,
  loss_forward 75.4 %, backward 90.5 %, optimizer_step 93.2 %. This resolves the open
  question above: NVML's ~80 % during sync was its averaging window, not GPU work.
  Caveat: the profiler adds CPU overhead (profiled rollout ≈ 0.93 s/step vs 0.75 s
  unprofiled), so idle fractions of CPU-heavy spans are upper bounds.
- **E3 results** (job 13930288, 3 reps per arm, one job; `results/E3_gen_length/*/arm_summary.json`):
  | cap | step s | rollout % | rollout s | tok/s | mean len | longest len | slot occupancy | truncated |
  |---|---|---|---|---|---|---|---|---|
  | 128 | 1.632 ± 0.012 | 26.0 | 0.425 | 12,719 ± 198 | 84 | 128 | 0.659 | 25.4 % |
  | 512 | 3.386 ± 0.078 | 38.2 | 1.294 | 9,077 ± 128 | 184 | 461 | 0.403 | 1.1 % |
  Rollout time tracks the longest completion in the batch (2.8–3.3 ms per position of
  the longest sample), so a 2.2× longer mean answer cost 3.0× the rollout time and 29 %
  of the throughput. Weight sync is unchanged at 0.179 s, so its share falls as steps
  get longer.
- **E2 results** (job 13930287, 3 reps per arm; `results/E2_group_size/*/arm_summary.json`).
  B = 8 prompts fixed, so completions per step are 32 / 64 / 128:
  | G | step s | rollout s | trainer-side s | tok/s | rollout % | zero-signal groups |
  |---|---|---|---|---|---|---|
  | 4 | 1.517 ± 0.012 | 0.655 | 0.649 | 6,600 ± 639 | 43.2 | 20.6 % |
  | 8 | 2.291 ± 0.029 | 0.765 | 1.304 | 11,179 ± 143 | 33.4 | 10.2 % |
  | 16 | 3.958 ± 0.044 | 1.089 | 2.601 | 17,534 ± 101 | 27.5 | 7.1 % |
  4× the completions cost 1.66× the rollout time (2.66× the throughput); the
  trainer side scaled 2.0× per doubling. G8 reproduces E1's baseline from another job
  (2.291 vs 2.266 s per step, 1.1 % apart). G4 rep0 ran at git `bd0d33a` and the other
  8 runs at `c7605be` (the port fix only; no change to measured code paths).
- **E0 results** (job 13931400, git `c7605be`, 3 seed-paired reps per arm;
  `results/E0_overhead/overhead.json`): full instrumentation 2.271 ± 0.010 s/step vs
  step-boundary-only 2.262 ± 0.012 s/step, **+0.40 %** mean paired difference
  (0.59 %, 0.29 %, 0.32 % per seed). Both arms generated identical tokens per seed.
- **E4 results** (job 13930289, git `c7605be`, 3 reps × 64 steps per arm plus two
  profiled runs; `results/E4_sync_freq/`): one sync costs 0.179–0.185 s regardless of
  interval; step time 2.287 ± 0.016 s (k=1), 2.159 ± 0.013 s (k=4), 2.113 ± 0.009 s
  (k=16), a 7.6 % saving; GPU busy 1.9 % during sync in both profiled arms. Training
  correctness, last 10 steps: 0.654 ± 0.029 (k=1), 0.642 ± 0.039 (k=4),
  0.590 ± 0.075 (k=16): suggestive of a staleness cost, not settled at 3 reps.
- **Housekeeping note:** a `git stash -u` on PACE (to let the checkout pull) moved
  untracked result folders into `stash@{0}` there. Every completed run had already
  been copied to the Mac and committed; the stash stays on PACE as a backup.
- **E5 submitted** (job 13931904): vLLM vs HF generate, 3 reps each. E1–E4 completed
  cleanly, which was the precondition for the stretch.
- **E5 results** (job 13931904, 3 reps each; `results/E5_backend/`): vLLM 2.278 ±
  0.018 s/step, rollout 0.761 s, 11,244 tok/s; HF generate 7.046 ± 0.148 s/step,
  rollout 6.013 s, 1,422 tok/s (7.9× slower generation, 3.1× slower step). vLLM's
  extra cost: sync 0.182 s + old-log-prob pass 0.293 s. Correctness, last 10 steps:
  0.588 ± 0.061 (vLLM) vs 0.601 ± 0.039 (HF). Baseline reproduces across three jobs:
  2.266 (E1), 2.291 (E2 G8), 2.278 (E5) s/step.
- **All planned experiments done (E0–E5).** WRITEUP.md, RESUME.md, and FUTURE.md
  complete.
