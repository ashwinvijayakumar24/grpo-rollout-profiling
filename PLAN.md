# RL Rollout-Infrastructure Study — Plan

Status: planning (2026-10-08). Nothing has been run yet. Every number in this file
is a planning estimate and is labelled as one; none of them may appear in WRITEUP.md.

## 1. The goal in one paragraph

We teach a small language model to do grade-school math better by letting it try
problems, checking its answers automatically, and nudging it toward the attempts that
were right. That training method is GRPO (Group Relative Policy Optimization: for each
question, sample a group of answers, score each, and push the model toward answers that
scored above the group's average). The training itself is not the point. The point is to
put a stopwatch on every stage of each training step and answer one systems question:
**where does the time actually go, and what in the "generate answers" path is the
bottleneck?** The expected answer is that generating answers dominates, but we measure it
rather than assume it.

## 2. Hardware reality (affects the whole plan)

- This Mac is an Apple M4 with 16 GB of memory and no NVIDIA GPU. vLLM, SGLang, and
  veRL all need CUDA, so **no real experiment can run here**.
- The GPU path is PACE Phoenix (`ssh pace`, account `paceship-simpliearn`, project
  storage `~/ps-simpliearn-0`, `module load cuda/12.9.1` + `module load anaconda3`).
  The working Slurm setup to copy is `custom_llm/llm_finetuning/scripts/slurm/`, and
  its job history is in `llm_finetuning/docs/PACE_JOBS.md`. SSH from this Mac works
  (checked 2026-10-08).
- Two QOS (quality-of-service tiers) are available:
  - `embers`: free but preemptible (a job can be killed to make room), 8-hour limit,
    and at most 50 queued jobs per user. The LoRA project already holds about 27 of
    those 50, so this study must stay well under the remaining slots.
  - `inferno`: not preemptible. The LoRA project uses it for its benchmarks.
- **Decision: experiment runs use `inferno` on one pinned GPU type** (H100 preferred).
  This is a timing study, so a run that gets preempted halfway, or reps that land on
  different GPU models, would corrupt the comparison. Smoke and setup jobs may use
  `embers`. Every run records the GPU name, memory, driver, and CUDA version.
- Two more PACE facts shape the run design:
  - **The queue is busy.** Short `--time` limits make jobs eligible for backfill
    (Slurm slotting small jobs into gaps), so each experiment is one short job, and
    all settings compared within an experiment run in the same job, on the same GPU.
  - **Compute nodes often have no internet.** Model weights, GSM8K, and pip packages are
    downloaded on the login node into `~/ps-simpliearn-0` before submitting, and jobs run
    with `HF_HUB_OFFLINE=1`.
- The study gets its own conda env (`grpo`), so it cannot break the `llm` env that the
  LoRA jobs use.
- Consequence: we build and test everything that does not need a GPU locally (reward
  function, trace format, analysis and charts, a tiny smoke loop using plain Hugging Face
  generation on the Mac), and ship Slurm scripts that run the real experiments on the
  GPU. Local smoke numbers are never reported as results.

## 3. Phases

### Phase 0 — Reading and framework decision (before any code)

1. **Reading, 30 minutes maximum, written up in NOTES.md.** Three topics only:
   - GRPO as data flow: which tensors exist at each stage (prompts → sampled token ids
     → rewards → group-normalized advantages → per-token log-probs → loss), and which
     process or device each one lives on.
   - The chosen framework's architecture doc.
   - How that framework wires its generation engine and copies updated weights from the
     trainer into the generator ("weight sync").
2. **Framework comparison in DECISIONS.md**, scoring veRL, OpenRLHF, and TRL's
   `GRPOTrainer` on three criteria: runs on one GPU; the generation backend can be
   swapped and timed; the weight-sync code is readable.
   - Starting expectation, to be confirmed by reading: veRL does run on one GPU, but it
     routes everything through Ray (a distributed task scheduler), which makes per-stage
     timing harder to attribute. TRL with vLLM in "colocate" mode (trainer and generator
     share one GPU) has a short, readable sync path. The task's default bias is veRL; we
     switch to TRL only if the reading shows veRL's instrumentation cost is too high.
   - **Checkpoint: report the decision to Ashwin before building.**

### Phase 1 — Local scaffolding (Mac, no GPU)

- Data: download GSM8K, take a fixed, seeded 1,000-problem training subset and a
  200-problem held-out set. Record the dataset revision hash.
- Reward function: exact-match on the final numeric answer, plus a small format reward
  (answer appears in the required `#### <number>` form). Unit tests with tricky cases:
  commas in numbers, trailing periods, negative numbers, multiple answers in one output.
- Trace schema: one JSON line per training step with these timed buckets —
  `rollout_gen`, `reward`, `advantage_loss` (includes the forward and backward pass),
  `optimizer_step`, `weight_sync`, `other` (computed as total minus the sum, so the
  buckets always add up). Each record also carries step id, config hash, git SHA, tokens
  generated, and mean reward.
- Timing method: CUDA events or `torch.cuda.synchronize()` at bucket boundaries, so GPU
  work that runs asynchronously is charged to the right bucket. This is the most
  important correctness detail in the project; it gets its own test.
- Analysis script: reads `results/<experiment>/<rep>/trace.jsonl` and produces the
  tables and charts. Built and tested on synthetic traces first.
- Smoke loop: the full pipeline on a tiny model with HF `generate` on the Mac, 5 steps,
  to prove the plumbing works end to end.

### Phase 2 — First GPU run (PACE)

- Pin the environment: exact GPU, driver, CUDA, torch, vLLM, and framework versions
  written to `results/env.json` by the run script itself.
- Baseline loop: Qwen2.5-1.5B-Instruct (or Llama-3.2-1B-Instruct if memory is tight),
  LoRA if a full fine-tune does not fit alongside the generator. Success criterion is
  only "the reward curve goes up and nothing crashes."
- Measure the timing overhead of the instrumentation itself (run once with timing on and
  once with it off) so we can state how much the stopwatch distorts the result.

### Phase 3 — Experiments (3 repetitions each, report mean and spread)

| ID | Question | What changes | Runs |
|---|---|---|---|
| E1 | Where does a step's time go at baseline? | Nothing — baseline config | 3 |
| E2 | How does generation throughput scale with answers per question? | Group size 4, 8, 16 | 9 |
| E3 | How much does answer length shift the time split? | Max new tokens 128 vs 512 | 6 |
| E4 | What does weight sync cost, and is the GPU idle during it? | Sync every 1, 4, 16 steps | 9 |
| E5 | Stretch: does the generation engine matter? | vLLM vs SGLang vs HF generate | 9 |

- E5 runs only if E1–E4 are complete and clean. An optional extra arm for E5 is the
  user's own `llm_serving_layer` as the generator; that goes in FUTURE.md unless E1–E4
  finish early.
- E4 also records GPU utilization over time (a sampler polling `nvidia-smi` or NVML)
  so we can see whether the GPU sits idle during a sync and whether overlap is possible.
- **Budget (planning estimate, not a result):** about 50 steps per run, a few minutes
  each on one GPU, so roughly 27 runs for E1–E4 at 1.5–2 GPU-hours total. No single run
  is designed to exceed 30 GPU-minutes. Any run whose first measured step time projects
  past 30 minutes is stopped and flagged before continuing.

### Phase 4 — Writeup

- **WRITEUP.md:** methodology; the E1 time-breakdown chart; one section per experiment
  with the mechanism behind the result; "what I'd build differently in a rollout system,"
  where every claim cites the experiment that supports it and anything not measured is
  marked *speculative*; the five interview questions this study answers and where each
  answer lives.
- **RESUME.md:** two honest bullet points, every number traceable to `results/`.
- **FUTURE.md:** anything that tempted us toward multi-node, reward models, or algorithm
  changes.
- **One command** reproduces the baseline: `make baseline` (or `bash run.sh baseline`).

## 4. Rules that apply throughout

- No number is ever estimated or invented in WRITEUP.md or RESUME.md. Every figure
  links to a file in `results/`. Cells for experiments not yet run say TODO.
- Scope creep goes to FUTURE.md, not into the code.
- Commit and push after each meaningful change; GPU-dependent work waiting on B3 is
  pinned in a BLOCKERS.md here, and local work continues meanwhile.

## 5. Planned layout

```
rl_rollout_study/
  PLAN.md  NOTES.md  DECISIONS.md  FUTURE.md  WRITEUP.md  RESUME.md  BLOCKERS.md
  configs/        baseline.yaml, e2_group_*.yaml, e3_len_*.yaml, e4_sync_*.yaml
  rlstudy/        data.py, reward.py, timing.py, trace.py, train.py, analyze.py
  scripts/slurm/  run_experiment.sbatch
  tests/          reward, trace schema, timing attribution, analysis
  results/        <experiment>/<rep>/{trace.jsonl, config.yaml, env.json, gpu_util.csv}
```

## 6. Open decisions for Ashwin

1. None open. Repo: github.com/ashwinvijayakumar24/grpo-rollout-profiling (standalone).
