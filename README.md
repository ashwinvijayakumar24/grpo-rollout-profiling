# grpo-rollout-profiling

Where does wall-clock time go in GRPO post-training of a small LLM, and what are the
bottlenecks in the rollout (answer-generation) path?

This is a systems study, not new RL research. It runs a small GRPO loop on GSM8K
math with an exact-match reward, times every stage of every training step, and
measures how the time split changes with group size, generation length, and
weight-sync frequency.

## Results (H100 80GB, Qwen2.5-1.5B, TRL + vLLM colocated)

![Where a GRPO step's time goes](results/figures/E1_breakdown.png)

- Rollout was **33 %** of a 2.27 s step; trainer-side compute was **57 %** (E1).
- Weight sync: **0.18 s** per sync with the GPU busy **2 %** of it (E1, E4).
- Rollout time follows the longest sample: **40 %** decode-slot occupancy at a
  512-token cap (E3).
- 4× the samples per step cost **1.66×** the rollout time (E2).
- vLLM generated **7.9×** faster than HF `generate` (E5).
- Instrumentation overhead: **0.4 %** (E0).

Full methodology, per-experiment mechanisms, and what I'd build differently:
[WRITEUP.md](WRITEUP.md). Every number traces to a file in `results/`; the run log is
[PROGRESS.md](PROGRESS.md).

## Layout

| Path | What |
|---|---|
| `rlstudy/timing.py` | step timer: GPU-synchronized, exclusive-time buckets |
| `rlstudy/instrumented.py` | `ProfiledGRPOTrainer`, TRL's GRPO trainer with timing hooks |
| `rlstudy/train.py` | one run from a YAML config |
| `rlstudy/experiment.py` | all arms x reps of one experiment, resumable |
| `rlstudy/analyze.py`, `plots.py`, `figures.py` | summaries, rep statistics, charts |
| `rlstudy/gpumon.py` | NVML sampler and profiler-trace GPU busy fraction |
| `configs/` | baseline and experiment specs (E0-E4) |
| `scripts/slurm/` | PACE job scripts |

## Commands

```
make test                 # unit tests
make smoke-local          # 3-step tiny-model run on any machine
make baseline             # instrumented baseline loop on a CUDA GPU
make submit-E1            # on the PACE login node
make analyze figures      # tables and charts from results/
```

Docs: [PLAN.md](PLAN.md) · [NOTES.md](NOTES.md) · [DECISIONS.md](DECISIONS.md) ·
[WRITEUP.md](WRITEUP.md) · [FUTURE.md](FUTURE.md)
