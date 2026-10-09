# Where does GRPO training time go? A rollout-infrastructure profile

> **Status: skeleton.** No GPU experiment has run yet. Every number below is TODO
> until it can be traced to a file in `results/`. Nothing here is estimated.

## Summary

TODO after E1–E4.

## 1. Setup

| Item | Value | Source |
|---|---|---|
| GPU / driver / CUDA | TODO | `results/<exp>/<arm>/rep*/run.json` → `env.gpus` |
| Model | Qwen2.5-1.5B-Instruct, full fine-tune, bf16 | `configs/baseline.yaml` |
| Framework | TRL 1.14.2 `GRPOTrainer`, vLLM 0.30.0 colocated on the same GPU | DECISIONS.md |
| Task | GSM8K, 1,000 frozen training problems (seed 0) | `data/gsm8k_subset/manifest.json` |
| Reward | exact match on the number after `####` (1.0), plus 0.1 for the `#### <n>` format | `rlstudy/reward.py` |
| Batch | 8 prompts × 8 samples = 64 completions per step, max 256 new tokens | `configs/baseline.yaml` |
| KL / reference model | none (`beta = 0`) | `configs/baseline.yaml` |

## 2. Methodology

**What one training step contains.** With TRL on one GPU, a step runs strictly in
sequence in one process: copy the latest weights into vLLM (weight sync), generate
64 answers (rollout), score them (reward), run a no-grad forward pass to get the
trainer's own log-probabilities for the sampled tokens (old log-probs, used to
correct for vLLM/trainer numeric differences), compute group-relative advantages and
the loss, backpropagate, and step the optimizer. NOTES.md traces this data flow.

**How time is attributed.** `rlstudy/timing.py` charges every nanosecond of a step
to exactly one bucket:

| Bucket | Contains |
|---|---|
| `rollout_gen` | vLLM `generate` (or HF `generate`) and converting outputs to token lists |
| `weight_sync` | `VLLMGeneration.sync_weights()`: one device-to-device copy per parameter |
| `reward` | the two reward functions (pure Python) |
| `advantage_loss` | tokenizing/padding, old-log-prob forward, advantages, loss forward, backward |
| `optimizer_step` | `optimizer.step()` |
| `other` | the remainder: next-batch fetch, gradient clipping, `zero_grad`, LR schedule, logging |

Three rules keep the attribution honest:

1. **Wait for the GPU before reading the clock.** GPU kernels run asynchronously, so
   every span boundary calls `torch.cuda.synchronize()`. Without it, the backward
   pass's time would be charged to whichever later stage first waits on the GPU.
   `tests/test_timing.py` demonstrates both cases on a simulated GPU.
2. **Never count time twice.** TRL calls generation and rewards from inside its
   training step, so stages nest. Each stage is charged its exclusive time (its time
   minus the stages inside it), and `other` is the remainder, so buckets always sum
   to the step's wall clock.
3. **Step windows tile the run.** Each step's window runs from one `on_step_end` to
   the next, so no wall-clock time falls between steps.

**What is excluded from summaries.** The first 2 steps of every run (CUDA graph
capture and compilation warm-up) and any step captured under `torch.profiler`
(profiling slows steps; those runs are separate "profiled" arms).

**Measurement overhead (E0).** TODO: step time with full instrumentation vs with only
step-boundary syncs.

**Repetitions.** Each configuration runs 3 times with seeds 0, 1, 2. Arms in one
experiment run interleaved in one Slurm job on one GPU. Tables report mean ± sample
standard deviation across reps; figures show min–max whiskers.

## 3. E1: Where the time goes at baseline

![E1 breakdown](results/figures/E1_breakdown.png)

| Stage | Share of step time | Seconds per step |
|---|---|---|
| Rollout generation | TODO | TODO |
| Advantage + loss + backward | TODO | TODO |
| of which: old-log-prob forward | TODO | TODO |
| Weight sync | TODO | TODO |
| Optimizer step | TODO | TODO |
| Reward | TODO | TODO |
| Other | TODO | TODO |

GPU busy fraction per stage (profiled arm): TODO (`results/E1_breakdown/baseline_profiled/rep0/profile_busy.json`).

Mechanism: TODO.

Reward curve (does training work at all?): ![E1 reward](results/figures/E1_reward.png) TODO.

## 4. E2: Rollout throughput vs samples per prompt

![E2 throughput](results/figures/E2_throughput.png)

TODO: tokens/s at G = 4, 8, 16; how the rollout share changes; mechanism (batch size
in vLLM's continuous batching, prefix sharing across a group's identical prompts).

## 5. E3: Generation-length cap

![E3 breakdown](results/figures/E3_breakdown.png)

TODO: rollout share at 128 vs 512 tokens; truncation rate at each cap; mechanism
(decode is sequential per token; a batch waits for its longest sample).

## 6. E4: Weight-sync cost and GPU idleness

![E4 sync per step](results/figures/E4_sync_per_step.png)

TODO: seconds per sync, per-step amortized cost at k = 1, 4, 16; GPU busy fraction
during sync from the profiled arms; reward next to time (stale samplers).

## 7. What I'd build differently in a rollout system

Every claim cites the experiment that supports it. Claims not directly measured are
marked *(speculative)*.

TODO after E1–E4.

## 8. Interview questions this study answers

| Question | Where the answer lives |
|---|---|
| In GRPO post-training, where does the time go, and why? | §3 (E1) |
| How do samples-per-prompt and generation length change rollout cost? | §4 (E2), §5 (E3) |
| What does trainer→sampler weight sync cost when they share one GPU, and is the GPU idle during it? | §6 (E4) |
| How do you time GPU work correctly when kernels run asynchronously? | §2, `tests/test_timing.py` |
| If you were building a rollout system, what would you change first, and what evidence says so? | §7 |
