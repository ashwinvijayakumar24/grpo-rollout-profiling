# Where Does GRPO Training Time Go? Profiling the Rollout Path of LLM Reinforcement Learning

An infrastructure study of reinforcement-learning post-training for LLMs. It runs a working GRPO loop (Group Relative Policy Optimization: sample several answers per question, score each with a verifiable reward, and push the model toward the above-average ones) on GSM8K math with Qwen2.5-1.5B-Instruct, using TRL's trainer with a vLLM sampler on the same GPU. It then attributes **every training step's wall-clock time** to rollout generation, reward, loss and backward, the optimizer, and the trainer→sampler weight sync, and explains each number mechanistically.

This is not a new algorithm, a training framework, or reward-model research. It is a measured answer to one systems question: **where does RL training time actually go, and what are the bottlenecks in the rollout path?**

Benchmarked on an NVIDIA H100 80GB HBM3 (driver 615.71.09, Georgia Tech PACE Phoenix). Every number below resolves to a committed artifact in [`results/`](results/) carrying its Slurm job ID, GPU, seed, config hash, git SHA, and library versions.

**The common assumption did not hold at this scale.** Rollout generation is often described as the dominant cost of RL post-training. Here it was a third of the step: the trainer side was larger, weight sync was mostly idle GPU time, and rollout cost was governed by the *longest* sample in a batch rather than the average one.

| | question | result |
|---|---|---|
| **E1** | where does a step's time go? | rollout **33%**, scoring + loss + backward **57%**, weight sync **8%** of a 2.27 s step |
| **E2** | rollout cost vs samples per prompt | **4×** the samples cost only **1.66×** the rollout time; trainer side scales linearly |
| **E3** | generation-length cap | 128→512 tokens: answers **2.2×** longer, rollout **3.0×** slower; only **40%** of decode slots busy |
| **E4** | weight-sync cost and frequency | **0.18 s** per sync with the GPU busy **2%** of it; syncing every 16 steps saves **7.6%** |
| **E5** | vLLM vs HF `generate` | vLLM generates **7.9×** faster; **3.1×** faster steps despite its sync and correction pass |
| **E6** | does training work? (held-out) | strict accuracy **23% → 71%**; mostly format learning, **+5 points** format-agnostic |
| **E0** | does measuring distort it? | instrumentation overhead **0.4%** of step time |

![Where a GRPO step's wall-clock time goes](results/figures/E1_breakdown.png)

---

## Why this exists

RL post-training with verifiable rewards (math, code) has become a standard way to improve LLM reasoning, and its cost is dominated by systems concerns, not math: a generation engine and a training engine have to take turns on the same hardware and stay in sync. Most write-ups report end-to-end throughput. This project opens up a single training step instead:

- **Two engines, one GPU.** TRL's `GRPOTrainer` keeps the trainable model in PyTorch and hands generation to vLLM, a separate inference engine with its own copy of the weights. After every optimizer step, the new weights have to be copied into vLLM. How much does that cost, and what is the GPU doing meanwhile?
- **Generation is decode-bound.** vLLM generates all samples together, one token per sample per step, until the longest finishes. How do group size and answer length change that?
- **The trainer side is not free either.** Besides the loss forward and backward, TRL runs an extra no-grad forward pass to correct for small numeric differences between vLLM and the trainer. How big is it?

Answering these needs timing that is correct on a GPU, where kernels run asynchronously and a naive timer charges work to the wrong stage. Building that, and proving it does not distort what it measures, is half of the project.

---

## Results

> All experiments use the same baseline unless stated: 8 prompts × 8 samples = 64 completions per step, max 256 new tokens, full fine-tune with bf16 mixed precision, no KL term (`beta = 0`), 3 seed-paired repetitions per configuration, first 2 warm-up steps dropped. Arms of one experiment run interleaved in **one** Slurm job on **one** GPU. Values are mean ± sample standard deviation across reps.

### Where the time goes at baseline (E1)

| stage | share of step | seconds per step | GPU busy during stage |
|---|---|---|---|
| rollout generation | **33.2 ± 0.4%** | 0.752 | 65.7% |
| scoring pass + advantages + loss + backward | **57.1 ± 0.5%** | 1.295 | — |
| ↳ backward (8 micro-batches) | | 0.653 | 90.5% |
| ↳ loss forward (8 micro-batches) | | 0.301 | 75.4% |
| ↳ old-log-prob correction pass | | 0.298 | 83.4% |
| weight sync (trainer → vLLM) | **7.8 ± 0.1%** | 0.178 | **2.0%** |
| optimizer step | 0.8% | 0.017 | 93.2% |
| reward (exact match, CPU) | < 0.1% | 0.001 | — |
| other (batch fetch, clipping, logging) | 1.1% | 0.024 | — |
| **whole step** | | **2.266 ± 0.014** | |

Rollout throughput was 11,383 ± 296 generated tokens/s. The trainer side outweighs generation because it runs full forward and backward passes over every prompt and completion token, while decode advances all 64 sequences one token per pass. The correction pass alone (13% of the step) exists only because the sampler and the trainer are separate engines.

GPU-busy fractions come from a separate profiled run (steps 20–21 under `torch.profiler`), so profiler overhead never touches the timing reps.

*Artifacts: [`results/E1_breakdown/`](results/E1_breakdown/) · job `13930286`.*

### Weight sync is idle time (E1, E4)

A sync pushes **338 tensors** (3.09 GB in bf16) into vLLM one `load_weights` call at a time. Each sync took about 180 ms, during which the GPU executed about 3.5 ms of kernels: it was idle **98%** of the time. The cost is fixed per sync, so it amortizes linearly with sync interval:

| sync every | step (s) | sync per step (s) | cost of one sync (s) | GPU busy during sync | correct, last 10 steps |
|---|---|---|---|---|---|
| 1 step | 2.287 ± 0.016 | 0.179 | 0.179 | 1.9% | 0.654 ± 0.029 |
| 4 steps | 2.159 ± 0.013 | 0.045 | 0.185 | — | 0.642 ± 0.039 |
| 16 steps | 2.113 ± 0.009 | 0.009 | 0.179 | 1.9% | 0.590 ± 0.075 |

Syncing every 16 steps saves 7.6% of step time, but the sampler then generates from weights up to 15 steps old, and learning looked slower. With 3 reps the final ranges overlap, so the staleness cost is reported as suggestive, not established.

**Measurement note:** NVML, the driver's utilization counter, reported **80%** GPU utilization during these same sync spans. It averages over up to a second, longer than the sync, so it was reporting the backward pass that ran just before. Sub-second stages need kernel-level traces.

*Artifacts: [`results/E4_sync_freq/`](results/E4_sync_freq/) · job `13930289` · tensor count: [`results/env/model_param_count.json`](results/env/model_param_count.json).*

### Rollout time follows the longest sample (E3)

| length cap | step (s) | rollout share | rollout (s) | tokens/s | mean length | longest | decode-slot occupancy |
|---|---|---|---|---|---|---|---|
| 128 | 1.632 ± 0.012 | 26.0% | 0.425 | 12,719 | 84 | 128 | 0.66 |
| 256 (E1) | 2.266 ± 0.014 | 33.2% | 0.752 | 11,383 | 134 | 255 | 0.53 |
| 512 | 3.386 ± 0.078 | 38.2% | 1.294 | 9,077 | 184 | 461 | **0.40** |

A generation round ends only when its longest sample finishes, so rollout time tracks the longest sample (2.8–3.3 ms per decode position at every cap), not the mean. As short answers finish the batch thins out: at a 512-token cap, only 40% of the batch's decode slots produced a token. Raising the cap made answers 2.2× longer on average but rollout 3.0× slower, and throughput fell 29%.

*Artifacts: [`results/E3_gen_length/`](results/E3_gen_length/) · job `13930288`. The 256 row is E1, a different job.*

### Bigger batches are cheap to generate (E2)

| samples per prompt | completions | step (s) | rollout (s) | trainer side (s) | tokens/s | rollout share | zero-signal groups |
|---|---|---|---|---|---|---|---|
| 4 | 32 | 1.517 ± 0.012 | 0.655 | 0.649 | 6,600 | 43.2% | 20.6% |
| 8 | 64 | 2.291 ± 0.029 | 0.765 | 1.304 | 11,179 | 33.4% | 10.2% |
| 16 | 128 | 3.958 ± 0.044 | 1.089 | 2.601 | 17,534 | 27.5% | 7.1% |

Quadrupling the batch cost only 1.66× the rollout time, consistent with memory-bound decode: each decode step reads the weights once, however many sequences it advances. The trainer side doubles with each doubling, so larger groups shift the bottleneck further toward training. Larger groups also waste fewer samples on *zero-signal groups*, where every sample got the same reward and the advantages are all zero.

*Artifacts: [`results/E2_group_size/`](results/E2_group_size/) · job `13930287`.*

### vLLM vs HF `generate` as the rollout backend (E5)

| backend | step (s) | rollout (s) | tokens/s | trainer side (s) | correction pass (s) | weight sync (s) |
|---|---|---|---|---|---|---|
| vLLM | 2.278 ± 0.018 | 0.761 | 11,244 | 1.292 | 0.293 | 0.182 |
| HF `generate` | 7.046 ± 0.148 | 6.013 | 1,422 | 0.990 | — | — |

With HF `generate` the trainer model *is* the sampler, so there is no sync and no correction pass. vLLM still wins by a wide margin: it adds 0.47 s per step and saves 5.25 s of generation. Training outcomes matched within noise. SGLang was out of scope: TRL has no SGLang backend ([`DECISIONS.md`](DECISIONS.md)).

*Artifacts: [`results/E5_backend/`](results/E5_backend/) · job `13931904`.*

### Held-out accuracy before and after training (E6)

On 200 problems from GSM8K's **test** split (never trained on), greedy decoding:

| model | strict (`#### n` format) | format-agnostic | format rate |
|---|---|---|---|
| base Qwen2.5-1.5B-Instruct | 23.0% | 66.0% | 33.5% |
| GRPO, 50 steps (3 seeds) | 66.8 ± 2.5% | 66.8 ± 2.5% | 97.5% |
| GRPO, 125 steps = 1 epoch (3 seeds) | **71.2 ± 2.6%** | **71.0 ± 2.3%** | 98.2% |

The strict gain is large and certain (paired McNemar p < 10⁻²¹ in every seed), but the format-agnostic score shows what it is: the base model could already solve most problems and rarely used the required answer format. Training taught the format first; better math came later and is modest (+5 points after an epoch, positive in 3 of 3 seeds, individually significant in 1). Splitting format from content is what keeps this number honest.

*Artifacts: [`results/E6_heldout/summary.json`](results/E6_heldout/summary.json) · jobs `13943559`, `13944626`, with every individual answer in each `eval.json`.*

### Reproducibility

The baseline configuration ran in three separate Slurm jobs and reproduced within 1.1%: 2.266 s (E1), 2.291 s (E2, G = 8), 2.278 s (E5) per step. Instrumentation overhead was measured directly (E0): 2.271 ± 0.010 s with full instrumentation vs 2.262 ± 0.012 s with step-boundary timing only, **+0.4%**. Seed-paired reps generated identical tokens in both arms, so the difference isolates the cost of measuring.

*Artifacts: [`results/E0_overhead/overhead.json`](results/E0_overhead/overhead.json) · job `13931400`.*

---

## Design

Three decisions worth arguing with:

**TRL over veRL, against the starting bias.** veRL is the more scalable framework and runs on one GPU, but it splits a step across four Ray-coordinated processes, so every stage timer would include Ray and serialization cost. It also syncs unconditionally every step, and its current release targets CUDA 13. TRL on one GPU is a single process with a short, readable sync path, and making the sync schedule configurable for E4 is a one-condition override. The full comparison, with pinned source links, is in [`DECISIONS.md`](DECISIONS.md).

**Stages are charged exclusive time, and `other` is the remainder.** TRL calls weight sync, generation, and rewards from *inside* its training step, so stages nest. Each stage is charged its time minus the stages opened inside it, like "self time" in a profiler, and `other` is defined as the step total minus the measured stages. Buckets therefore always sum exactly to wall clock, and a double count would show up as a negative remainder, which the timer refuses.

**Every stage boundary waits for the GPU.** A kernel launch returns before the kernel runs. Without a `torch.cuda.synchronize()` at each boundary, the backward pass's time is charged to whichever later stage first waits on the GPU. The synchronizations cost 0.4% (E0), and `tests/test_timing.py` demonstrates the misattribution on a simulated asynchronous GPU.

---

## Architecture

One training step on one H100, in one Python process (E1 baseline numbers):

```
on_step_end (previous step) ─────────────────────────────────────────── step window opens
 │
 ├─ weight_sync      trainer weights → vLLM, 338 tensors, one call each     0.178 s   GPU busy 2%
 ├─ rollout_gen      vLLM generates 8 prompts × 8 samples (≤256 tokens)     0.752 s   GPU busy 66%
 ├─ reward           exact match after "####" + format bonus (CPU)          0.001 s
 ├─ advantage_loss   old-log-prob pass → group-relative advantages →        1.295 s
 │                   8 micro-batches × (loss forward + backward)
 ├─ optimizer_step   AdamW                                                  0.017 s   GPU busy 93%
 └─ other            next batch, grad clipping, zero_grad, LR, logging      0.024 s
 │
on_step_end ─────────────────────────────────────────────────────────── 2.266 s total
```

```
rlstudy/
  timing.py         StepTimer: GPU-synchronized, exclusive-time buckets, span events
  instrumented.py   ProfiledGRPOTrainer: TRL GRPOTrainer with timing hooks + sync-every-k
  train.py          one run from a YAML config, with full environment capture
  trace.py          run.json (config, git SHA, Slurm job, GPU, driver, versions) + per-step JSONL
  gpumon.py         NVML sampler and torch.profiler busy-fraction analysis
  experiment.py     all arms × reps of one experiment: interleaved, seed-paired, resumable
  analyze.py        per-run summaries, mean/stdev across reps, per-experiment tables
  evaluate.py       held-out GSM8K accuracy: strict, format-agnostic, format rate
  heldout_summary.py  paired McNemar tests against the base model
  reward.py         exact-match and format rewards (Decimal comparison, no reward model)
  data.py           frozen, checksummed GSM8K subset (1,000 train / 200 eval)
  plots.py, figures.py, budget.py
configs/            baseline.yaml and one spec per experiment (E0–E5)
scripts/slurm/      PACE job scripts (experiments, held-out eval, env probe)
results/            COMMITTED artifacts: every published number resolves here
```

---

## Benchmark methodology

The parts that decide whether a number means anything:

- **Same GPU for every comparison.** All arms of one experiment run interleaved (rep 0 of every arm, then rep 1, …) in a single Slurm allocation on a non-preemptible queue with a pinned GPU type, so node drift spreads across arms instead of biasing one.
- **Seed-paired reps.** Rep *k* uses seed *k* in every arm. In E0 the paired arms generated identical tokens, which is how the overhead measurement isolates the cost of timing.
- **Warm-up and profiled steps are excluded.** The first 2 steps of every run contain CUDA graph capture and compilation; steps run under `torch.profiler` are flagged and dropped from timing.
- **Shares are ratios of sums**, so long steps weigh as much as they do in wall clock.
- **Kernel traces for short stages, not utilization counters** (see the NVML note under E4).
- **Inference is labelled.** The writeup separates what was measured from the mechanism inferred from it, and marks every unbuilt remedy as speculative.
- **Failures are kept.** Failed runs are moved aside, never overwritten, and logged with their diagnosis in [`PROGRESS.md`](PROGRESS.md).

Full methodology and per-experiment mechanisms: [`WRITEUP.md`](WRITEUP.md).

---

## Setup

Local (CPU or Apple Silicon: unit tests, a tiny-model smoke run, analysis, figures):

```bash
git clone https://github.com/ashwinvijayakumar24/grpo-rollout-profiling.git
cd grpo-rollout-profiling
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[train,analysis,dev]"

make test           # 104 fast tests, no model download
make smoke-local    # 3-step GRPO run with a tiny random model and HF generate
```

GPU (CUDA 12.9 driver ≥ 575; tested on H100 80GB): the pinned stack is torch 2.13.0+cu129, vLLM 0.30.0, TRL 1.14.2, transformers 5.17.0, accelerate 1.15.0, peft 0.21.2, datasets 5.0.1. [`scripts/setup_pace_env.sh`](scripts/setup_pace_env.sh) builds it on a cluster login node.

```bash
make baseline                       # instrumented baseline loop (E1 timing arm, 3 reps)
python -m rlstudy.experiment configs/experiments/E2_group_size.yaml
python -m rlstudy.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --out results/E6_heldout/base
make analyze figures                # tables and charts from results/
```

On Slurm: `sbatch scripts/slurm/run_experiment.sbatch configs/experiments/<spec>.yaml`. The eight result jobs (E0–E6) used 2.9 H100-hours in total (Slurm `sacct` elapsed time).

---

## Testing

**117 tests: 104 fast, 13 slow** (the slow ones train a tiny random model end to end). Everything that does not need a GPU is tested on a laptop, so cluster queue time never blocks development:

- the timer's attribution on a simulated asynchronous GPU, including a test showing the *wrong* answer without synchronization;
- an end-to-end GRPO run checking that buckets sum to the step total and every hooked stage records time;
- the E4 sync schedule with a fake vLLM sampler (syncs land on steps 0, k, 2k, and TRL's own per-step sync never fires);
- reward edge cases (commas, `$`, trailing periods, `72` vs `72.0`), data checksums, and the paired statistics.

Getting the GPU runs to work surfaced four failures, each now fixed and logged with its diagnosis in [`PROGRESS.md`](PROGRESS.md):

- **Out of memory, twice.** Pushing all 64 completions through one loss forward exhausted 80 GB. After micro-batching, the trainer alone peaked at 57 GB, and vLLM's default 30% reservation pushed the total over. Fixed with micro-batches of 8 and a 15% vLLM share. Sharing one GPU between a trainer and an inference engine is a real constraint of this design.
- **Port collision.** Two jobs landed on the same node, and both vLLM process groups bound the default rendezvous port 29500. Each run now claims a free port first.
- **Missing build tool.** FlashInfer JIT-compiles a kernel for greedy decoding with `ninja`, which was installed in the environment but not on the job's `PATH`.

---

## What I'd build next

- **Bulk weight sync.** Pack the 338 tensors into one buffer, or one batched `load_weights` call. The measurements bound the opportunity: about 3.5 ms of GPU work inside a 180 ms sync.
- **Slot refill for long-tail generation.** Feed new prompts into decode slots freed by finished samples, instead of waiting on the longest sample in a fixed batch (40% occupancy at a 512-token cap).
- **Matched sampler numerics,** which would remove the correction pass (13% of the step).
- **More seeds on held-out accuracy and staleness,** to settle the two effects reported here as suggestive.

Details and evidence for each: [`FUTURE.md`](FUTURE.md) and [`WRITEUP.md`](WRITEUP.md) §9.

## Going deeper

| Document | What it covers |
|---|---|
| [`WRITEUP.md`](WRITEUP.md) | Full report: methodology, every experiment with its mechanism, what I'd build differently |
| [`DECISIONS.md`](DECISIONS.md) | Framework comparison (veRL vs OpenRLHF vs TRL) and the pinned package versions |
| [`PROGRESS.md`](PROGRESS.md) | Chronological run log: every job, result, failure, and fix, with artifact paths |
| [`PLAN.md`](PLAN.md) | Original plan, scope rules, and GPU budget |
| [`FUTURE.md`](FUTURE.md) | Out-of-scope ideas and the next things to build |
| [`results/env/commit_map.txt`](results/env/commit_map.txt) | Maps the git SHAs recorded in result files to current commits (history was rewritten once) |

## License

MIT — see [LICENSE](LICENSE). Model weights are not included; Qwen2.5-1.5B-Instruct is released under Apache 2.0.
