# Phase 0 Reading Notes

Status: reading done 2026-10-08. No code run yet. Everything below comes from reading
source at the pinned commits linked in each section. Claims I could not confirm from
source are marked **(uncertain)**.

Source versions read:

| Project | Version | Commit | Base URL used in links |
|---|---|---|---|
| TRL | v1.14.2 | `a01dc41` | `T` = https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560 |
| veRL | v0.9.1 | `1876b06` | `V` = https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b |
| OpenRLHF | v0.11.2 | `17fd4a7` | `O` = https://github.com/OpenRLHF/OpenRLHF/blob/17fd4a79cc87dfb2f6185ffac634066525bcd917 |
| transformers | v5.17.0 | tag | `HF` = https://github.com/huggingface/transformers/blob/v5.17.0 |

The framework chosen in DECISIONS.md is **TRL `GRPOTrainer` with vLLM in colocate
mode**, so sections 2 and 3 describe TRL in detail. veRL is described where the
contrast matters.

---

## 1. GRPO as data flow

GRPO (Group Relative Policy Optimization) replaces PPO's learned value model (a
second network that predicts how good a state is) with a simple statistic: for each
prompt, sample G answers, score them, and use each answer's score relative to its
group's mean as its "advantage" (how much better than average it was). The loss then
raises the probability of tokens in above-average answers and lowers it for
below-average ones.

Notation: **B** = prompts per generation round, **G** = samples per prompt
(`num_generations`, default 8), **N = B·G** completions, **P** = padded prompt length,
**T** = padded completion length (at most `max_completion_length`, default 512).

One caution about TRL's batch arguments: `per_device_train_batch_size` and
`generation_batch_size` count **completions**, not prompts. The generation batch must
be divisible by `num_generations` so that every group is complete
([`T/trl/trainer/grpo_config.py#L1116-L1120`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_config.py#L1116-L1120)).
So B = generation_batch_size / G.

### Stage by stage (TRL colocate, single GPU, single process)

On one GPU in colocate mode there is exactly **one Python process**. The trainer
model and the vLLM engine both live in it and share the one GPU. "Where it lives"
below therefore means CPU memory vs GPU memory, not which process.

| # | Tensor | Shape | Lives on | Produced by |
|---|---|---|---|---|
| 1 | Prompt token ids | B × P (list of lists, unpadded) | CPU | tokenizer in `_generate_and_score_completions` |
| 2 | Completion token ids | N × T (Python lists from vLLM, padded later to a tensor) | CPU, then GPU | `vllm_generation.generate` |
| 3 | Sampling log-probs (from vLLM) | N × T | CPU, then GPU | vLLM `logprobs` output, top-1 kept |
| 4 | Decoded completion text | N strings | CPU | tokenizer decode |
| 5 | Rewards | N × (number of reward funcs), then N after weighting | CPU float list → GPU tensor | `_calculate_rewards` |
| 6 | Group-normalized advantages | N (one scalar per completion, broadcast over its tokens) | GPU | `(r - mean_group) / (std_group + 1e-4)` |
| 7 | "Old" per-token log-probs | N × T | GPU | extra no-grad forward of the trainer model |
| 8 | Reference log-probs | N × T | GPU | only if `beta > 0` |
| 9 | New per-token log-probs | N × T, with grad | GPU | forward pass inside `compute_loss` |
| 10 | Loss | scalar | GPU | clipped ratio × advantage, token-averaged |

Notes on each stage:

- **Rewards (5).** For GSM8K exact match, the reward function is pure Python on
  strings. It runs on CPU and is cheap. TRL wraps each reward function in its own
  timer ([`T/trl/trainer/grpo_trainer.py#L1674`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L1674)).
- **Advantages (6).** Reshape rewards to B × G, take the mean (and standard deviation,
  because `scale_rewards="group"` is the default) along G, subtract and divide
  ([`T/.../grpo_trainer.py#L2831`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L2831)).
  This is a few kernels on an N-element tensor. Its cost is negligible; what matters
  is that a group where all G answers got the same reward has advantage 0 everywhere
  and contributes no gradient.
- **Old log-probs (7).** This is a full forward pass of the trainer model over
  N × (P+T) tokens, without gradients. TRL computes it whenever vLLM is used, because
  `vllm_importance_sampling_correction` defaults to `True`
  ([`T/trl/trainer/grpo_config.py#L920`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_config.py#L920),
  condition at [`T/.../grpo_trainer.py#L2704-L2719`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L2704-L2719)).
  Why it exists: vLLM's numerics (different kernels, possibly different precision)
  give slightly different probabilities from the training model. TRL compares the
  two (tensor 3 vs tensor 7) and reweights the loss to correct for the mismatch. For
  this study it is a real, separately measurable cost of using a separate inference
  engine. It does not exist with HF `generate`, where the sampler *is* the trainer
  model. That asymmetry matters for experiment E5.
- **Reference model and KL (8).** A reference model is a frozen copy of the starting
  model. A KL term (a penalty for drifting away from the reference's probabilities)
  needs its log-probs. **TRL's default is `beta=0.0`, so there is no reference model
  and no KL term**
  ([`T/trl/trainer/grpo_config.py#L676`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_config.py#L676),
  [`T/.../grpo_trainer.py#L979-L985`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L979-L985)).
  If `beta > 0` and the model is full fine-tuned, TRL loads a second full copy of the
  model (about 3 GB in bf16 for a 1.5B model) and runs one more no-grad forward per
  generation round. If `beta > 0` with LoRA, TRL uses the same model with the adapter
  switched off, so it costs a forward pass but no extra memory. veRL has the same
  default: `use_kl_loss: false` and `use_kl_in_reward: false`, so no reference policy
  unless enabled
  ([`V/verl/trainer/ppo/utils.py#L75-L79`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/utils.py#L75-L79)).
  With LoRA, veRL also reuses the actor without the adapter as the reference
  (`ref_in_actor`, [`V/verl/trainer/ppo/ray_trainer.py#L360`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py#L360)).
  **Plan: keep `beta=0` for all experiments**, and say so in the writeup.
- **Loss (9, 10).** The default `loss_type` is `"dapo"`: token losses are summed and
  divided by the number of real (non-padding) completion tokens in the accumulated
  batch ([`T/trl/trainer/grpo_config.py#L236-L248`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_config.py#L236-L248)).
  The forward with gradients plus backward is the largest training-side cost and
  scales with N × (P+T).

### Memory on one GPU (planning estimate, not a measurement)

Two things share the GPU: the training model with its optimizer state, and vLLM
(its own copy of the weights plus a KV cache, which is the per-token attention memory
vLLM reserves for in-flight sequences). vLLM takes a fixed fraction of GPU memory set
by `vllm_gpu_memory_utilization` (TRL default 0.3,
[`T/trl/trainer/grpo_config.py#L651`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_config.py#L651)).
A full fine-tune of a 1.5B model with AdamW in fp32 needs roughly 16 bytes per
parameter, about 24 GB, plus activations. That should fit next to a 0.3 vLLM share on
an 80 GB H100. On a 40 GB A100 it probably does not, which is where LoRA comes in.
These are estimates to be checked in Phase 2.

---

## 2. TRL GRPOTrainer architecture and the weight-sync path

### Architecture in one paragraph

`GRPOTrainer` is a subclass of the Hugging Face `Trainer`. The `Trainer` owns the
outer loop: get a batch, call `training_step`, and every `gradient_accumulation_steps`
micro-batches clip gradients and call `optimizer.step()`. GRPO hooks in by overriding
`_prepare_inputs`, which `training_step` calls first. `_prepare_inputs` generates and
scores a whole generation batch once every `steps_per_generation × num_iterations`
micro-steps, caches it in `_buffered_inputs`, and hands out one slice per micro-step
([`T/.../grpo_trainer.py#L1617-L1641`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L1617-L1641)).
The vLLM engine is wrapped in a helper class, `VLLMGeneration`, in
[`T/trl/generation/vllm_generation.py`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py).

Call chain for one optimizer step (default settings, one GPU):

```
Trainer._inner_training_loop                      (transformers trainer.py)
  for each micro-batch:
    GRPOTrainer.training_step                     grpo_trainer.py#L1605
      Trainer.training_step
        GRPOTrainer._prepare_inputs               #L1617   (generates only on round boundaries)
          _generate_and_score_completions         #L2387
            _generate -> _generate_single_turn    #L2261 / #L1858
              VLLMGeneration.sync_weights()       vllm_generation.py#L481  (only if global_step changed)
              VLLMGeneration.generate()           vllm_generation.py#L550 -> self.llm.generate  #L729
            _calculate_rewards                    #L1674
            old log-probs forward (no grad)       #L2704-L2719
            group-normalized advantages           #L2831
        compute_loss -> _compute_loss             #L3005 / #L3091  (forward with grad)
        accelerator.backward(loss)
  clip_grad_norm
  on_pre_optimizer_step callback                  trainer.py#L1867
  optimizer.step()                                trainer.py#L1868
  on_optimizer_step callback                      trainer.py#L1869
  lr_scheduler.step, zero_grad, global_step += 1
```

Sources: [`HF/src/transformers/trainer.py#L1825-L1882`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/trainer.py#L1825-L1882),
[`T/.../grpo_trainer.py#L1605`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L1605).

### How the vLLM engine is created (colocate mode)

`VLLMGeneration._init_vllm` builds a plain `vllm.LLM` object inside the trainer
process ([`T/trl/generation/vllm_generation.py#L347-L362`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L347-L362)),
with these settings that matter for us:

- `distributed_executor_backend="external_launcher"`: vLLM does not spawn its own
  worker processes; it reuses the ranks the trainer was launched with. The sync code
  reaches straight into `llm.llm_engine.model_executor.driver_worker.model_runner.model`,
  which only works because the vLLM model object sits in the same process.
- `gpu_memory_utilization` from config (default 0.3), `max_num_batched_tokens=4096`.
- `enable_sleep_mode` from config (default `False`). Sleep mode lets vLLM give its GPU
  memory back while the trainer runs. If it is on, TRL immediately calls
  `llm.sleep(level=2)` (level 2 discards the weights entirely, not just the KV cache).

`vllm_mode` defaults to `"colocate"` but `use_vllm` defaults to `False`, so we must set
`use_vllm=True` explicitly
([`T/trl/trainer/grpo_config.py#L582-L607`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_config.py#L582-L607)).

### What the weight sync physically does

`VLLMGeneration.sync_weights`
([`#L481-L517`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L481-L517)):

1. If sleep mode is on: `empty_cache()`, then `llm.wake_up(tags=["weights"])` to
   re-map vLLM's weight memory.
2. Iterate the trainer's parameters one at a time with `_iter_named_params`
   ([`#L435-L479`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L435-L479)).
   On one GPU with no FSDP or DeepSpeed, this is just `model.named_parameters()`,
   with names cleaned up to match vLLM's naming.
3. For each `(name, tensor)`, call
   `driver_worker.model_runner.model.load_weights([(name, param)])`
   ([`#L511`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L511)).
   vLLM's `load_weights` finds the matching vLLM parameter and copies into it. For
   layers that vLLM fuses (for example q, k, v projections stored as one `qkv_proj`
   matrix), the model's weight loader copies the incoming tensor into the right slice.
   Physically this is **one device-to-device copy per parameter on the same GPU**: no
   IPC (inter-process communication), no NCCL (NVIDIA's collective communication
   library), no serialization. There are a few hundred parameters for a 1.5B model,
   so it is a few hundred small Python calls plus copies.
4. `llm.reset_prefix_cache()` so cached KV entries computed with old weights are
   not reused.

**LoRA path.** TRL never hands vLLM a LoRA adapter. When the model is a PEFT model,
`_iter_named_params` calls `model.merge_adapter()` (adds B·A into the base weights in
place), yields the merged base weights under their original names while skipping the
`lora_` tensors, then calls `model.unmerge_adapter()` (subtracts it back)
([`#L446-L469`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L446-L469)).
Consequence: with LoRA, sync still copies **all** base weights every time (not just
the small adapter), plus the merge and unmerge kernels. Sync cost is therefore about
the same with or without LoRA. That is a finding worth stating in the writeup, because
the intuition "LoRA makes sync cheap" is false for this path. veRL, by contrast, can
ship only the adapter to vLLM through `add_lora` (see section 4).

**When sync runs.** Sync is lazy. It is not called after the optimizer step. It is
called at the start of the next generation, guarded by
`if self.state.global_step != self._last_loaded_step`
([`T/.../grpo_trainer.py#L1862-L1867`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L1862-L1867)).
So in a timeline, `weight_sync` sits just before `rollout_gen`, inside
`_prepare_inputs`.

**Sleep mode interacts with sync.** If sleep mode is on, `generate()` itself re-pushes
the weights whenever they are sleeping, because level-2 sleep threw them away
([`#L586-L590`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L586-L590)),
and puts vLLM back to sleep after every generate call
([`#L750-L752`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L750-L752)).
**Experiment E4 (sync every 1, 4, 16 steps) is therefore only meaningful with sleep
mode off.** With sleep mode on, every generation forces a full sync regardless of the
schedule. E4 should be run with `vllm_enable_sleep_mode=False` and the guard
changed to "sync when `global_step - _last_loaded_step >= k`". That change is a
subclass override of `_generate_single_turn`, a few lines.

---

## 3. Instrumentation hook points for the six buckets

The study's buckets are `rollout_gen`, `reward`, `advantage_loss`, `optimizer_step`,
`weight_sync`, and `other` (total minus the sum).

### Existing timers in TRL, and why we cannot use them as-is

TRL already wraps sync, generation, and each reward function in `profiling_context`
([`T/trl/extras/profiling.py#L35-L105`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/extras/profiling.py#L35-L105)).
It uses `time.perf_counter()` with **no `torch.cuda.synchronize()`**. That is fine for
vLLM's `generate` (which returns finished Python lists, so the GPU work is done) and
for CPU reward functions. It is wrong for anything that queues GPU kernels and
returns immediately. We will use TRL's timer names as a cross-check, not as the
source of truth.

### Hook points

| Bucket | Start | End | How to hook |
|---|---|---|---|
| `weight_sync` | before `self.vllm_generation.sync_weights()` | after it | override `_generate_single_turn` (also where the E4 every-k guard goes); sync before stop |
| `rollout_gen` | before `vllm_generation.generate(...)` | after it returns | same override; or wrap `VLLMGeneration.generate` |
| `reward` | entry of `_calculate_rewards` | exit | wrap the method (CPU work; a sync at entry flushes earlier GPU work into the previous bucket) |
| `advantage_loss` | after reward returns | end of last `training_step` before the optimizer | covers advantage math, old-log-prob forward, loss forward, backward. Sub-bucket `old_logprob` by wrapping `_get_per_token_logps_and_entropies` when called under `no_grad` |
| `optimizer_step` | `on_pre_optimizer_step` callback | `on_optimizer_step` callback | `TrainerCallback`, sync in both ([`HF/trainer.py#L1867-L1869`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/trainer.py#L1867-L1869)) |
| `other` | `on_step_begin` | `on_step_end` | total minus the five above: data loading, tokenizing, padding, gradient clipping, logging, scheduler |

Gradient clipping runs just before `on_pre_optimizer_step`
([`HF/trainer.py#L1862-L1865`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/trainer.py#L1862-L1865)),
so by the table above it lands in `other`. We could also move it into
`optimizer_step` by starting that timer at the end of the last `training_step`
instead. Either choice is fine as long as it is written down.

### Pitfalls

1. **Asynchronous CUDA work charged to the wrong bucket.** PyTorch returns from a
   kernel launch before the kernel finishes. If we read the clock without
   synchronizing, the backward pass's GPU time shows up in whatever bucket next
   forces a sync (often `optimizer_step` or `other`, through a `.item()` in logging).
   Rule: call `torch.cuda.synchronize()` immediately before reading the clock at every
   bucket boundary. CUDA events (`torch.cuda.Event(enable_timing=True)`) are an
   alternative that avoids a full stall, but they only measure GPU time on one stream,
   and the vLLM bucket is mostly CPU scheduling plus GPU work, so wall clock with
   synchronize is the simpler, correct choice here. The extra synchronizations cost
   a little throughput, which Phase 2 measures (timing on vs off).
2. **Generation and training are strictly sequential on one GPU.** In colocate mode the
   trainer process calls `llm.generate()` and blocks until it returns. There is no
   overlap between rollout and training, so the buckets do not overlap and the "sum
   to total" property holds. (veRL's colocated mode is also sequential: the driver
   blocks on generation, then puts vLLM to sleep, then trains.)
3. **Gradient accumulation splits one step across several `training_step` calls.**
   Generation happens on the first micro-step of a round only; the other micro-steps
   reuse `_buffered_inputs`. Our per-step record must sum the `advantage_loss` time
   across all micro-steps of the optimizer step. Simplest baseline: set
   `gradient_accumulation_steps=1` and `steps_per_generation=1`, so one optimizer
   step = one generation round = one `training_step`.
4. **vLLM's own timing.** vLLM reports per-request metrics (time to first token,
   queueing, decode time) through `RequestOutput.metrics` when stats logging is on
   **(uncertain: exact field names in vLLM 0.30 not checked)**. Those are
   engine-internal times and exclude Python overhead around `generate`. Use them only
   to split `rollout_gen` into prefill vs decode, never as the bucket total.
5. **First-step outliers.** Step 0 includes CUDA graph capture, `torch.compile`
   warm-up inside vLLM, and memory allocator growth. Drop the first 1 to 2 steps from
   every summary, and record that rule in the analysis code.
6. **`empty_cache` hides inside sync.** With sleep mode on, `sync_weights` calls
   `empty_cache()` and `wake_up`. Those costs belong to `weight_sync` but are memory
   management, not data movement. For E4, log them as a sub-timer so we can say how
   much of "sync" is copying.
7. **Reward function placement.** If any reward code touches GPU tensors (it should
   not for exact match), it needs the same synchronize treatment.

---

## 4. Contrast: veRL's wiring on one GPU (for DECISIONS.md)

veRL does run on one GPU, but even then it is a multi-process Ray system. Ray is a
distributed task scheduler: the training loop runs in a "driver" process and calls
remote methods on "actor" processes.

- **Processes.** `main_ppo` always calls `ray.init`
  ([`V/verl/trainer/main_ppo.py#L74`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/main_ppo.py#L74)).
  The loop runs in a driver task. The FSDP training engine runs in a Ray worker
  actor. vLLM runs as a separate `vLLMHttpServer` Ray actor that creates an
  `AsyncLLM` with `distributed_executor_backend="mp"`, which starts yet another
  vLLM worker process
  ([`V/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L87`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L87),
  [`#L329-L330`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L329-L330),
  [`#L494`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L494)).
- **Built-in timers.** `RayPPOTrainer.fit` already times `gen`, `reward`,
  `old_log_prob`, `ref`, `adv`, `update_actor`, `update_weights` with `marked_timer`
  ([`V/verl/trainer/ppo/ray_trainer.py#L1510-L1716`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py#L1510-L1716)).
  These are wall-clock timers on the driver around blocking Ray calls, so they
  include RPC and serialization time. `update_actor` bundles forward, backward and
  optimizer step inside the worker; splitting out `optimizer_step` needs an edit to
  the worker code. `gen` also includes `sleep_replicas()`
  ([`#L1514`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py#L1514)).
- **Weight sync (default `checkpoint_engine.backend: naive`,**
  [`V/verl/trainer/config/rollout/rollout.yaml#L282`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/config/rollout/rollout.yaml#L282)**).**
  `CheckpointEngineManager.update_weights` calls the actor's `update_weights`
  ([`V/verl/checkpoint_engine/base.py#L510-L520`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/checkpoint_engine/base.py#L510-L520)).
  The actor ([`V/verl/workers/engine_workers.py#L728-L825`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/engine_workers.py#L728)):
  wakes vLLM's weight memory (`resume(tags=["weights"])`), gets a generator of full
  tensors from FSDP (`get_per_tensor_param`), and streams them to vLLM through
  `BucketedWeightSender`: tensors are packed into buckets (default 2048 MB), each
  bucket is shared with the vLLM process as a **CUDA IPC handle** over a **ZMQ** socket
  ([`V/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L210-L253`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L210-L253),
  [`bucketed_weight_transfer.py#L73`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py#L73)).
  On the vLLM side a worker extension rebuilds the tensors from the handles and calls
  `model.load_weights` per bucket, then `process_weights_after_loading`
  ([`V/verl/workers/rollout/vllm_rollout/utils.py#L248-L345`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/utils.py#L248-L345)).
  Then the actor offloads, and wakes vLLM's KV cache.
- **LoRA in veRL.** Without `lora.merge`, veRL syncs the base weights once, then on
  every later step removes the old adapter and calls vLLM's `add_lora` with a
  `TensorLoRARequest` built from the new adapter tensors
  ([`utils.py#L367-L384`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/utils.py#L367-L384)).
  So veRL really does ship only the adapter, which TRL does not.
- **Sleep is the default and is per step.** `free_cache_engine: True` and vLLM is put
  to sleep after every generation
  ([`rollout.yaml#L61`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/config/rollout/rollout.yaml#L61),
  [`vllm_async_server.py#L853-L862`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L853-L862)).
  `update_weights` is called unconditionally after every `update_actor`
  ([`ray_trainer.py#L1715-L1716`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py#L1715-L1716)).
  Syncing every k steps would require editing `fit` and disabling the sleep cycle.
- **Rollout backends registered** for the main trainer: `vllm`, `sglang`, `trtllm`
  ([`V/verl/workers/rollout/replica.py#L387-L389`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/replica.py#L387-L389)).
  An `HFRollout` class exists but I did not find it wired into the current
  `RayPPOTrainer` loop **(uncertain)**.

### OpenRLHF, briefly

Also Ray-based. In colocated mode its sync is `broadcast_to_vllm`: for **each
parameter**, clone it, make a CUDA IPC handle, `all_gather_object` the handles, and
issue a Ray remote call `update_weight_cuda_ipc` to each vLLM engine, which calls
`load_weights` for that one tensor
([`O/openrlhf/trainer/ray/ppo_actor.py#L409-L490`](https://github.com/OpenRLHF/OpenRLHF/blob/17fd4a79cc87dfb2f6185ffac634066525bcd917/openrlhf/trainer/ray/ppo_actor.py#L409),
[`O/openrlhf/trainer/ray/vllm_worker_wrap.py#L52-L69`](https://github.com/OpenRLHF/OpenRLHF/blob/17fd4a79cc87dfb2f6185ffac634066525bcd917/openrlhf/trainer/ray/vllm_worker_wrap.py#L52)).
That is one Ray round trip per tensor. The loop iterates `model.named_parameters()`
with no LoRA merge step, so I could not confirm that LoRA training syncs correctly
into vLLM **(uncertain)**.
