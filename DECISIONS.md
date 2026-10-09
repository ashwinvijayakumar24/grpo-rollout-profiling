# Decisions

Status: Phase 0 decision, 2026-10-08. Waiting for Ashwin's sign-off before building
(PLAN.md checkpoint). Source links point at the commits listed in NOTES.md.

## Decision: TRL `GRPOTrainer` + vLLM in colocate mode

The task's default bias was veRL with vLLM rollouts, if veRL runs on one GPU. It
does run on one GPU (its own quickstart uses `trainer.n_gpus_per_node=1`,
[`V/docs/start/quickstart.rst#L123`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/docs/start/quickstart.rst#L123)).
I am still overriding the bias, for three concrete reasons found in the source:

1. **Install on PACE.** veRL 0.9.1's supported stack is built for CUDA 13
   (torch 2.11.0 cu130, vLLM 0.24.0 cu130, a cu130 flash-attn wheel), and its install
   docs push Docker. That needs an NVIDIA driver of at least 580. PACE's driver
   version is not yet known. TRL has published CUDA 12.9 wheels for every piece
   (details below), so it runs on drivers from 575 up.
2. **Clean timing attribution.** On one GPU, TRL runs in **one process**. The weight
   sync is a Python loop that calls `load_weights` for each parameter, in that same
   process. veRL uses four processes on one GPU: a Ray driver, an FSDP worker, a
   vLLM server actor and a vLLM engine worker. Its sync goes through ZMQ plus CUDA IPC
   buckets. Every veRL timer therefore includes Ray RPC (remote procedure call) and
   serialization costs, and those costs are not part of the systems question we are
   asking.
3. **The experiments need to change the sync schedule.** E4 asks what happens when
   we sync every 1, 4 or 16 steps. TRL decides whether to sync with a single `if`
   guard, which a small subclass override can change. veRL calls `update_weights`
   after every actor update with no condition. It also puts vLLM to sleep after
   every generation by default, and in full-weight mode that sleep throws the
   weights away. Getting E4 out of veRL would mean editing `RayPPOTrainer.fit` and
   the sleep/wake cycle. TRL also gives the `optimizer_step` bucket for free through
   the `on_pre_optimizer_step` / `on_optimizer_step` callbacks. veRL folds forward,
   backward and the optimizer step into one remote `update_actor` call.

What we give up: **SGLang as a stretch (E5) backend.** TRL supports vLLM, HF
`generate`, and transformers continuous batching. It has no SGLang path: a search of
`trl/` and `docs/` at v1.14.2 found no SGLang mention. veRL supports vLLM, SGLang
and TRT-LLM. For E5 under TRL, the arms are vLLM vs HF `generate` (both built in).
SGLang would need a custom generation class, so it goes in FUTURE.md unless E1–E4
finish early. This trade is acceptable because E5 is a stretch goal and E1–E4 are not.

## Comparison

### (a) Single-GPU friendliness

| | Runs on 1 GPU? | Ray required? | Processes on 1 GPU |
|---|---|---|---|
| TRL 1.14.2 | Yes. Colocate mode was built for this case | No | 1 |
| veRL 0.9.1 | Yes (quickstart uses 1 GPU, 24 GB+) | Yes, always `ray.init` | 4 (driver, FSDP worker, vLLM server actor, vLLM `mp` worker) |
| OpenRLHF 0.11.2 | Probably, with `--train.colocate_all` **(uncertain: I found no 1-GPU example)** | Yes (`ray[default]==2.55.0` is a hard dependency) | several Ray actors |

Sources: TRL colocate `LLM(..., distributed_executor_backend="external_launcher")`
[`T/trl/generation/vllm_generation.py#L347-L362`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L347-L362);
veRL `ray.init` [`V/verl/trainer/main_ppo.py#L74`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/main_ppo.py#L74)
and vLLM server [`V/.../vllm_async_server.py#L329`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L329);
OpenRLHF deps from PyPI metadata of `openrlhf==0.11.2`.

### (b) Rollout backend: pluggable and instrumentable?

| | vLLM | SGLang | HF generate | Hooks for timing |
|---|---|---|---|---|
| TRL | yes (colocate or server) | no | yes (`use_vllm=False`) | override `_generate_single_turn`; HF callbacks |
| veRL | yes | yes | class exists, not found in the main loop **(uncertain)** | `marked_timer` on driver; registry `RolloutReplicaRegistry` |
| OpenRLHF | yes | no (none found in source) | no | edit Ray actor code |

Sources: TRL dispatch [`T/.../grpo_trainer.py#L1858-L1945`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/trainer/grpo_trainer.py#L1858);
veRL registry [`V/verl/workers/rollout/replica.py#L387-L389`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/replica.py#L387-L389).

veRL is the better choice for swapping backends. TRL is the better choice for timing
them, because each one is a direct, blocking call in the same process.

### (c) Readability of the weight-sync path

- **TRL:** about 80 lines in one file:
  [`VLLMGeneration._iter_named_params` and `sync_weights`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L435-L517).
  It iterates the parameters, merges LoRA if present, calls `load_weights`, and
  resets the prefix cache. Best of the three.
- **veRL:** spread across five files. The chain is `CheckpointEngineManager.update_weights`
  → `ActorRolloutRefWorker.update_weights` → `ServerAdapter.update_weights` →
  `BucketedWeightSender`/`Receiver` (ZMQ + CUDA IPC) → vLLM worker extension
  `update_weights_from_ipc` → `load_weights`. On top of that sit sleep/wake memory
  tags, QAT/FP8 branches, and LoRA base-vs-adapter state. The code is readable, but
  there is a lot of it. (Links in NOTES.md section 4.)
- **OpenRLHF:** one function, `broadcast_to_vllm`, but it makes one Ray remote call
  per tensor
  ([`O/openrlhf/trainer/ray/ppo_actor.py#L409`](https://github.com/OpenRLHF/OpenRLHF/blob/17fd4a79cc87dfb2f6185ffac634066525bcd917/openrlhf/trainer/ray/ppo_actor.py#L409)).

### LoRA + vLLM in the sync path

- **TRL:** supported. The adapter is merged into the base weights, the **full** base
  weights are pushed, and the adapter is unmerged again
  ([`#L446-L469`](https://github.com/huggingface/trl/blob/a01dc41fb1f09d56d8709c3ea2a2868f8c035560/trl/generation/vllm_generation.py#L446-L469)).
  So LoRA does not shrink the sync. With LoRA, TRL uses the adapter-off model as the
  reference model, so no second copy is loaded.
- **veRL:** supported, and smarter. The base weights are sent once, and after that
  only the adapter goes over, through vLLM `add_lora` with a `TensorLoRARequest`
  ([`V/.../vllm_rollout/utils.py#L367-L384`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/utils.py#L367-L384)).
  There is also a merge option (`model.lora.merge`).
- **OpenRLHF:** LoRA training exists (`--ds.lora.rank`). The vLLM sync loop has no
  merge step, so I could not confirm that LoRA + vLLM sync is correct **(uncertain)**.

Possible later finding: "LoRA adapter-only sync vs full merged sync" is a clean E4
sub-experiment that only veRL can run. It goes in FUTURE.md.

### Install pain on an offline-compute cluster with the CUDA 12.9 module

A key fact first: **pip wheels ship their own CUDA runtime**. Running
`module load cuda/12.9.1` does not decide which wheels work. The **NVIDIA driver** on
the compute node decides that. The CUDA 12.9 module only matters for code compiled
on the fly (flashinfer JIT kernels, a flash-attn source build). According to NVIDIA's
release notes, CUDA 12.9 needs driver ≥ 575.51.03 and CUDA 13.x needs driver ≥ 580
([CUDA toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
**PACE's driver version is unknown.** The first action of Phase 2 is a 1-minute
`embers` job that runs `nvidia-smi --query-gpu=name,driver_version --format=csv`.

- Default PyPI `torch` wheels for 2.11 and 2.13 depend on `cuda-toolkit==13.0.x`
  (PyPI metadata of `torch==2.11.0` and `torch==2.13.0`). So a plain
  `pip install vllm` now pulls a CUDA 13 stack.
- **TRL path (chosen):** everything has a cu129 build. These were checked:
  `torch-2.13.0+cu129`, `torchvision-0.28.0+cu129` and `torchaudio-2.11.0+cu129` on
  https://download.pytorch.org/whl/cu129, and `vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`
  as a GitHub release asset of [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0).
  TRL itself is pure Python. The install has no flash-attn build step, because vLLM
  brings its own attention kernels.
- **veRL path:** the `vllm` extra pins `torch==2.11.0`, `vllm==0.24.0` and
  `transformers==5.9.0`. Its pyproject says "All backends use torch 2.11.0:
  vllm/sglang/fsdp/megatron on cu130", and the `fsdp` extra pulls the cu130 flash-attn
  wheel ([`V/pyproject.toml#L19`, `#L78`, `#L90-L95`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/pyproject.toml#L19)).
  The docs recommend Docker or uv with a lockfile
  ([`V/docs/start/install.rst`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/docs/start/install.rst)).
  PACE offers conda, not Docker. A cu129 veRL environment is possible by hand
  (`vllm-0.24.0+cu129` and `torch-2.11.0+cu129` wheels both exist), but that is
  off the tested path.
- **OpenRLHF path:** hard-pins `deepspeed==0.19.6`, `flash-attn==2.8.3`,
  `ray[default]==2.55.0`, `transformers==5.15.0`, and `vllm==0.29.0` for the `vllm`
  extra (PyPI metadata of `openrlhf==0.11.2`). flash-attn 2.8.3 without a matching
  prebuilt wheel means a long source build with nvcc.
- **Offline compute nodes (all three):** do `pip download` / `pip install` into the
  `grpo` env on the login node, pre-download model and dataset into
  `~/ps-simpliearn-0`, and run jobs with `HF_HUB_OFFLINE=1` (as PLAN.md says).
  vLLM compiles kernels and captures CUDA graphs at the first run. Its caches go to
  `~/.cache/vllm` and the flashinfer JIT cache, both of which must be writable on
  the compute node **(uncertain: exact cache paths for vLLM 0.30 not checked)**.

## Recommended version pins (TRL path)

These pins are mutually compatible according to each package's published
requirements:

| Package | Pin | Why this version | Source |
|---|---|---|---|
| python | 3.12 | TRL 1.14.2 needs ≥3.10; vLLM needs <3.15 | PyPI metadata |
| trl | **1.14.2** | Released 2026-10-06. Requires `vllm>=0.20.0,<=0.30.0`. I skipped 1.15.0 (released today, 2026-10-08) because it changes 314 lines of `grpo_trainer.py` and has had no time to settle | PyPI; `git diff --stat v1.14.2 v1.15.0` |
| vllm | **0.30.0** (`+cu129` wheel) | Newest version inside TRL 1.14.2's range; released 2026-09-22; has a cu129 asset | [release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) |
| torch | **2.13.0+cu129** | vLLM 0.30.0 requires `torch==2.13.0` | PyPI metadata of vllm 0.30.0 |
| torchvision | **0.28.0+cu129** | vLLM 0.30.0 requires `torchvision==0.28.0` | same |
| torchaudio | **2.11.0+cu129** | vLLM 0.30.0 requires `torchaudio==2.11.0` | same |
| flashinfer-python | **0.6.18.post1** | exact pin pulled by vLLM 0.30.0 | same |
| transformers | **5.17.0** | vLLM 0.30.0 needs `>=5.10.4`; TRL needs `>=4.56.2`; released 2026-09-09 | PyPI |
| accelerate | **1.15.0** | TRL needs `>=1.4.0`; released 2026-09-09 | PyPI |
| peft | **0.21.2** | TRL `peft` extra needs `>=0.13.0`; needed only for the LoRA arm | PyPI |
| datasets | **5.0.1** | TRL needs `>=4.7.0` | PyPI |

Install sketch (on the login node, into the `grpo` conda env):

```
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu129
pip install https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
pip install trl==1.14.2 transformers==5.17.0 accelerate==1.15.0 peft==0.21.2 datasets==5.0.1
pip check   # then freeze to results/env.json via the run script
```

Caveats on the pins:

- The asset URL above follows the asset name listed on the GitHub release; I did
  not download it **(uncertain: confirm the URL resolves)**.
- I have not tested whether installing vLLM after the cu129 torch keeps the cu129
  torch rather than pulling a cu130 one from PyPI **(uncertain)**. If pip tries to
  replace it, add `--extra-index-url https://download.pytorch.org/whl/cu129` to the
  vLLM install, or use `uv pip install ... --torch-backend cu129`. Check with
  `python -c "import torch; print(torch.version.cuda)"`.
- If the driver turns out to be ≥ 580, the plain PyPI wheels (cu130) work as well,
  and the cu129 steps become optional.

**Stretch (E5) SGLang, separate env only:** `sglang==0.5.20` (2026-09-18) requires
`torch==2.13.0`, `transformers==5.12.1`, `cuda-python>=13.0` and
`flashinfer_python[cu13]==0.6.18`, so it needs a CUDA 13 driver (≥ 580) and its own
conda env (PyPI metadata of `sglang==0.5.20`). It cannot share the `grpo` env because
its transformers pin is exact.

## Risks

1. **Driver too old for CUDA 13, or too old even for 12.9.** If the driver is below
   575, none of the pins above work, and every recent vLLM is ruled out. Mitigation:
   run the `nvidia-smi` check before building anything. A login-node check
   (2026-10-08) could not read the driver, because login nodes have no GPU driver
   loaded. `sinfo` there does list `h100`, `h200`, `a100`, `l40s` and
   `rtx_pro_6000_blackwell` GPU nodes. Blackwell support suggests a recent driver on
   at least some nodes, but that is a guess, and nodes may differ. Record the driver
   for the exact GPU type we pin.
2. **Memory on one GPU for a full fine-tune.** The trainer (about 24 GB of weights
   and AdamW state for 1.5B, an estimate) plus vLLM's 0.3 share has to fit. It should
   fit on an 80 GB H100. On a 40 GB A100, use LoRA or Llama-3.2-1B. Sleep mode would
   save memory, but it forces a sync every generation and breaks E4 (NOTES.md §2).
3. **Changing the sync schedule changes the algorithm.** Syncing every k > 1 steps
   means vLLM samples from stale weights. TRL's importance-sampling correction (on by
   default) partly compensates for this. E4 must report reward next to time, so that
   a faster schedule is not mistaken for a free win.
4. **TRL internals move fast.** `_generate_single_turn` is a private method, and
   1.15.0 already changed `grpo_trainer.py` heavily. Pin exactly, and keep our
   override small and in one file.

## Update 2026-10-09: risk 1 resolved

Probe job 13908530 on `inferno`/`gpu-h100` measured an **NVIDIA H100 80GB HBM3 with
driver 615.71.09** (`results/env/probe-13908530.txt`). That is above both the cu129
minimum (575.51.03) and the CUDA 13 minimum (580), so the pin set above works. The
`grpo` env's torch 2.13.0+cu129 sees the GPU and runs a matmul. With 80 GB, the full
fine-tune of the 1.5B model is the baseline; LoRA is not needed for memory.
