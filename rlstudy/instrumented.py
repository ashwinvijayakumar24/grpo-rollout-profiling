"""TRL GRPOTrainer with every step's wall clock attributed to timing buckets.

Written against trl==1.14.2. The hooks, and what each bucket therefore contains:

    weight_sync     ``VLLMGeneration.sync_weights()``, called by us (not TRL) at the
                    start of a generation round, every ``sync_every`` optimizer steps.
    rollout_gen     ``_generate_single_turn`` minus the sync: vLLM ``generate`` (or HF
                    ``generate``) plus turning the outputs into token-id lists.
    reward          ``_calculate_rewards`` (our CPU reward functions).
    advantage_loss  everything else inside ``training_step``: prompt tokenizing and
                    padding, decoding completions, the advantage math, the no-grad
                    old-log-prob forward (sub-bucket ``old_logprob``), the loss forward
                    (``loss_forward``), and the backward pass (``backward``).
    optimizer_step  ``optimizer.step()``, between the ``on_pre_optimizer_step`` and
                    ``on_optimizer_step`` callbacks.
    other           the remainder: fetching the next batch, gradient clipping,
                    ``zero_grad``, the LR scheduler, and TRL/HF logging.

Step boundaries: a step's window runs from one ``on_step_end`` to the next, so the
windows tile the whole training run with no gaps. The batch fetch and logging that
happen between ``on_step_end`` and the next step's work land in the next step's
``other``. The first window opens at ``on_train_begin``.

Timing assumes the baseline shape ``gradient_accumulation_steps=1`` and
``steps_per_generation=1``: one optimizer step = one generation round = one
``training_step``. Other shapes still sum correctly, but generation then happens only
on some steps, which the analysis has to account for.
"""

from __future__ import annotations

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer

from rlstudy.timing import StepTimer
from rlstudy.trace import TraceWriter


class ProfiledGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        timer: StepTimer,
        trace: TraceWriter,
        sync_every: int = 1,
        profile_steps: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sync_every < 1:
            raise ValueError("sync_every must be >= 1")
        if self.use_vllm and self.args.vllm_enable_sleep_mode and sync_every != 1:
            # Level-2 sleep discards vLLM's weights, so generate() re-pushes them every
            # round regardless of our schedule (NOTES.md section 2).
            raise ValueError("sync_every > 1 is meaningless with vllm_enable_sleep_mode=True")
        if self.args.gradient_accumulation_steps != 1 or self.args.steps_per_generation != 1:
            raise ValueError("profiling assumes gradient_accumulation_steps=1 and steps_per_generation=1")
        self.timer = timer
        self.trace = trace
        self.sync_every = sync_every
        self._sampler_step: int | None = None  # optimizer step whose weights vLLM holds
        self._in_scoring = False
        self._step_stats: dict = {}
        self._wrap_backward()
        self.add_callback(_TimingCallback(self, profile_steps))

    # ----- weight_sync + rollout_gen ---------------------------------------------

    def _sync_due(self) -> bool:
        if self._sampler_step is None:
            return True
        return self.state.global_step - self._sampler_step >= self.sync_every

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        synced = False if self.use_vllm else None  # None: no separate sampler to sync
        if self.use_vllm and self.state.global_step != self._last_loaded_step:
            if self._sync_due():
                with self.timer.span("weight_sync"):
                    self.vllm_generation.sync_weights()
                self._sampler_step = self.state.global_step
                synced = True
            # Tell TRL this step is handled, so its own sync guard does not fire.
            self._last_loaded_step = self.state.global_step
        self._step_stats["weight_synced"] = synced
        self._step_stats["sampler_staleness"] = (
            self.state.global_step - self._sampler_step if self._sampler_step is not None else None
        )
        with self.timer.span("rollout_gen"):
            return super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, num_generations, has_tool_images
            )

    # ----- reward ------------------------------------------------------------------

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        with self.timer.span("reward"):
            rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._step_stats["rewards_per_func"] = rewards_per_func.detach()
        return rewards_per_func

    # ----- advantage_loss and its sub-buckets ---------------------------------------

    def training_step(self, model, inputs, num_items_in_batch):
        with self.timer.span("advantage_loss"):
            return super().training_step(model, inputs, num_items_in_batch)

    def _generate_and_score_completions(self, inputs):
        self._in_scoring = True
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            self._in_scoring = False
        self._step_stats["completion_mask"] = output["completion_mask"].detach()
        self._step_stats["prompt_mask"] = output["prompt_mask"].detach()
        self._step_stats["advantages"] = output["advantages"].detach()
        return output

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        if not self._in_scoring:
            return super()._get_per_token_logps_and_entropies(model, *args, **kwargs)
        # Inside scoring, a log-prob pass is either the old-policy pass (importance
        # sampling correction) or the reference pass (only when beta > 0).
        sub = "ref_logprob" if (self.beta != 0.0 and model is getattr(self, "ref_model", None)) else "old_logprob"
        with self.timer.span(sub):
            return super()._get_per_token_logps_and_entropies(model, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.timer.span("loss_forward"):
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        self._step_stats["loss"] = loss.detach()
        return loss

    def _wrap_backward(self) -> None:
        original = self.accelerator.backward
        timer = self.timer

        def timed_backward(loss, **kwargs):
            with timer.span("backward"):
                return original(loss, **kwargs)

        self.accelerator.backward = timed_backward

    def on_profile_exported(self, trace_path) -> None:
        """Reduce the (large) profiler trace to per-span GPU busy fractions."""
        import json

        from rlstudy.gpumon import busy_fraction, load_trace_events
        from rlstudy.timing import BUCKETS, SUB_BUCKETS

        busy = busy_fraction(load_trace_events(trace_path), set(BUCKETS) | set(SUB_BUCKETS))
        (trace_path.parent / "profile_busy.json").write_text(json.dumps(busy, indent=2) + "\n")

    # ----- per-step metrics -----------------------------------------------------------

    def pop_step_metrics(self) -> dict:
        """Turn this step's captured tensors into plain numbers (after timing closed)."""
        s, self._step_stats = self._step_stats, {}
        m: dict = {
            "weight_synced": s.get("weight_synced"),
            "sampler_staleness": s.get("sampler_staleness"),
        }
        if "completion_mask" in s:
            lengths = s["completion_mask"].sum(dim=1).float()
            m["completions"] = int(lengths.numel())
            m["completion_tokens"] = int(lengths.sum().item())
            m["completion_len_mean"] = float(lengths.mean().item())
            m["completion_len_max"] = int(lengths.max().item())
            m["truncated_frac"] = float((lengths >= self.args.max_completion_length).float().mean().item())
            m["prompt_tokens"] = int(s["prompt_mask"].sum().item())
        if "rewards_per_func" in s:
            r = s["rewards_per_func"]
            names = [getattr(f, "__name__", str(f)) for f in self.reward_funcs]
            for i, name in enumerate(names):
                m[f"reward/{name}"] = float(torch.nanmean(r[:, i]).item())
            total = (r * self.reward_weights.to(r.device)).nansum(dim=1)
            m["reward_mean"] = float(total.mean().item())
            groups = total.view(-1, self.num_generations)
            # Groups where every sample got the same reward have zero advantage and
            # contribute no gradient: generation work that taught the model nothing.
            m["zero_signal_group_frac"] = float((groups.std(dim=1) == 0).float().mean().item())
        if "loss" in s:
            m["loss"] = float(s["loss"].item())
        return m


class _TimingCallback(TrainerCallback):
    """Opens and closes step windows, and runs an optional torch.profiler capture.

    Steps ``profile_steps = (first, last)`` inclusive run under the profiler, with
    every span annotated so it appears in the trace. Profiling slows those steps, so
    their records carry ``profiled: true`` and the analysis excludes them from timing.
    """

    def __init__(self, trainer: ProfiledGRPOTrainer, profile_steps: tuple[int, int] | None):
        self.t = trainer
        self.profile_steps = profile_steps
        self._prof = None

    def _maybe_start_profiler(self, next_step: int) -> None:
        if self.profile_steps is None or next_step != self.profile_steps[0]:
            return
        import torch
        from torch.profiler import ProfilerActivity, profile, record_function

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self._prof = profile(activities=activities)
        self._prof.__enter__()
        self.t.timer.annotate = record_function

    def _maybe_stop_profiler(self, finished_step: int) -> None:
        if self._prof is None or finished_step < self.profile_steps[1]:
            return
        self.t.timer.annotate = None
        self._prof.__exit__(None, None, None)
        out = self.t.trace.run_dir / "profile_trace.json.gz"
        self._prof.export_chrome_trace(str(out))
        self._prof = None
        self.t.on_profile_exported(out)

    def on_train_begin(self, args, state, control, **kwargs):
        self._maybe_start_profiler(state.global_step)
        self.t.timer.begin_step(state.global_step)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.t.timer.open("optimizer_step")

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.t.timer.close("optimizer_step")

    def on_step_end(self, args, state, control, **kwargs):
        record = self.t.timer.end_step().to_dict()
        finished = state.global_step - 1
        record["profiled"] = self._prof is not None
        self.t.trace.write_step(record, self.t.pop_step_metrics())
        self._maybe_stop_profiler(finished)
        self._maybe_start_profiler(state.global_step)
        self.t.timer.begin_step(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        # The last window holds only post-training teardown; it is not a step.
        self.t.timer.abort_step()
        if self._prof is not None:  # training ended inside the profile window
            self._maybe_stop_profiler(self.profile_steps[1])
