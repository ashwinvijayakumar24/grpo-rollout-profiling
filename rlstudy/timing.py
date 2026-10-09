"""Per-step wall-clock attribution into fixed buckets.

One training step is split into five measured top-level buckets plus ``other``:

    rollout_gen     sampling completions (vLLM or HF generate)
    reward          reward functions (CPU)
    advantage_loss  advantages, the old-log-prob forward, loss forward, backward
    optimizer_step  optimizer.step()
    weight_sync     copying trainer weights into the sampler
    other           step total minus the five above (data, padding, clipping, logging)

Because ``other`` is defined as the remainder, the buckets always add up to the step
total. That only means something if no time is counted twice, so top-level spans may
not nest: opening one inside another raises. Sub-buckets (for example
``old_logprob`` inside ``advantage_loss``) are allowed only inside their parent and
are reported separately; they never feed the sum.

GPU work is asynchronous: a kernel launch returns before the kernel runs. Reading the
clock without waiting would charge, say, the backward pass to whichever later bucket
first waits on the GPU. So every span boundary calls ``sync()`` (``torch.cuda.
synchronize`` on CUDA) before reading the clock. ``enabled=False`` turns spans into
no-ops and keeps only the step-boundary syncs, which is how the overhead of the
instrumentation itself is measured.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

BUCKETS = ("rollout_gen", "reward", "advantage_loss", "optimizer_step", "weight_sync")
SUB_BUCKETS = {
    "old_logprob": "advantage_loss",
    "loss_forward_backward": "advantage_loss",
    "sync_merge": "weight_sync",
    "sync_copy": "weight_sync",
    "sync_unmerge": "weight_sync",
}

# Measurement noise tolerance when checking that buckets do not exceed the total.
_SUM_TOLERANCE_S = 1e-6


def device_sync_fn(device: str | None = None) -> Callable[[], None]:
    """Return the function that waits for all queued work on the given device."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device.startswith("mps"):
        return torch.mps.synchronize
    return lambda: None


@dataclass
class StepRecord:
    step: int
    total_s: float
    buckets: dict[str, float]
    sub_buckets: dict[str, float]
    counts: dict[str, int]
    other_s: float
    timing_enabled: bool
    outside_step: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "total_s": self.total_s,
            "buckets": dict(self.buckets),
            "other_s": self.other_s,
            "sub_buckets": dict(self.sub_buckets),
            "counts": dict(self.counts),
            "outside_step": dict(self.outside_step),
            "timing_enabled": self.timing_enabled,
        }


class StepTimer:
    def __init__(
        self,
        sync: Callable[[], None] = lambda: None,
        clock: Callable[[], int] = time.perf_counter_ns,
        enabled: bool = True,
    ):
        self._sync = sync
        self._clock = clock
        self.enabled = enabled
        self._step: int | None = None
        self._step_start = 0
        self._open: list[str] = []
        self._reset_accumulators()
        # Time spent in spans while no step is open (for example the very first
        # weight sync before training starts). Reported on the next step's record.
        self._outside: dict[str, float] = {}

    def _reset_accumulators(self) -> None:
        self._buckets = {b: 0.0 for b in BUCKETS}
        self._sub = {}
        self._counts = {}

    def _now(self) -> int:
        self._sync()
        return self._clock()

    @property
    def in_step(self) -> bool:
        return self._step is not None

    def begin_step(self, step: int) -> None:
        if self._step is not None:
            raise RuntimeError(f"begin_step({step}) while step {self._step} is still open")
        self._reset_accumulators()
        self._step = step
        self._step_start = self._now()

    def end_step(self) -> StepRecord:
        if self._step is None:
            raise RuntimeError("end_step() with no open step")
        if self._open:
            raise RuntimeError(f"end_step() with spans still open: {self._open}")
        total = (self._now() - self._step_start) / 1e9
        measured = sum(self._buckets.values())
        if measured > total + _SUM_TOLERANCE_S:
            raise AssertionError(f"buckets sum to {measured:.6f}s, more than the step total {total:.6f}s")
        record = StepRecord(
            step=self._step,
            total_s=total,
            buckets=dict(self._buckets),
            sub_buckets=dict(self._sub),
            counts=dict(self._counts),
            other_s=max(total - measured, 0.0),
            timing_enabled=self.enabled,
            outside_step=self._outside,
        )
        self._outside = {}
        self._step = None
        return record

    @contextmanager
    def span(self, name: str):
        """Time a block and charge it to ``name`` (a bucket or a sub-bucket)."""
        if not self.enabled:
            yield
            return
        self._check_nesting(name)
        self._open.append(name)
        start = self._now()
        try:
            yield
        finally:
            elapsed = (self._now() - start) / 1e9
            self._open.pop()
            self._charge(name, elapsed)

    def _check_nesting(self, name: str) -> None:
        if name in BUCKETS:
            top = [s for s in self._open if s in BUCKETS]
            if top:
                raise RuntimeError(f"span {name!r} opened inside {top[-1]!r}; top-level buckets may not nest")
        elif name in SUB_BUCKETS:
            parent = SUB_BUCKETS[name]
            if parent not in self._open:
                raise RuntimeError(f"sub-bucket {name!r} must be inside {parent!r}, open spans: {self._open}")
        else:
            raise KeyError(f"unknown bucket {name!r}")

    def _charge(self, name: str, elapsed: float) -> None:
        if self._step is None:
            self._outside[name] = self._outside.get(name, 0.0) + elapsed
            return
        if name in BUCKETS:
            self._buckets[name] += elapsed
        else:
            self._sub[name] = self._sub.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1
