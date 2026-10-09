import pytest

from rlstudy.timing import BUCKETS, StepTimer


class FakeGPU:
    """A clock plus an asynchronous work queue, like a CUDA stream.

    ``launch(s)`` queues s seconds of GPU work and returns immediately (CPU time does
    not advance). ``sync()`` waits for the queue to drain, advancing the clock.
    ``cpu(s)`` advances the clock directly, like Python work.
    """

    def __init__(self):
        self.now_ns = 0
        self.gpu_busy_until_ns = 0

    def clock(self) -> int:
        return self.now_ns

    def cpu(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)

    def launch(self, seconds: float) -> None:
        start = max(self.now_ns, self.gpu_busy_until_ns)
        self.gpu_busy_until_ns = start + int(seconds * 1e9)

    def sync(self) -> None:
        self.now_ns = max(self.now_ns, self.gpu_busy_until_ns)


def make(gpu: FakeGPU, **kw) -> StepTimer:
    return StepTimer(sync=gpu.sync, clock=gpu.clock, **kw)


def test_buckets_plus_other_equal_total():
    gpu = FakeGPU()
    t = make(gpu)
    t.begin_step(0)
    gpu.cpu(0.01)  # untimed: lands in other
    with t.span("weight_sync"):
        gpu.cpu(0.2)
    with t.span("rollout_gen"):
        gpu.cpu(3.0)
    with t.span("reward"):
        gpu.cpu(0.05)
    with t.span("advantage_loss"):
        gpu.launch(1.0)
    with t.span("optimizer_step"):
        gpu.launch(0.1)
    gpu.cpu(0.02)
    r = t.end_step()
    assert r.buckets == pytest.approx(
        {"weight_sync": 0.2, "rollout_gen": 3.0, "reward": 0.05, "advantage_loss": 1.0, "optimizer_step": 0.1}
    )
    assert r.other_s == pytest.approx(0.03)
    assert sum(r.buckets.values()) + r.other_s == pytest.approx(r.total_s)


def test_async_gpu_work_is_charged_to_the_bucket_that_launched_it():
    # The bug this guards against: without a sync at the span boundary, the backward
    # pass's 1.0 s would be charged to optimizer_step, which is the first thing to wait.
    gpu = FakeGPU()
    t = make(gpu)
    t.begin_step(0)
    with t.span("advantage_loss"):
        gpu.launch(1.0)  # backward kernels, returns immediately
    with t.span("optimizer_step"):
        gpu.launch(0.1)
    r = t.end_step()
    assert r.buckets["advantage_loss"] == pytest.approx(1.0)
    assert r.buckets["optimizer_step"] == pytest.approx(0.1)


def test_without_sync_the_attribution_is_wrong():
    # Documents why sync matters: the same workload timed with a no-op sync.
    gpu = FakeGPU()
    t = StepTimer(sync=lambda: None, clock=gpu.clock)
    t.begin_step(0)
    with t.span("advantage_loss"):
        gpu.launch(1.0)
    with t.span("optimizer_step"):
        gpu.launch(0.1)
        gpu.sync()  # e.g. a .item() inside the optimizer path
    r = t.end_step()
    assert r.buckets["advantage_loss"] == 0.0
    assert r.buckets["optimizer_step"] == pytest.approx(1.1)


def test_work_queued_before_the_step_is_not_charged_to_the_step():
    gpu = FakeGPU()
    t = make(gpu)
    gpu.launch(5.0)  # leftover from a previous step
    t.begin_step(1)
    with t.span("reward"):
        gpu.cpu(0.1)
    r = t.end_step()
    assert r.total_s == pytest.approx(0.1)


def test_repeated_spans_accumulate_and_are_counted():
    gpu = FakeGPU()
    t = make(gpu)
    t.begin_step(0)
    for _ in range(4):  # e.g. gradient accumulation micro-steps
        with t.span("advantage_loss"):
            gpu.launch(0.25)
    r = t.end_step()
    assert r.buckets["advantage_loss"] == pytest.approx(1.0)
    assert r.counts["advantage_loss"] == 4


def test_top_level_buckets_may_not_nest():
    t = make(FakeGPU())
    t.begin_step(0)
    with t.span("rollout_gen"):
        with pytest.raises(RuntimeError, match="may not nest"):
            with t.span("weight_sync"):
                pass


def test_sub_bucket_inside_parent_is_reported_separately():
    gpu = FakeGPU()
    t = make(gpu)
    t.begin_step(0)
    with t.span("advantage_loss"):
        with t.span("old_logprob"):
            gpu.launch(0.3)
        gpu.launch(0.7)
    r = t.end_step()
    assert r.buckets["advantage_loss"] == pytest.approx(1.0)
    assert r.sub_buckets["old_logprob"] == pytest.approx(0.3)
    assert r.other_s == pytest.approx(0.0)


def test_sub_bucket_outside_parent_raises():
    t = make(FakeGPU())
    t.begin_step(0)
    with pytest.raises(RuntimeError, match="must be inside"):
        with t.span("old_logprob"):
            pass


def test_unknown_bucket_raises():
    t = make(FakeGPU())
    t.begin_step(0)
    with pytest.raises(KeyError):
        with t.span("backward"):
            pass


def test_span_outside_a_step_is_reported_on_the_next_record():
    gpu = FakeGPU()
    t = make(gpu)
    with t.span("weight_sync"):  # initial sync before training starts
        gpu.cpu(0.4)
    t.begin_step(0)
    r = t.end_step()
    assert r.outside_step == {"weight_sync": pytest.approx(0.4)}
    assert r.buckets["weight_sync"] == 0.0
    t.begin_step(1)
    assert t.end_step().outside_step == {}


def test_span_closes_on_exception():
    gpu = FakeGPU()
    t = make(gpu)
    t.begin_step(0)
    with pytest.raises(ValueError):
        with t.span("reward"):
            gpu.cpu(0.1)
            raise ValueError("bad reward")
    r = t.end_step()
    assert r.buckets["reward"] == pytest.approx(0.1)


def test_disabled_timer_measures_only_the_total():
    gpu = FakeGPU()
    t = make(gpu, enabled=False)
    t.begin_step(0)
    with t.span("advantage_loss"):
        gpu.launch(1.0)
    r = t.end_step()
    assert r.total_s == pytest.approx(1.0)  # the step-end sync still waits for the GPU
    assert all(v == 0.0 for v in r.buckets.values())
    assert r.timing_enabled is False


def test_step_lifecycle_errors():
    t = make(FakeGPU())
    with pytest.raises(RuntimeError):
        t.end_step()
    t.begin_step(0)
    with pytest.raises(RuntimeError):
        t.begin_step(1)


def test_record_dict_has_every_bucket():
    gpu = FakeGPU()
    t = make(gpu)
    t.begin_step(7)
    d = t.end_step().to_dict()
    assert d["step"] == 7
    assert set(d["buckets"]) == set(BUCKETS)
