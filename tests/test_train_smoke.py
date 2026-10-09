"""End-to-end: a 3-step instrumented GRPO run on a tiny random model (HF generate).

Checks the plumbing, not performance: every step is traced, the buckets sum to the
step total, and each hooked stage actually recorded time.
"""

import os
import sys

import pytest

from rlstudy.timing import BUCKETS
from rlstudy.trace import read_trace

pytestmark = pytest.mark.slow

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    if sys.platform == "darwin":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from rlstudy.train import load_config, run

    config = load_config(None, [
        f"model={TINY}", "rollout.backend=hf", "grpo.max_steps=3", "grpo.prompts_per_step=2",
        "grpo.num_generations=4", "grpo.max_completion_length=8", "data.limit=8",
        "grpo.micro_batch=4",
    ])
    return run(config, tmp_path_factory.mktemp("smoke") / "rep0")


def test_every_step_is_traced(run_dir):
    header, steps = read_trace(run_dir)
    assert header["status"] == "completed"
    assert [s["step"] for s in steps] == [0, 1, 2]


def test_buckets_sum_to_total(run_dir):
    _, steps = read_trace(run_dir)
    for s in steps:
        assert set(s["buckets"]) == set(BUCKETS)
        assert sum(s["buckets"].values()) + s["other_s"] == pytest.approx(s["total_s"], abs=1e-6)
        assert s["other_s"] >= 0


def test_hooked_stages_record_time(run_dir):
    _, steps = read_trace(run_dir)
    for s in steps:
        for b in ("rollout_gen", "reward", "optimizer_step"):
            assert s["buckets"][b] > 0, (s["step"], b)
            assert s["counts"][b] == 1  # once per optimizer step
        # 8 completions / micro_batch 4 = 2 micro-steps, each a forward + backward.
        assert s["counts"]["advantage_loss"] == 2
        assert s["counts"]["loss_forward"] == 2
        assert s["counts"]["backward"] == 2
        assert s["buckets"]["weight_sync"] == 0  # HF generate: no separate sampler
        assert s["sub_buckets"]["loss_forward"] > 0
        assert s["sub_buckets"]["backward"] > 0
        assert "old_logprob" not in s["sub_buckets"]  # only computed with vLLM


def test_step_metrics(run_dir):
    _, steps = read_trace(run_dir)
    m = steps[0]["metrics"]
    assert m["completions"] == 8
    assert 0 < m["completion_tokens"] <= 8 * 8
    assert m["weight_synced"] is None
    assert 0.0 <= m["zero_signal_group_frac"] <= 1.0
    assert "reward/correctness_reward" in m
    assert m["micro_batches"] == 2


def test_step_windows_cover_training_time(run_dir):
    header, steps = read_trace(run_dir)
    covered = sum(s["total_s"] for s in steps)
    assert covered <= header["train_s"]
    assert covered > 0.5 * header["train_s"]


@pytest.fixture(scope="module")
def profiled_run_dir(tmp_path_factory):
    if sys.platform == "darwin":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from rlstudy.train import load_config, run

    config = load_config(None, [
        f"model={TINY}", "rollout.backend=hf", "grpo.max_steps=4", "grpo.prompts_per_step=2",
        "grpo.num_generations=2", "grpo.max_completion_length=4", "data.limit=8",
        "timing.profile_steps=[1, 2]",
    ])
    return run(config, tmp_path_factory.mktemp("prof") / "rep0")


def test_profile_window_marks_steps_and_writes_busy(profiled_run_dir):
    import json

    _, steps = read_trace(profiled_run_dir)
    assert [s["profiled"] for s in steps] == [False, True, True, False]
    assert (profiled_run_dir / "profile_trace.json.gz").exists()
    busy = json.loads((profiled_run_dir / "profile_busy.json").read_text())
    # Each profiled step annotates every hooked stage once.
    assert busy["rollout_gen"]["count"] == 2
    assert busy["advantage_loss"]["count"] == 2


def test_events_recorded(run_dir):
    _, steps = read_trace(run_dir)
    names = {e[0] for e in steps[1]["events"]}
    assert {"rollout_gen", "reward", "advantage_loss", "optimizer_step", "backward"} <= names
