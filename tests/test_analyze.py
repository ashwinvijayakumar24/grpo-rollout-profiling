import json

import pytest

from rlstudy.analyze import aggregate_reps, analyze_experiment, summarize_run
from rlstudy.timing import BUCKETS
from rlstudy.trace import TraceWriter


def fake_step(step, rollout=3.0, reward=0.1, adv=1.0, opt=0.2, sync=0.0, other=0.2, tokens=1000, r=0.5):
    buckets = dict.fromkeys(BUCKETS, 0.0)
    buckets.update(rollout_gen=rollout, reward=reward, advantage_loss=adv, optimizer_step=opt, weight_sync=sync)
    return {
        "step": step,
        "total_s": sum(buckets.values()) + other,
        "buckets": buckets,
        "other_s": other,
        "sub_buckets": {"old_logprob": 0.3},
        "counts": {},
        "outside_step": {},
        "timing_enabled": True,
        "metrics": {
            "completions": 64, "completion_tokens": tokens, "completion_len_mean": tokens / 64,
            "completion_len_max": 50,
            "truncated_frac": 0.0, "zero_signal_group_frac": 0.25, "reward_mean": r,
            "reward/correctness_reward": r, "weight_synced": sync > 0,
        },
    }


def write_run(run_dir, steps, status="completed"):
    w = TraceWriter(run_dir, {"arm": run_dir.parent.name}, extra={"experiment": "T"})
    for s in steps:
        w.write_step({k: v for k, v in s.items() if k != "metrics"}, s["metrics"])
    w.finish(status=status)


def test_warmup_steps_are_dropped():
    steps = [fake_step(0, rollout=100.0), fake_step(1, rollout=50.0)] + [fake_step(i) for i in range(2, 6)]
    s = summarize_run({}, steps, warmup=2)
    assert s["steps_kept"] == 4
    assert s["buckets"]["rollout_gen"]["mean_s_per_step"] == pytest.approx(3.0)


def test_shares_sum_to_one_and_are_ratio_of_sums():
    steps = [fake_step(i, rollout=2.0 if i % 2 else 6.0) for i in range(2, 6)]
    s = summarize_run({}, steps, warmup=0)
    assert sum(b["share"] for b in s["buckets"].values()) == pytest.approx(1.0)
    total = sum(st["total_s"] for st in steps)
    assert s["buckets"]["rollout_gen"]["share"] == pytest.approx(16.0 / total)


def test_throughput_is_tokens_over_rollout_time():
    s = summarize_run({}, [fake_step(i, rollout=2.0, tokens=1000) for i in range(4)], warmup=0)
    assert s["rollout_tokens_per_s"] == pytest.approx(500.0)
    assert s["rollout_completions_per_s"] == pytest.approx(32.0)


def test_sync_cost_is_per_sync_not_per_step():
    steps = [fake_step(i, sync=0.4 if i % 4 == 0 else 0.0) for i in range(8)]
    s = summarize_run({}, steps, warmup=0)
    assert s["sync"]["steps_with_sync"] == 2
    assert s["sync"]["mean_s_per_sync"] == pytest.approx(0.4)
    assert s["buckets"]["weight_sync"]["mean_s_per_step"] == pytest.approx(0.1)


def test_reward_curve_includes_warmup():
    steps = [fake_step(i, r=i / 10) for i in range(10)]
    rc = summarize_run({}, steps, warmup=2)["reward"]
    assert rc["window"] == 5
    assert rc["reward_first"] == pytest.approx(0.2)
    assert rc["reward_last"] == pytest.approx(0.7)


def test_too_few_steps_raises():
    with pytest.raises(ValueError):
        summarize_run({}, [fake_step(0)], warmup=2)


def test_aggregate_reports_spread_across_reps():
    a = summarize_run({}, [fake_step(i, rollout=3.0) for i in range(4)], warmup=0)
    b = summarize_run({}, [fake_step(i, rollout=5.0) for i in range(4)], warmup=0)
    agg = aggregate_reps([a, b])
    m = agg["metrics"]["buckets.rollout_gen.mean_s_per_step"]
    assert m["mean"] == pytest.approx(4.0)
    assert m["stdev"] == pytest.approx(2 ** 0.5)
    assert (m["min"], m["max"], m["n"]) == (3.0, 5.0, 2)


def test_analyze_experiment_end_to_end(tmp_path):
    exp = tmp_path / "E2"
    for arm, rollout in (("g4", 2.0), ("g8", 3.0)):
        for rep in range(3):
            write_run(exp / arm / f"rep{rep}", [fake_step(i, rollout=rollout + 0.1 * rep) for i in range(6)])
    write_run(exp / "g8" / "rep_failed", [fake_step(i) for i in range(6)], status="failed")

    arms = analyze_experiment(exp)
    assert set(arms) == {"g4", "g8"}
    assert arms["g8"]["n_reps"] == 3  # the failed run is skipped
    assert (exp / "g4" / "rep0" / "summary.json").exists()
    agg = json.loads((exp / "g4" / "arm_summary.json").read_text())
    assert agg["metrics"]["buckets.rollout_gen.mean_s_per_step"]["mean"] == pytest.approx(2.1)
    table = (exp / "summary.md").read_text()
    assert "| g4 | 3 |" in table and "±" in table


def test_profiled_steps_are_excluded_from_timing():
    steps = [fake_step(i) for i in range(6)]
    steps[3]["profiled"] = True
    steps[3]["buckets"]["rollout_gen"] = 99.0
    s = summarize_run({}, steps, warmup=2)
    assert s["steps_kept"] == 3
    assert s["steps_profiled"] == 1
    assert s["buckets"]["rollout_gen"]["mean_s_per_step"] == pytest.approx(3.0)


def test_tail_metrics():
    s = summarize_run({}, [fake_step(i, rollout=2.0, tokens=1600) for i in range(4)], warmup=0)
    assert s["completion_len_max_mean"] == 50
    assert s["slot_occupancy"] == pytest.approx(1600 / (64 * 50))
    assert s["rollout_ms_per_longest_token"] == pytest.approx(40.0)
