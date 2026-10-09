import json

import pytest

from rlstudy.timing import StepTimer
from rlstudy.trace import SCHEMA_VERSION, TraceWriter, config_hash, read_trace


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_round_trip(tmp_path):
    cfg = {"group_size": 8, "max_new_tokens": 128}
    w = TraceWriter(tmp_path / "run", cfg, extra={"experiment": "E1", "rep": 0})
    t = StepTimer()
    for step in range(3):
        t.begin_step(step)
        with t.span("reward"):
            pass
        w.write_step(t.end_step().to_dict(), {"reward_mean": 0.5, "completion_tokens": 100})
    w.finish(steps=3)

    header, steps = read_trace(tmp_path / "run")
    assert header["config"] == cfg
    assert header["config_hash"] == config_hash(cfg)
    assert header["experiment"] == "E1"
    assert header["status"] == "completed"
    assert header["steps"] == 3
    assert {"hostname", "gpus", "libraries", "slurm"} <= set(header["env"])
    assert header["env"]["libraries"]["trl"] == "1.14.2"
    assert len(header["git"]["sha"]) == 40
    assert [s["step"] for s in steps] == [0, 1, 2]
    assert steps[0]["schema_version"] == SCHEMA_VERSION
    assert steps[0]["metrics"]["reward_mean"] == 0.5
    assert "reward" in steps[0]["buckets"]


def test_lines_are_flushed_before_finish(tmp_path):
    # A run killed mid-way must still leave its completed steps on disk.
    w = TraceWriter(tmp_path, {})
    w.write_step({"step": 0})
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["step"] == 0
    assert "status" not in json.loads((tmp_path / "run.json").read_text())


def test_refuses_to_append_to_an_existing_trace(tmp_path):
    w = TraceWriter(tmp_path, {})
    w.write_step({"step": 0})
    w.finish()
    with pytest.raises(FileExistsError):
        TraceWriter(tmp_path, {})
