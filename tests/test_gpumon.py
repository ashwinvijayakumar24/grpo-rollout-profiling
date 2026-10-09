import gzip
import json

import pytest

from rlstudy.gpumon import NvmlSampler, _union_length, busy_fraction, load_trace_events


def span(name, ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": name, "ts": ts, "dur": dur}


def kern(ts, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": "k", "ts": ts, "dur": dur}


def test_union_length_merges_overlaps():
    assert _union_length([(0, 10), (5, 15), (20, 25)]) == 20
    assert _union_length([]) == 0


def test_idle_sync_span():
    # A 100 us sync with one 10 us copy: the GPU is idle 90% of the span.
    events = [span("weight_sync", 1000, 100), kern(1050, 10, "gpu_memcpy")]
    r = busy_fraction(events)["weight_sync"]
    assert r["busy_frac"] == pytest.approx(0.1)
    assert r["by_kind_s"]["memcpy"] == pytest.approx(10e-6)


def test_kernels_are_clipped_to_span_and_streams_not_double_counted():
    events = [
        span("rollout_gen", 100, 100),
        kern(50, 100),    # starts before the span: 50 us inside
        kern(120, 40),    # overlaps the first on another stream
        kern(190, 50),    # runs past the end: 10 us inside
        kern(500, 10),    # after the span
    ]
    r = busy_fraction(events)["rollout_gen"]
    assert r["busy_s"] == pytest.approx(70e-6)  # [100,160) U [190,200)
    assert r["busy_frac"] == pytest.approx(0.7)


def test_multiple_occurrences_are_summed():
    events = [span("weight_sync", 0, 100), kern(0, 100), span("weight_sync", 1000, 100)]
    r = busy_fraction(events)["weight_sync"]
    assert r["count"] == 2
    assert r["busy_frac"] == pytest.approx(0.5)


def test_filter_by_name():
    events = [span("weight_sync", 0, 10), span("ProfilerStep#3", 0, 10)]
    assert set(busy_fraction(events, {"weight_sync"})) == {"weight_sync"}


def test_load_gzipped_trace(tmp_path):
    p = tmp_path / "t.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"traceEvents": [span("reward", 0, 5)]}, f)
    assert load_trace_events(p)[0]["name"] == "reward"


def test_sampler_is_noop_without_nvidia_gpu(tmp_path):
    s = NvmlSampler(tmp_path / "gpu_util.csv").start()
    s.stop()
    if not s.available:
        assert not (tmp_path / "gpu_util.csv").exists()
