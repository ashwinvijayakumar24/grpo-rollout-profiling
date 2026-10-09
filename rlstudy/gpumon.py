"""GPU activity during each stage: an NVML sampler and profiler-trace analysis.

Two instruments, because they answer the question at different resolutions:

1. ``NvmlSampler``: a background thread polling NVML (NVIDIA's management library)
   every ``interval_s`` and writing ``gpu_util.csv``. Cheap enough to run for whole
   runs. Its ``utilization.gpu`` counter is "percent of the last sample period in
   which any kernel ran", averaged by the driver over a period between 1/6 s and 1 s
   depending on the GPU. So it cannot resolve anything shorter than that; a 50 ms
   weight sync is invisible to it.

2. A torch.profiler capture over a few steps (``busy_fraction``). It records every
   GPU kernel and memory copy with start and end times, on the same clock as our
   span annotations, so we can compute exactly what fraction of, say, the
   ``weight_sync`` span the GPU was executing anything. This is the instrument E4's
   "is the GPU idle during sync?" question relies on.
"""

from __future__ import annotations

import csv
import gzip
import json
import threading
import time
from collections import defaultdict
from pathlib import Path


class NvmlSampler:
    """Polls one GPU's utilization, memory, power, and SM clock in a background thread.

    Timestamps use ``time.perf_counter_ns``, the same clock as StepTimer, so samples
    line up with span events directly. If NVML is unavailable (no NVIDIA GPU), the
    sampler is a no-op and ``available`` is False.
    """

    FIELDS = ("t_ns", "util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w", "sm_clock_mhz")

    def __init__(self, out_path: Path, device_index: int = 0, interval_s: float = 0.05):
        self.out_path = Path(out_path)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:  # ImportError, or NVMLError when there is no driver
            self._nvml = None
            self.available = False

    def _sample(self) -> tuple:
        n, h = self._nvml, self._handle
        u = n.nvmlDeviceGetUtilizationRates(h)
        mem = n.nvmlDeviceGetMemoryInfo(h)
        return (
            time.perf_counter_ns(),
            u.gpu,
            u.memory,
            mem.used // (1024 * 1024),
            n.nvmlDeviceGetPowerUsage(h) / 1000.0,
            n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
        )

    def _run(self) -> None:
        with self.out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.FIELDS)
            while not self._stop.is_set():
                w.writerow(self._sample())
                self._stop.wait(self.interval_s)
            f.flush()

    def start(self) -> "NvmlSampler":
        if self.available:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._run, name="nvml-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None


# ----- profiler trace analysis ------------------------------------------------------

GPU_CATEGORIES = {"kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset"}


def load_trace_events(path: Path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def busy_fraction(events: list[dict], span_names: set[str] | None = None) -> dict[str, dict]:
    """For each annotated span name: wall time, and the fraction with GPU work running.

    "Busy" is the union of kernel, memcpy, and memset intervals clipped to the span,
    so overlapping streams are not double counted. Returned per span name, summed over
    all occurrences in the trace (times in seconds).
    """
    gpu = [
        (e["ts"], e["ts"] + e["dur"], GPU_CATEGORIES[e["cat"]])
        for e in events
        if e.get("ph") == "X" and e.get("cat") in GPU_CATEGORIES and "dur" in e
    ]
    gpu.sort()
    starts = [g[0] for g in gpu]
    spans = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat") == "user_annotation"
        and (span_names is None or e.get("name") in span_names)
    ]

    import bisect

    out: dict[str, dict] = defaultdict(lambda: {"wall_s": 0.0, "busy_s": 0.0, "count": 0,
                                                 "by_kind_s": defaultdict(float)})
    for sp in spans:
        s0, s1 = sp["ts"], sp["ts"] + sp["dur"]
        # Kernels sorted by start; anything starting after s1 cannot overlap.
        hi = bisect.bisect_right(starts, s1)
        clipped, by_kind = [], defaultdict(list)
        for g0, g1, kind in gpu[:hi]:
            if g1 <= s0:
                continue
            iv = (max(g0, s0), min(g1, s1))
            clipped.append(iv)
            by_kind[kind].append(iv)
        rec = out[sp["name"]]
        rec["wall_s"] += sp["dur"] / 1e6
        rec["busy_s"] += _union_length(clipped) / 1e6
        rec["count"] += 1
        for kind, ivs in by_kind.items():
            rec["by_kind_s"][kind] += _union_length(ivs) / 1e6

    result = {}
    for name, rec in out.items():
        result[name] = {
            "wall_s": rec["wall_s"],
            "busy_s": rec["busy_s"],
            "busy_frac": rec["busy_s"] / rec["wall_s"] if rec["wall_s"] > 0 else None,
            "count": rec["count"],
            "by_kind_s": dict(rec["by_kind_s"]),
        }
    return result


def nvml_by_span(csv_path: Path, steps: list[dict], names: set[str] | None = None) -> dict[str, dict]:
    """Mean NVML utilization of samples taken inside each span (inclusive times).

    Coarse by construction (see NvmlSampler): use it for long spans like rollout_gen,
    and the profiler's busy_fraction for short ones like weight_sync.
    """
    import bisect

    rows = list(csv.DictReader(Path(csv_path).open()))
    if not rows:
        return {}
    t = [int(r["t_ns"]) for r in rows]
    util = [float(r["util_gpu_pct"]) for r in rows]
    acc: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        for name, start, end in step.get("events", []):
            if names is not None and name not in names:
                continue
            lo, hi = bisect.bisect_left(t, start), bisect.bisect_right(t, end)
            acc[name].extend(util[lo:hi])
    return {
        name: {"mean_util_gpu_pct": sum(v) / len(v), "samples": len(v)}
        for name, v in acc.items() if v
    }
