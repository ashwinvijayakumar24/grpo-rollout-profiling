"""Turn step traces into per-run summaries and per-arm statistics across reps.

Layout it reads::

    results/<experiment>/<arm>/rep<k>/{run.json, trace.jsonl}

and writes ``summary.json`` next to each trace, ``arm_summary.json`` in each arm
directory, and ``summary.md`` (a table per experiment) in the experiment directory.

Rules, applied identically to every run:

- Steps run under torch.profiler (``profiled: true``) are dropped; the profiler
  slows them down.
- The first ``WARMUP_STEPS`` steps are dropped. They include CUDA graph capture,
  vLLM compilation warm-up, and allocator growth (NOTES.md pitfall 5).
- A bucket's *share* is its summed time over the summed step time (a ratio of sums),
  so long steps weigh more, the same way they do in wall clock.
- Rollout throughput is generated (completion) tokens divided by ``rollout_gen`` time.
- Across reps, every metric is reported as mean, sample standard deviation, min, max,
  and n. Variance is between reps, not between steps of one run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rlstudy.timing import BUCKETS
from rlstudy.trace import read_trace

WARMUP_STEPS = 2
ALL_BUCKETS = (*BUCKETS, "other")


def _bucket(step: dict, name: str) -> float:
    return step["other_s"] if name == "other" else step["buckets"][name]


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def summarize_run(header: dict, steps: list[dict], warmup: int = WARMUP_STEPS) -> dict:
    kept = [s for s in steps if s["step"] >= warmup and not s.get("profiled")]
    if not kept:
        raise ValueError(f"no steps left after dropping {warmup} warm-up steps (run has {len(steps)})")
    total = sum(s["total_s"] for s in kept)
    n = len(kept)

    buckets = {}
    for b in ALL_BUCKETS:
        t = sum(_bucket(s, b) for s in kept)
        buckets[b] = {"mean_s_per_step": t / n, "share": t / total}

    sub_names = sorted({k for s in kept for k in s["sub_buckets"]})
    sub = {k: {"mean_s_per_step": sum(s["sub_buckets"].get(k, 0.0) for s in kept) / n} for k in sub_names}

    m = [s["metrics"] for s in kept]
    gen_tokens = sum(x.get("completion_tokens", 0) for x in m)
    gen_time = sum(s["buckets"]["rollout_gen"] for s in kept)
    completions = sum(x.get("completions", 0) for x in m)

    synced = [s for s in kept if s["metrics"].get("weight_synced")]
    out = {
        "run_dir": header.get("run_dir"),
        "experiment": header.get("experiment"),
        "config_hash": header.get("config_hash"),
        "git_sha": (header.get("git") or {}).get("sha"),
        "slurm_job": ((header.get("env") or {}).get("slurm") or {}).get("SLURM_JOB_ID"),
        "gpu": [g.get("name") for g in (header.get("env") or {}).get("gpus", [])],
        "status": header.get("status"),
        "steps_total": len(steps),
        "steps_kept": n,
        "steps_profiled": sum(1 for s in steps if s.get("profiled")),
        "warmup_dropped": warmup,
        "step_time_mean_s": total / n,
        "step_time_stdev_s": statistics.stdev([s["total_s"] for s in kept]) if n > 1 else 0.0,
        "buckets": buckets,
        "sub_buckets": sub,
        "rollout_tokens_per_s": gen_tokens / gen_time if gen_time > 0 else None,
        "rollout_completions_per_s": completions / gen_time if gen_time > 0 else None,
        "completion_tokens_per_step": gen_tokens / n,
        "completion_len_mean": _mean([x["completion_len_mean"] for x in m if "completion_len_mean" in x]),
        # Decode runs until the longest sample finishes, so the longest sample, not the
        # mean, sets rollout time. Slot occupancy = generated tokens / (samples x longest):
        # the fraction of the batch's decode slots doing useful work.
        "completion_len_max_mean": _mean([x["completion_len_max"] for x in m if "completion_len_max" in x]),
        "slot_occupancy": _mean([
            x["completion_tokens"] / (x["completions"] * x["completion_len_max"])
            for x in m if x.get("completion_len_max")
        ]),
        "rollout_ms_per_longest_token": _mean([
            1000 * s["buckets"]["rollout_gen"] / s["metrics"]["completion_len_max"]
            for s in kept if s["metrics"].get("completion_len_max")
        ]),
        "truncated_frac_mean": _mean([x["truncated_frac"] for x in m if "truncated_frac" in x]),
        "zero_signal_group_frac_mean": _mean([x["zero_signal_group_frac"] for x in m if "zero_signal_group_frac" in x]),
        "sync": {
            "steps_with_sync": len(synced),
            "mean_s_per_sync": (sum(s["buckets"]["weight_sync"] for s in synced) / len(synced)) if synced else None,
        },
        "reward": _reward_curve(steps),
    }
    return out


def _reward_curve(steps: list[dict], window: int = 10) -> dict:
    """Reward at the start and end of training, over all steps (warm-up included).

    Learning progress is about the whole run, so warm-up steps are not dropped here.
    """
    r = [s["metrics"]["reward_mean"] for s in steps if "reward_mean" in s["metrics"]]
    c = [s["metrics"]["reward/correctness_reward"] for s in steps if "reward/correctness_reward" in s["metrics"]]
    if not r:
        return {}
    w = max(1, min(window, len(r) // 2))
    return {
        "window": w,
        "reward_first": statistics.fmean(r[:w]),
        "reward_last": statistics.fmean(r[-w:]),
        "correct_first": statistics.fmean(c[:w]) if c else None,
        "correct_last": statistics.fmean(c[-w:]) if c else None,
    }


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = float(v)
    return flat


def stats(xs: list[float]) -> dict:
    return {
        "mean": statistics.fmean(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else None,
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
    }


def aggregate_reps(summaries: list[dict]) -> dict:
    """Mean/stdev/min/max/n across reps for every numeric field in the summaries."""
    flats = [_flatten(s) for s in summaries]
    keys = sorted(set().union(*flats))
    return {
        "n_reps": len(summaries),
        "runs": [s.get("run_dir") for s in summaries],
        "slurm_jobs": sorted({s.get("slurm_job") for s in summaries if s.get("slurm_job")}),
        "gpus": sorted({g for s in summaries for g in s.get("gpu", [])}),
        "config_hashes": sorted({s.get("config_hash") for s in summaries if s.get("config_hash")}),
        "metrics": {k: stats([f[k] for f in flats if k in f]) for k in keys},
    }


def find_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).rglob("trace.jsonl"))


def analyze_experiment(exp_dir: Path, warmup: int = WARMUP_STEPS) -> dict[str, dict]:
    """Summarize every run under exp_dir and aggregate reps per arm (the run's parent dir)."""
    exp_dir = Path(exp_dir)
    by_arm: dict[Path, list[dict]] = {}
    for run_dir in find_runs(exp_dir):
        header, steps = read_trace(run_dir)
        if header.get("status") != "completed":
            print(f"skipping {run_dir}: status={header.get('status')}")
            continue
        header["run_dir"] = str(run_dir.relative_to(exp_dir.parent))
        summary = summarize_run(header, steps, warmup)
        if (run_dir / "gpu_util.csv").exists():
            from rlstudy.gpumon import nvml_by_span

            kept = [s for s in steps if s["step"] >= warmup and not s.get("profiled")]
            summary["nvml_util_by_span"] = nvml_by_span(run_dir / "gpu_util.csv", kept)
        if (run_dir / "profile_busy.json").exists():
            summary["profile_busy"] = json.loads((run_dir / "profile_busy.json").read_text())
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        by_arm.setdefault(run_dir.parent, []).append(summary)

    arms = {}
    for arm_dir, summaries in sorted(by_arm.items()):
        agg = aggregate_reps(summaries)
        (arm_dir / "arm_summary.json").write_text(json.dumps(agg, indent=2) + "\n")
        arms[str(arm_dir.relative_to(exp_dir))] = agg
    if arms:
        (exp_dir / "summary.md").write_text(markdown_table(exp_dir.name, arms))
    return arms


def _fmt(s: dict | None, scale: float = 1.0, digits: int = 3) -> str:
    if not s:
        return "—"
    mean = s["mean"] * scale
    if s["stdev"] is None:
        return f"{mean:.{digits}f} (n=1)"
    return f"{mean:.{digits}f} ± {s['stdev'] * scale:.{digits}f}"


def markdown_table(name: str, arms: dict[str, dict]) -> str:
    """One row per arm: step time, bucket shares (%), and rollout throughput, mean ± stdev over reps."""
    cols = ["arm", "reps", "step s"] + [f"{b} %" for b in ALL_BUCKETS] + ["rollout tok/s"]
    lines = [
        f"# {name}: per-arm summary",
        "",
        f"Mean ± sample standard deviation across reps. First {WARMUP_STEPS} steps of each run dropped.",
        "Generated by `python -m rlstudy.analyze`; do not edit by hand.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for arm, agg in arms.items():
        m = agg["metrics"]
        row = [arm, str(agg["n_reps"]), _fmt(m.get("step_time_mean_s"))]
        row += [_fmt(m.get(f"buckets.{b}.share"), 100, 1) for b in ALL_BUCKETS]
        row += [_fmt(m.get("rollout_tokens_per_s"), 1, 0)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    args = parser.parse_args()
    for d in args.experiment_dirs:
        arms = analyze_experiment(d, args.warmup)
        print(f"{d}: {len(arms)} arm(s)")
        if arms:
            print((d / "summary.md").read_text())


if __name__ == "__main__":
    main()
