"""Run every arm x rep of one experiment, each as a fresh process.

    python -m rlstudy.experiment configs/experiments/E2_group_size.yaml
    python -m rlstudy.experiment configs/experiments/E2_group_size.yaml --dry-run

Design choices, all aimed at comparable and non-redundant measurements:

- **Interleaved order.** Reps are the outer loop and arms the inner loop
  (G4 rep0, G8 rep0, G16 rep0, G4 rep1, ...), so slow drift on the node (thermals,
  a noisy neighbour) spreads across arms instead of biasing one of them.
- **Paired seeds.** Rep k uses seed k in every arm.
- **Fresh process per run.** vLLM does not reliably give back GPU memory inside one
  process, and a clean process keeps runs independent.
- **Resumable.** A run whose run.json says ``completed`` is skipped, so a requeued
  job never repeats finished work. An unfinished run directory is moved aside to
  ``<dir>.incomplete-<time>`` (kept as evidence) and rerun.
- **Budget guard.** Each run has a wall-clock limit (default 30 minutes). A run that
  hits it is killed and recorded as ``timeout`` in the ledger.
- **Ledger.** One JSON line per attempted run in ``results/<name>/ledger.jsonl``:
  arm, rep, status, seconds, Slurm job, git SHA.

Profile arms (``profile_arms`` in the experiment file) run once each after the
timing reps, with seed 0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from rlstudy.trace import git_info

REPO = Path(__file__).resolve().parent.parent


def plan_runs(spec: dict) -> list[dict]:
    runs = []
    for rep in range(spec.get("reps", 1)):
        for arm, overrides in (spec.get("arms") or {}).items():
            runs.append({"arm": arm, "rep": rep, "seed": rep, "overrides": overrides or {}})
    for arm, overrides in (spec.get("profile_arms") or {}).items():
        runs.append({"arm": arm, "rep": 0, "seed": 0, "overrides": overrides or {}})
    return runs


def run_status(run_dir: Path) -> str | None:
    header = run_dir / "run.json"
    if not header.exists():
        return None
    return json.loads(header.read_text()).get("status", "incomplete")


def _set_args(overrides: dict) -> list[str]:
    args = []
    for k, v in overrides.items():
        args += ["--set", f"{k}={json.dumps(v)}"]
    return args


def execute(spec_path: Path, results_root: Path, timeout_s: float, dry_run: bool, only_arms: set[str] | None) -> int:
    spec = yaml.safe_load(Path(spec_path).read_text())
    name = spec["name"]
    exp_dir = results_root / name
    ledger = exp_dir / "ledger.jsonl"
    base = REPO / spec["base"]
    failures = 0

    for r in plan_runs(spec):
        if only_arms and r["arm"] not in only_arms:
            continue
        run_dir = exp_dir / r["arm"] / f"rep{r['rep']}"
        status = run_status(run_dir)
        if status == "completed":
            print(f"[skip] {run_dir} already completed")
            continue
        cmd = [
            sys.executable, "-m", "rlstudy.train", "--config", str(base), "--out", str(run_dir),
            "--set", f"experiment={json.dumps(name)}", "--set", f"seed={r['seed']}",
            *_set_args(r["overrides"]),
        ]
        if dry_run:
            print("[plan]", " ".join(cmd[2:]))
            continue
        if run_dir.exists():
            aside = run_dir.with_name(f"{run_dir.name}.incomplete-{time.strftime('%Y%m%dT%H%M%S')}")
            run_dir.rename(aside)
            print(f"[moved] unfinished {run_dir} -> {aside.name}")
        run_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"[run] {name}/{r['arm']}/rep{r['rep']}", flush=True)
        t0 = time.time()
        log_path = run_dir.parent / f"rep{r['rep']}.log"
        with log_path.open("w") as log:
            try:
                proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
                outcome = "completed" if proc.returncode == 0 else f"exit_{proc.returncode}"
            except subprocess.TimeoutExpired:
                outcome = "timeout"
        elapsed = time.time() - t0
        if outcome != "completed":
            failures += 1
        entry = {
            "arm": r["arm"], "rep": r["rep"], "seed": r["seed"], "status": outcome,
            "wall_s": round(elapsed, 2), "run_dir": str(run_dir.relative_to(results_root)),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "git_sha": git_info(REPO)["sha"],
            "finished_unix": time.time(),
        }
        exp_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{outcome}] {r['arm']}/rep{r['rep']} in {elapsed:.1f}s (log: {log_path})", flush=True)

    if not dry_run:
        from rlstudy.analyze import analyze_experiment

        try:
            analyze_experiment(exp_dir)
        except Exception as e:  # analysis problems must not hide the run results
            print(f"[analyze] failed: {e!r}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--timeout-min", type=float, default=30.0)
    parser.add_argument("--arms", nargs="*", help="only run these arms")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    failures = execute(args.spec, args.results, args.timeout_min * 60, args.dry_run,
                       set(args.arms) if args.arms else None)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
