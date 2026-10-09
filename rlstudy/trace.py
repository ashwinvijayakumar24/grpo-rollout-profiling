"""Run artifacts: one ``run.json`` header and one ``trace.jsonl`` line per step.

Layout of a run directory::

    results/<experiment>/<arm>/rep<k>/
        run.json       config, config hash, git SHA, Slurm job, hardware, library versions
        trace.jsonl    one StepRecord per optimizer step, plus training metrics
        summary.json   written by rlstudy.analyze

Everything needed to trace a number in the writeup back to its source is in run.json,
so a result never has to be rerun just to find out how it was produced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

SCHEMA_VERSION = 1

_LIBRARIES = ("torch", "vllm", "trl", "transformers", "accelerate", "peft", "datasets", "flashinfer-python")


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(repo: Path) -> dict:
    sha = _run(["git", "rev-parse", "HEAD"], repo)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def gpu_info() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm", "--format=csv,noheader"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        name, mem, driver, clock = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "memory_total": mem, "driver_version": driver, "max_sm_clock": clock})
    return gpus


def library_versions() -> dict:
    versions = {}
    for lib in _LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda
        versions["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    except ImportError:
        pass
    return versions


def environment() -> dict:
    slurm_keys = ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "SLURM_JOB_NODELIST", "SLURM_ARRAY_TASK_ID")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm": {k: os.environ.get(k) for k in slurm_keys if os.environ.get(k)},
        "gpus": gpu_info(),
        "libraries": library_versions(),
    }


class TraceWriter:
    """Writes run.json once, then appends one JSON line per step, flushed immediately.

    Flushing every line means a run killed partway (preemption, OOM, time limit) still
    leaves every completed step on disk.
    """

    def __init__(self, run_dir: Path, config: dict, repo: Path | None = None, extra: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.run_dir / "trace.jsonl"
        if trace_path.exists() and trace_path.stat().st_size > 0:
            raise FileExistsError(f"{trace_path} already has data; refusing to mix two runs in one trace")
        repo = repo or Path(__file__).resolve().parent.parent
        self.header = {
            "schema_version": SCHEMA_VERSION,
            "started_unix": time.time(),
            "config": config,
            "config_hash": config_hash(config),
            "git": git_info(repo),
            "env": environment(),
            **(extra or {}),
        }
        (self.run_dir / "run.json").write_text(json.dumps(self.header, indent=2, default=str) + "\n")
        self._f = trace_path.open("a")

    def write_step(self, record: dict, metrics: dict | None = None) -> None:
        line = {"schema_version": SCHEMA_VERSION, "wall_unix": time.time(), **record, "metrics": metrics or {}}
        self._f.write(json.dumps(line, default=float) + "\n")
        self._f.flush()

    def finish(self, status: str = "completed", **fields) -> None:
        self._f.close()
        self.header.update({"finished_unix": time.time(), "status": status, **fields})
        (self.run_dir / "run.json").write_text(json.dumps(self.header, indent=2, default=str) + "\n")


def read_trace(run_dir: Path) -> tuple[dict, list[dict]]:
    run_dir = Path(run_dir)
    header = json.loads((run_dir / "run.json").read_text())
    steps = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines() if line.strip()]
    return header, steps
