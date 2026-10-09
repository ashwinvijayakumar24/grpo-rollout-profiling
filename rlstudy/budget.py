"""Project each experiment's GPU time from a measured smoke run, before submitting it.

    python -m rlstudy.budget results/smoke/vllm/rep0

Per run: setup time (model load, vLLM start-up, CUDA graph capture) plus
steps x measured seconds per step, scaled for the arm's batch and length changes.
The scaling is a deliberately rough upper-bound model (step time scales linearly with
completions per step and with max completion length), used only to decide whether an
experiment needs a longer Slurm time limit or must be flagged as over 30 GPU-minutes.
These projections are planning numbers; they are printed, never written into
results or the writeup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rlstudy.experiment import REPO, plan_runs
from rlstudy.train import load_config
from rlstudy.trace import read_trace

FLAG_MINUTES = 30.0


def measured(smoke_run: Path) -> dict:
    header, steps = read_trace(smoke_run)
    kept = [s for s in steps if s["step"] >= 2 and not s.get("profiled")] or steps
    step_s = sum(s["total_s"] for s in kept) / len(kept)
    cfg = header["config"]
    return {
        "setup_s": header["setup_s"],
        "step_s": step_s,
        "completions": cfg["grpo"]["prompts_per_step"] * cfg["grpo"]["num_generations"],
        "max_len": cfg["grpo"]["max_completion_length"],
    }


def project(spec_path: Path, m: dict) -> list[dict]:
    spec = yaml.safe_load(Path(spec_path).read_text())
    rows = []
    for r in plan_runs(spec):
        sets = [f"{k}={json.dumps(v)}" for k, v in r["overrides"].items()]
        cfg = load_config(REPO / spec["base"], sets)
        g = cfg["grpo"]
        scale = (g["prompts_per_step"] * g["num_generations"] / m["completions"]) * max(
            1.0, g["max_completion_length"] / m["max_len"]
        )
        minutes = (m["setup_s"] + g["max_steps"] * m["step_s"] * scale) / 60
        rows.append({"arm": r["arm"], "rep": r["rep"], "minutes": minutes})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smoke_run", type=Path)
    parser.add_argument("specs", nargs="*", type=Path)
    args = parser.parse_args()
    m = measured(args.smoke_run)
    print(f"measured: setup {m['setup_s']:.0f}s, {m['step_s']:.2f}s/step at "
          f"{m['completions']} completions x {m['max_len']} max tokens")
    specs = args.specs or sorted((REPO / "configs" / "experiments").glob("E*.yaml"))
    grand = 0.0
    for spec in specs:
        rows = project(spec, m)
        total = sum(r["minutes"] for r in rows)
        grand += total
        worst = max(rows, key=lambda r: r["minutes"])
        flag = "  <-- FLAG: a single run over 30 GPU-min" if worst["minutes"] > FLAG_MINUTES else ""
        print(f"{spec.stem:16s} {len(rows):2d} runs  ~{total:6.1f} GPU-min total, "
              f"longest {worst['arm']} ~{worst['minutes']:.1f} min{flag}")
    print(f"{'all':16s}          ~{grand:6.1f} GPU-min (~{grand / 60:.1f} GPU-h)")


if __name__ == "__main__":
    main()
