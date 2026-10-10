"""Build every writeup figure from whatever results exist; skip (and say so) for the rest.

    python -m rlstudy.figures            # writes results/figures/*.png

Each figure's caption note names the experiment directory it came from, so a figure
can always be traced to its run artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlstudy.plots import breakdown, load_arms, reward_curve, sweep

REPO = Path(__file__).resolve().parent.parent


def _param(arm: str, prefix: str) -> float | None:
    if not arm.startswith(prefix):
        return None
    try:
        return float(arm[len(prefix):])
    except ValueError:
        return None


def build(results: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []

    def have(name: str) -> dict:
        arms = load_arms(results / name) if (results / name).exists() else {}
        if not arms:
            skipped.append(name)
        return arms

    if arms := have("E1_breakdown"):
        timing = {k: v for k, v in arms.items() if "profiled" not in k}
        made.append(breakdown(timing, out / "E1_breakdown.png", "Where a GRPO step's wall-clock time goes",
                              note="Source: results/E1_breakdown. Mean share over reps; warm-up steps dropped."))
        reps = sorted(d for d in (results / "E1_breakdown" / "baseline").glob("rep*") if d.is_dir())
        if reps:
            made.append(reward_curve(reps, out / "E1_reward.png", "Training reward at baseline (E1)"))

    if arms := have("E2_group_size"):
        made.append(breakdown(arms, out / "E2_breakdown.png", "Step-time split vs samples per prompt (G)",
                              note="Source: results/E2_group_size."))
        pts = [(g, a["metrics"]["rollout_tokens_per_s"]) for k, a in arms.items() if (g := _param(k, "G"))]
        made.append(sweep(pts, out / "E2_throughput.png", "Rollout throughput vs samples per prompt",
                          "Samples per prompt (G)", "Generated tokens per second", log_x=True,
                          note="Source: results/E2_group_size."))

    if arms := have("E3_gen_length"):
        made.append(breakdown(arms, out / "E3_breakdown.png", "Step-time split vs generation-length cap",
                              note="Source: results/E3_gen_length."))

    if arms := have("E4_sync_freq"):
        timing = {k: v for k, v in arms.items() if "profiled" not in k}
        made.append(breakdown(timing, out / "E4_breakdown.png", "Step-time split vs weight-sync interval",
                              note="Source: results/E4_sync_freq."))
        pts = [(k_, a["metrics"]["buckets.weight_sync.mean_s_per_step"])
               for k, a in timing.items() if (k_ := _param(k, "sync"))]
        made.append(sweep(pts, out / "E4_sync_per_step.png", "Weight-sync time per training step",
                          "Sync every k steps", "Seconds per step", log_x=True,
                          note="Source: results/E4_sync_freq."))

    if arms := have("E5_backend"):
        made.append(breakdown(arms, out / "E5_breakdown.png", "Step-time split by rollout backend",
                              note="Source: results/E5_backend. Same loop, 64 completions/step, 256-token cap."))

    for name in skipped:
        print(f"TODO: no results yet for {name}")
    return [str(p) for p in made]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "figures")
    args = parser.parse_args()
    for p in build(args.results, args.out):
        print("wrote", p)


if __name__ == "__main__":
    main()
