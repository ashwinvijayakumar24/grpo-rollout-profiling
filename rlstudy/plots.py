"""Charts for the writeup, drawn only from arm_summary.json / trace files.

- ``breakdown``: 100% horizontal stacked bar of step time per stage, one bar per arm.
  (A stacked bar instead of a pie: segment sizes are easier to compare, and several
  arms fit in one figure.)
- ``sweep``: one metric vs the swept parameter, mean with min-max whiskers over reps.
- ``reward_curve``: reward per step, mean over reps with a min-max band.

Colors are fixed per stage (never by rank) from a palette validated for color-vision
deficiency; three hues are under 3:1 contrast on white, so segments carry direct
percent labels and every figure has a table (summary.md) next to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = ("rollout_gen", "advantage_loss", "weight_sync", "optimizer_step", "reward", "other")
LABELS = {
    "rollout_gen": "Rollout generation",
    "advantage_loss": "Advantage + loss + backward",
    "weight_sync": "Weight sync",
    "optimizer_step": "Optimizer step",
    "reward": "Reward",
    "other": "Other",
}
COLORS = {
    "rollout_gen": "#2a78d6",
    "advantage_loss": "#eb6834",
    "weight_sync": "#1baf7a",
    "optimizer_step": "#eda100",
    "reward": "#e87ba4",
    "other": "#a8a7a2",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SERIES = "#2a78d6"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _metric(agg: dict, key: str) -> dict | None:
    return agg["metrics"].get(key)


def breakdown(arms: dict[str, dict], out: Path, title: str, note: str = "") -> Path:
    names = list(arms)
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 1.9), facecolor=SURFACE)
    _style(ax)
    for i, arm in enumerate(names):
        left = 0.0
        for st in STAGES:
            m = _metric(arms[arm], f"buckets.{st}.share")
            share = 100 * m["mean"] if m else 0.0
            # 2px surface gap between segments via the edge line.
            ax.barh(i, share, left=left, height=0.6, color=COLORS[st], edgecolor=SURFACE, linewidth=2)
            if share >= 6:
                ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="white" if st in ("rollout_gen", "advantage_loss") else INK)
            left += share
    ax.set_yticks(range(len(names)), labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of step wall-clock time (%)", color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in STAGES]
    ax.legend(handles, [LABELS[s] for s in STAGES], ncol=3, frameon=False, fontsize=8.5,
              loc="upper left", bbox_to_anchor=(0, -0.28 if len(names) < 3 else -0.18), labelcolor=INK_2)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def sweep(points: list[tuple[float, dict]], out: Path, title: str, xlabel: str, ylabel: str,
          scale: float = 1.0, log_x: bool = False, note: str = "") -> Path:
    """points: (x, stats dict with mean/min/max/n) per arm."""
    pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    mean = [p[1]["mean"] * scale for p in pts]
    lo = [m - p[1]["min"] * scale for m, p in zip(mean, pts)]
    hi = [p[1]["max"] * scale - m for m, p in zip(mean, pts)]
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.errorbar(xs, mean, yerr=[lo, hi], color=SERIES, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, elinewidth=1)
    for x, m, p in zip(xs, mean, pts):
        ax.annotate(f"{m:,.0f}" if abs(m) >= 100 else f"{m:.3g}", (x, m), textcoords="offset points", xytext=(9, 4), ha="left",
                    fontsize=8.5, color=INK)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs, labels=[f"{x:g}" for x in xs])
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0, top=max(m + h for m, h in zip(mean, hi)) * 1.18)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    n = {p[1]["n"] for p in pts}
    fig.text(0.01, 0.005, (note + "  " if note else "") + f"Points: mean of n={'/'.join(map(str, sorted(n)))} reps; whiskers: min-max.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def reward_curve(run_dirs: list[Path], out: Path, title: str, key: str = "reward/correctness_reward") -> Path:
    from rlstudy.trace import read_trace

    series = []
    for d in run_dirs:
        _, steps = read_trace(d)
        series.append([s["metrics"].get(key) for s in steps])
    n = min(len(s) for s in series)
    import statistics

    xs = list(range(n))
    mean = [statistics.fmean(s[i] for s in series) for i in xs]
    lo = [min(s[i] for s in series) for i in xs]
    hi = [max(s[i] for s in series) for i in xs]
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.fill_between(xs, lo, hi, color=SERIES, alpha=0.15, linewidth=0)
    ax.plot(xs, mean, color=SERIES, linewidth=2)
    ax.set_xlabel("Training step", color=INK_2, fontsize=9)
    ax.set_ylabel("Fraction of samples correct", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005, f"Line: mean over {len(series)} reps; band: min-max. Per-step batch mean, not a held-out eval.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def _natural_key(name: str):
    """Sort arm names like G4 < G8 < G16 and len128 < len512, not alphabetically."""
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def load_arms(exp_dir: Path) -> dict[str, dict]:
    paths = sorted(Path(exp_dir).glob("*/arm_summary.json"), key=lambda p: _natural_key(p.parent.name))
    return {p.parent.name: json.loads(p.read_text()) for p in paths}
