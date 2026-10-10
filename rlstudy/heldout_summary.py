"""Summarize E6: held-out accuracy of the base model vs GRPO-trained models.

    python -m rlstudy.heldout_summary            # writes results/E6_heldout/summary.json

Every model answers the same 200 problems, so the comparison with the base model is
paired: for each trained model we count problems that flipped wrong -> right and
right -> wrong, and run an exact two-sided McNemar test on the flips (a binomial test
on the discordant pairs). That is fairer than comparing two percentages as if they
came from different problem sets.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "E6_heldout"
METRICS = ("strict_accuracy", "lenient_accuracy", "format_rate")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for b vs c discordant pairs under p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(base: dict, other: dict, key: str) -> dict:
    a = {i["id"]: i[key] for i in base["items"]}
    b = {i["id"]: i[key] for i in other["items"]}
    assert a.keys() == b.keys(), "models were scored on different problems"
    gained = sum(1 for k in a if not a[k] and b[k])
    lost = sum(1 for k in a if a[k] and not b[k])
    return {"gained": gained, "lost": lost, "net": gained - lost, "mcnemar_p": mcnemar_exact(gained, lost)}


def _stats(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "stdev": statistics.stdev(xs) if len(xs) > 1 else None,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def summarize(root: Path = ROOT) -> dict:
    base = json.loads((root / "base" / "eval.json").read_text())
    out = {"base": {m: base[m] for m in METRICS}, "n_problems": base["n"], "arms": {}}
    for arm_dir in sorted(p for p in root.glob("trained_*") if p.is_dir()):
        reps = sorted(arm_dir.glob("rep*/eval.json"))
        evals = [json.loads(p.read_text()) for p in reps]
        arm = {m: _stats([e[m] for e in evals]) for m in METRICS}
        arm["reps"] = [str(p.parent.relative_to(root)) for p in reps]
        arm["paired_vs_base"] = {
            key: [paired(base, e, key) for e in evals] for key in ("strict", "lenient")
        }
        out["arms"][arm_dir.name] = arm
    (root / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    s = summarize()
    b = s["base"]
    print(f"base: strict {b['strict_accuracy']:.3f} lenient {b['lenient_accuracy']:.3f} format {b['format_rate']:.3f}")
    for name, a in s["arms"].items():
        f = lambda m: f"{a[m]['mean']:.3f}" + (f"±{a[m]['stdev']:.3f}" if a[m]["stdev"] is not None else "")
        print(f"{name} (n={a['strict_accuracy']['n']}): strict {f('strict_accuracy')} lenient {f('lenient_accuracy')} "
              f"format {f('format_rate')}")
        for key in ("strict", "lenient"):
            print(f"   {key} vs base: " + "; ".join(
                f"+{p['gained']}/-{p['lost']} (p={p['mcnemar_p']:.3g})" for p in a["paired_vs_base"][key]))


if __name__ == "__main__":
    main()
