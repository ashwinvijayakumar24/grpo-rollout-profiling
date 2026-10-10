"""Plots render from synthetic summaries (charts are checked by eye separately)."""

from rlstudy.analyze import analyze_experiment
from rlstudy.plots import breakdown, load_arms, reward_curve, sweep
from tests.test_analyze import fake_step, write_run


def test_plots_render(tmp_path):
    exp = tmp_path / "E2"
    for arm, rollout in (("G4", 2.0), ("G8", 3.0), ("G16", 5.0)):
        for rep in range(3):
            write_run(exp / arm / f"rep{rep}",
                      [fake_step(i, rollout=rollout + 0.1 * rep, r=min(1.0, i / 10)) for i in range(8)])
    analyze_experiment(exp)
    arms = load_arms(exp)
    assert set(arms) == {"G4", "G8", "G16"}
    p1 = breakdown(arms, tmp_path / "b.png", "Synthetic breakdown", note="synthetic data")
    pts = [(int(a[1:]), arms[a]["metrics"]["rollout_tokens_per_s"]) for a in arms]
    p2 = sweep(pts, tmp_path / "s.png", "Synthetic sweep", "G", "tok/s", log_x=True)
    p3 = reward_curve(sorted((exp / "G8").glob("rep*")), tmp_path / "r.png", "Synthetic reward")
    for p in (p1, p2, p3):
        assert p.stat().st_size > 5000


def test_arm_order_is_natural():
    from rlstudy.plots import _natural_key

    assert sorted(["G16", "G4", "G8"], key=_natural_key) == ["G4", "G8", "G16"]
    assert sorted(["sync16", "sync1", "sync4"], key=_natural_key) == ["sync1", "sync4", "sync16"]
