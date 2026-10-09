import json
from pathlib import Path

import yaml

from rlstudy.experiment import REPO, execute, plan_runs, run_status
from rlstudy.train import load_config


def test_plan_interleaves_reps_and_pairs_seeds():
    spec = {"reps": 2, "arms": {"a": {"x": 1}, "b": None}, "profile_arms": {"p": {"y": 2}}}
    runs = plan_runs(spec)
    assert [(r["arm"], r["rep"], r["seed"]) for r in runs] == [
        ("a", 0, 0), ("b", 0, 0), ("a", 1, 1), ("b", 1, 1), ("p", 0, 0),
    ]
    assert runs[1]["overrides"] == {}


def test_every_experiment_config_resolves():
    # Each arm's overrides must name real config keys, so a typo fails here, not on the GPU.
    from rlstudy.experiment import _set_args

    for path in sorted((REPO / "configs" / "experiments").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        base = REPO / spec["base"]
        defaults = load_config(base, [])
        for r in plan_runs(spec):
            sets = _set_args(r["overrides"])[1::2]
            cfg = load_config(base, sets)
            for key in r["overrides"]:
                node_default, node = defaults, cfg
                for part in key.split("."):
                    assert part in node_default, f"{path.name}: unknown key {key}"
                    node_default, node = node_default[part], node[part]
                assert node == r["overrides"][key]


def test_completed_runs_are_skipped(tmp_path, capsys):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(yaml.safe_dump({"name": "T", "base": "configs/baseline.yaml", "reps": 1, "arms": {"a": {}}}))
    done = tmp_path / "results" / "T" / "a" / "rep0"
    done.mkdir(parents=True)
    (done / "run.json").write_text(json.dumps({"status": "completed"}))
    assert execute(spec_path, tmp_path / "results", 60, dry_run=False, only_arms=None) == 0
    assert "[skip]" in capsys.readouterr().out
    assert not (tmp_path / "results" / "T" / "ledger.jsonl").exists()


def test_run_status(tmp_path):
    assert run_status(tmp_path) is None
    (tmp_path / "run.json").write_text("{}")
    assert run_status(tmp_path) == "incomplete"


def test_dry_run_prints_plan(tmp_path, capsys):
    spec = Path(REPO / "configs/experiments/E2_group_size.yaml")
    execute(spec, tmp_path, 60, dry_run=True, only_arms={"G4"})
    out = capsys.readouterr().out
    assert out.count("[plan]") == 3
    assert "grpo.num_generations=4" in out
