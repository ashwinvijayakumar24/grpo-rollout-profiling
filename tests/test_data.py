import json

import pytest

from rlstudy.data import DEFAULT_DIR, SYSTEM_PROMPT, load_split, make_prompt
from rlstudy.reward import normalize_number


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DEFAULT_DIR / "manifest.json").read_text())


def test_manifest_pins_revision_and_seed(manifest):
    assert manifest["dataset"] == "openai/gsm8k"
    assert len(manifest["revision"]) == 40  # a full git commit sha
    assert manifest["seed"] == 0
    assert manifest["system_prompt"] == SYSTEM_PROMPT


@pytest.mark.parametrize("split, n", [("train", 1000), ("eval", 200)])
def test_split_sizes_match_manifest(manifest, split, n):
    rows = load_split(split)
    assert len(rows) == n == manifest["files"][split]["rows"]


def test_ids_are_unique_and_match_indices(manifest):
    train = load_split("train")
    assert [r["id"] for r in train] == [f"train-{i}" for i in manifest["train_indices"]]
    eval_ = load_split("eval")
    assert [r["id"] for r in eval_] == [f"test-{i}" for i in manifest["eval_indices"]]


def test_every_gold_answer_parses():
    for split in ("train", "eval"):
        for row in load_split(split):
            assert normalize_number(row["answer"]) is not None, row["id"]


def test_prompt_shape():
    row = load_split("train", limit=1)[0]
    assert row["prompt"] == make_prompt(row["prompt"][1]["content"])
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]


def test_tampered_file_is_rejected(tmp_path):
    for name in ("manifest.json", "train.jsonl", "eval.jsonl"):
        (tmp_path / name).write_bytes((DEFAULT_DIR / name).read_bytes())
    with (tmp_path / "train.jsonl").open("a") as f:
        f.write('{"id": "extra"}\n')
    with pytest.raises(RuntimeError, match="checksum"):
        load_split("train", tmp_path)


def test_limit():
    assert len(load_split("train", limit=8)) == 8
