import os
import sys

import pytest

from rlstudy.evaluate import score


def test_score_separates_format_from_math():
    rows = [{"id": "a", "answer": "72"}, {"id": "b", "answer": "5"}, {"id": "c", "answer": "9"}]
    texts = ["work\n#### 72", "the answer is 5.", "#### 8"]
    r = score(rows, texts)
    assert r["strict_accuracy"] == pytest.approx(1 / 3)
    assert r["lenient_accuracy"] == pytest.approx(2 / 3)
    assert r["format_rate"] == pytest.approx(2 / 3)
    assert [i["strict"] for i in r["items"]] == [True, False, False]


@pytest.mark.slow
def test_evaluate_end_to_end_hf(tmp_path):
    if sys.platform == "darwin":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    import rlstudy.evaluate as ev

    ev.MAX_TOKENS = 4
    r = ev.evaluate("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", tmp_path, backend="hf", limit=3)
    assert r["n"] == 3 and len(r["items"]) == 3
    assert (tmp_path / "eval.json").exists()
