from decimal import Decimal

import pytest

from rlstudy.reward import (
    FORMAT_REWARD,
    correctness_reward,
    extract_answer,
    format_reward,
    gold_answer,
    has_valid_format,
    is_correct,
    normalize_number,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("72", Decimal(72)),
        ("1,000", Decimal(1000)),
        ("$18", Decimal(18)),
        ("-5", Decimal(-5)),
        ("3.50", Decimal("3.5")),
        ("42.", Decimal(42)),
        (" 7 ", Decimal(7)),
        ("abc", None),
        ("", None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


def test_gold_answer_parses_gsm8k_field():
    field = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n48+24 = <<48+24=72>>72\n#### 72"
    assert gold_answer(field) == Decimal(72)


def test_gold_answer_with_commas():
    assert gold_answer("... so the total is 1,250.\n#### 1,250") == Decimal(1250)


def test_gold_answer_requires_marker():
    with pytest.raises(ValueError):
        gold_answer("the answer is 5")


def test_extract_uses_last_marker():
    # A model that restates a wrong intermediate answer and then corrects itself is
    # graded on its final marker.
    assert extract_answer("#### 10\nWait, I made an error.\n#### 12") == Decimal(12)


def test_extract_without_marker_is_none():
    assert extract_answer("The answer is 72.") is None


@pytest.mark.parametrize(
    "completion, gold, correct",
    [
        ("so she sold 72 clips.\n#### 72", "72", True),
        ("#### 72.0", "72", True),          # numeric equality, not string equality
        ("#### $1,000", "1000", True),
        ("#### 72.", "72", True),           # trailing sentence period
        ("#### -3", "-3", True),
        ("#### 0.5", ".5", True),
        ("#### 71", "72", False),
        ("The answer is 72", "72", False),  # strict: no marker means no credit
        ("####", "72", False),
        ("#### seventy-two", "72", False),
    ],
)
def test_is_correct(completion, gold, correct):
    assert is_correct(completion, gold) is correct


@pytest.mark.parametrize(
    "completion, ok",
    [
        ("work...\n#### 72", True),
        ("work...\n#### 72\n", True),
        ("work...\n####72", True),
        ("#### 1,234.5.", True),
        ("#### 72 because reasons", False),  # text after the number
        ("#### 72\nMore text after.", False),
        ("72", False),
    ],
)
def test_has_valid_format(completion, ok):
    assert has_valid_format(completion) is ok


def test_trl_signature_plain_strings():
    prompts = ["q1", "q2", "q3"]
    completions = ["#### 4", "#### 5", "four"]
    answers = ["4", "4", "4"]
    assert correctness_reward(prompts, completions, answer=answers) == [1.0, 0.0, 0.0]
    assert format_reward(prompts, completions, answer=answers) == [FORMAT_REWARD, FORMAT_REWARD, 0.0]


def test_trl_signature_chat_messages():
    completions = [[{"role": "assistant", "content": "2+2=4\n#### 4"}]]
    assert correctness_reward(["q"], completions, answer=["4"]) == [1.0]
    assert format_reward(["q"], completions) == [FORMAT_REWARD]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        correctness_reward(["q"], ["#### 1", "#### 2"], answer=["1"])
