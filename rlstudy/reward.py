"""Verifiable rewards for GSM8K.

Two reward functions, both pure Python on strings (no GPU work, no learned model):

- ``correctness_reward``: 1.0 if the number after the last ``####`` in the
  completion equals the gold answer, else 0.0. Strict: a completion with no
  ``####`` marker scores 0 even if the right number appears somewhere in it.
- ``format_reward``: FORMAT_REWARD if the completion ends with a well-formed
  ``#### <number>`` line, else 0.0.

Both follow TRL's reward-function signature: ``f(prompts, completions, **kwargs)``
returning one float per completion. Dataset columns arrive as keyword arguments,
so the gold answers come in as ``answer``.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

FORMAT_REWARD = 0.1

# A number as models and GSM8K write it: optional sign, optional $, digits with
# optional thousands commas, optional decimal part.
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?|-?\$?\.\d+"
_MARKER_RE = re.compile(r"####\s*(" + _NUMBER + r")")
_FORMAT_RE = re.compile(r"####\s*(?:" + _NUMBER + r")\s*\.?\s*$")


def normalize_number(text: str) -> Decimal | None:
    """Parse a number string into a Decimal, ignoring $, commas, and a trailing period.

    Returns None if the text is not a number. Decimal (not float) so that 0.1 + 0.2
    style rounding never decides whether an answer is correct.
    """
    cleaned = text.strip().rstrip(".").replace(",", "").replace("$", "")
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return value


def extract_answer(completion: str) -> Decimal | None:
    """Return the number after the last ``####`` marker, or None if there is none."""
    matches = _MARKER_RE.findall(completion)
    if not matches:
        return None
    return normalize_number(matches[-1])


def gold_answer(gsm8k_answer_field: str) -> Decimal:
    """Parse the gold number from a GSM8K ``answer`` field (worked solution + ``#### n``)."""
    value = extract_answer(gsm8k_answer_field)
    if value is None:
        raise ValueError(f"GSM8K answer has no #### marker: {gsm8k_answer_field[-80:]!r}")
    return value


def _completion_text(completion) -> str:
    """TRL passes plain strings, or for chat datasets a list of message dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    raise TypeError(f"unsupported completion type: {type(completion).__name__}")


def is_correct(completion: str, gold: Decimal | str) -> bool:
    if isinstance(gold, str):
        gold = normalize_number(gold)
    predicted = extract_answer(completion)
    return predicted is not None and gold is not None and predicted == gold


def has_valid_format(completion: str) -> bool:
    return _FORMAT_RE.search(completion) is not None


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """TRL reward function. ``answer`` holds the gold numbers as strings."""
    return [
        1.0 if is_correct(_completion_text(c), a) else 0.0
        for c, a in zip(completions, answer, strict=True)
    ]


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """TRL reward function for the ``#### <number>`` ending."""
    return [FORMAT_REWARD if has_valid_format(_completion_text(c)) else 0.0 for c in completions]
