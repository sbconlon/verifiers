"""Phase 4 (action-level ARM) -- the teacher-forcing scorer helpers.

Tests the pure request-body builder and the by-token-id span summation. The live
/v1/completions call (scored-vs-generation logprob reproduction) is the Phase 9
entry gate, not a laptop test.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_score_continuations.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()
_vf_mock.MultiTurnEnv = object
sys.modules["verifiers"] = _vf_mock
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.modules.pop("environments.alfworld.alfworld_env", None)

from environments.alfworld.alfworld_env import (  # noqa: E402
    _build_score_body,
    _span_logprob_sum,
)


def test_build_score_body_shape():
    body = _build_score_body([[1, 2, 3], [1, 2, 4]], model="qwen", temperature=1.0)
    assert body["model"] == "qwen"
    assert body["prompt"] == [[1, 2, 3], [1, 2, 4]]
    assert body["max_tokens"] == 1
    assert body["temperature"] == 1.0
    assert body["top_p"] == 1.0
    assert body["extra_body"] == {"prompt_logprobs": 1}


def test_span_sum_by_token_id():
    # prompt token-ids; we score positions [1, 3) -> tokens 20 and 30.
    prompt_token_ids = [10, 20, 30, 40]
    prompt_logprobs = [
        None,  # position 0 has no logprob
        {"20": {"logprob": -0.5}, "999": {"logprob": -0.01}},  # actual token is 20, not the top one
        {"30": {"logprob": -0.25}},
        {"40": {"logprob": -5.0}},  # outside the span, must be ignored
    ]
    total = _span_logprob_sum(prompt_logprobs, prompt_token_ids, start=1, end=3)
    assert total == pytest.approx(-0.75)


def test_span_sum_reads_actual_token_not_rank0():
    """The actual token may not be rank 0; summation must key by token-id."""
    prompt_token_ids = [7, 8]
    prompt_logprobs = [
        None,
        {"5": {"logprob": -0.1}, "8": {"logprob": -2.0}},  # actual is 8 (the worse one)
    ]
    total = _span_logprob_sum(prompt_logprobs, prompt_token_ids, start=1, end=2)
    assert total == pytest.approx(-2.0)


def test_span_sum_object_entry():
    """Entry may be an object with a .logprob attribute rather than a dict."""

    class _LP:
        def __init__(self, lp):
            self.logprob = lp

    prompt_token_ids = [1, 2]
    prompt_logprobs = [None, {"2": _LP(-0.3)}]
    total = _span_logprob_sum(prompt_logprobs, prompt_token_ids, start=1, end=2)
    assert total == pytest.approx(-0.3)
