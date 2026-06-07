"""Phase 4 (action-level ARM) -- continuation tokenization alignment (REAL tokenizer).

The highest-risk correctness check, retired on laptop before any GPU time: that
the action continuation includes its </action> terminator, that prefix-overlapping
actions tokenize to distinct (non-prefix) sequences, and that the continuation
reconstructs the generated action span. Uses the real Qwen2.5 tokenizer; skips if
it cannot be loaded offline.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_tokenization_alignment.py
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

from environments.alfworld.alfworld_env import ALFWorldEnvironment  # noqa: E402

_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def env():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_MODEL)
    except Exception as exc:  # offline / not cached
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")
    e = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    e._tokenizer = tok
    return e


_REASONING = "<think>I should open the cabinet to find the mug.</think>"


def test_continuation_includes_terminator(env):
    # The continuation may begin with a seam token (the '>' of </think> re-tokenizes
    # across the </think><action> BPE boundary); that shared token cancels in the
    # softmax over candidates. What matters: the full <action>...</action> span,
    # including the closing terminator, is present.
    cont = env._action_continuation_ids(_REASONING, "look")
    decoded = env._tokenizer.decode(cont)
    assert decoded.endswith("<action>look</action>")
    assert "</action>" in decoded


def test_prefix_overlap_actions_distinct(env):
    """go to cabinet 1 must NOT be a token-prefix of go to cabinet 12 -- the
    </action> terminator after '1' forces divergence."""
    c1 = env._action_continuation_ids(_REASONING, "go to cabinet 1")
    c12 = env._action_continuation_ids(_REASONING, "go to cabinet 12")
    assert c1 != c12
    assert c12[: len(c1)] != c1  # c1 is not a prefix of c12
    assert c1[: len(c12)] != c12  # and vice versa


def test_continuation_reconstructs_generated_span(env):
    """With the policy tokenizing the assistant content as the same tokenizer
    would, the synthesized continuation equals the generated action span."""
    action = "take mug 1 from cabinet 2"
    content = _REASONING + "<action>" + action + "</action>"
    completion_ids = env._tokenizer.encode(content, add_special_tokens=False)
    own_cont = env._action_continuation_ids(_REASONING, action)
    boundary = len(completion_ids) - len(own_cont)
    assert boundary >= 0
    assert completion_ids[boundary:] == own_cont


def test_continuation_distinct_actions_differ(env):
    a = env._action_continuation_ids(_REASONING, "look")
    b = env._action_continuation_ids(_REASONING, "go north")
    assert a != b
