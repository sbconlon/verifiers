"""Phase 4 (action-level ARM) -- the teacher-forcing scorer helpers.

Tests the pure request-body builder and the by-token-id span summation. The live
/v1/completions call (scored-vs-generation logprob reproduction) is the Phase 9
entry gate, not a laptop test.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_score_continuations.py
"""
import asyncio
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
    ALFWorldEnvironment,
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
    # prompt_logprobs is a TOP-LEVEL field (cluster fix): the raw token_client.post
    # does not merge an "extra_body" key, so nesting it there made vLLM ignore it.
    assert body["prompt_logprobs"] == 1
    assert "extra_body" not in body


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


# ---------------------------------------------------------------------------
# Phase 9: /v1/score route dispatch + fallback
# ---------------------------------------------------------------------------


class _Choice:
    def __init__(self, index, prompt_logprobs):
        self.index = index
        self.prompt_logprobs = prompt_logprobs


class _CompletionsResp:
    def __init__(self, choices):
        self.choices = choices


class _ScoreResp:
    def __init__(self, scores):
        self.scores = scores


class _ScoreClient:
    """Fake verifiers client: token_client.post routes /v1/score and /v1/completions."""

    def __init__(self, scores=None, fail_score=False):
        self.token_client = self
        self.posts = []
        self._scores = scores
        self._fail_score = fail_score

    async def post(self, path, body, cast_to=None):
        self.posts.append((path, body))
        if path == "/v1/score":
            if self._fail_score:
                raise RuntimeError("404 no /v1/score route")
            return _ScoreResp(self._scores)
        if path == "/v1/completions":
            # Minimal full-prompt prompt_logprobs response: logprob 0 at each span pos.
            choices = []
            for i, toks in enumerate(body["prompt"]):
                pl = [None] * len(toks)
                for p in range(len(toks)):
                    pl[p] = {str(toks[p]): {"logprob": 0.0}}
                choices.append(_Choice(i, pl))
            return _CompletionsResp(choices)
        raise ValueError(f"unexpected path {path}")


def _env():
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.use_score_route = True
    return env


def test_score_continuations_uses_score_route():
    env = _env()
    client = _ScoreClient(scores=[-1.0, -2.0])
    prompts = [[1, 2, 3], [1, 2, 4]]
    spans = [(2, 3), (2, 3)]
    out = asyncio.run(env._score_continuations(client, prompts, spans, "m"))
    assert out == [-1.0, -2.0]
    # Hit /v1/score with the prompts + spans (spans as lists), no full-payload call.
    assert client.posts[0][0] == "/v1/score"
    assert client.posts[0][1]["prompts"] == prompts
    assert client.posts[0][1]["spans"] == [[2, 3], [2, 3]]
    assert all(path != "/v1/completions" for path, _ in client.posts)


def test_score_continuations_falls_back_to_completions():
    env = _env()
    client = _ScoreClient(fail_score=True)  # /v1/score raises -> fallback
    prompts = [[1, 2, 3]]
    spans = [(2, 3)]
    out = asyncio.run(env._score_continuations(client, prompts, spans, "m"))
    assert out == [0.0]  # the fake completions response gives logprob 0 over the span
    paths = [path for path, _ in client.posts]
    assert "/v1/score" in paths and "/v1/completions" in paths


def test_score_continuations_route_disabled():
    env = _env()
    env.use_score_route = False
    client = _ScoreClient(scores=[-9.0])  # would be used if /v1/score were called
    prompts = [[1, 2, 3]]
    spans = [(2, 3)]
    out = asyncio.run(env._score_continuations(client, prompts, spans, "m"))
    assert out == [0.0]  # went straight to /v1/completions
    assert [path for path, _ in client.posts] == ["/v1/completions"]
