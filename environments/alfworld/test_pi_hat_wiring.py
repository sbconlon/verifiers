"""Phase 4 (action-level ARM) -- pi_hat wiring in add_trajectory_step.

Drives the scoring branch with a fake scorer (so the RBMC leave-one-out math is
isolated from tokenization/HTTP) and a trivial fake tokenizer. Verifies the LOO
estimate, the m=1 / error-turn skips, the inadmissible-a* union, and stash cleanup.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_pi_hat_wiring.py
"""
import asyncio
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()


class _FakeMultiTurnEnv:
    async def add_trajectory_step(self, state, trajectory_step):
        state["trajectory"].append(trajectory_step)


class _AssistantMessage:
    def __init__(self, content=""):
        self.role = "assistant"
        self.content = content


_vf_mock.MultiTurnEnv = _FakeMultiTurnEnv
_vf_mock.AssistantMessage = _AssistantMessage
_vf_mock.cleanup = lambda f: f
sys.modules["verifiers"] = _vf_mock
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.modules.pop("environments.alfworld.alfworld_env", None)

from environments.alfworld.alfworld_env import ALFWorldEnvironment  # noqa: E402


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _Toks:
    def __init__(self, prompt_ids, completion_ids):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids


class _Msg:
    def __init__(self, content, tokens):
        self.content = content
        self.tokens = tokens


class _Block:
    def __init__(self, content):
        self.message = _Msg(content, _Toks([1, 2], list(range(40))))


def _env(m, fake_scorer=None):
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.num_reasoning_blocks = m
    env._tokenizer = _FakeTok()
    if fake_scorer is not None:
        env._score_continuations = fake_scorer
    return env


def _completion(action):
    return [_AssistantMessage(f"<think>r</think><action>{action}</action>")]


def _scorer_from_conditionals(per_block_probs):
    """Fake _score_continuations. Phase 9 consolidation: _compute_pi_hat now makes
    ONE call with all m x |A| prompts (blocks concatenated in order), so return the
    flat concatenation of log(probs) -- _compute_pi_hat splits it back per block and
    conditional_renorm reproduces each block's `probs` exactly."""

    async def fake(client, prompts, spans, model, temperature=1.0):
        out: list[float] = []
        for block in per_block_probs:
            out.extend(math.log(max(p, 1e-12)) for p in block)
        return out

    return fake


def test_pi_hat_loo_at_astar():
    # admissible includes a* ("look") at index 1; candidates == admissible.
    admissible = ["go", "look", "take"]
    # cond[a*]=probs[1] per block; j*=0 excluded -> mean of blocks 1,2,3.
    per_block = [
        [0.1, 0.8, 0.1],
        [0.3, 0.5, 0.2],
        [0.25, 0.25, 0.5],
        [0.4, 0.6, 0.0],
    ]
    env = _env(4, fake_scorer=_scorer_from_conditionals(per_block))
    state = {
        "trajectory": [],
        "_last_admissible_commands": admissible,
        "_pending_reasoning_blocks": [_Block("<think>r</think><action>look</action>") for _ in range(4)],
        "_executed_block_idx": 0,
        "client": object(),
        "model": "m",
    }
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    expected = (0.5 + 0.25 + 0.6) / 3
    assert step["extras"]["pi_hat"] == pytest.approx(expected)
    assert step["extras"]["executed_action"] == "look"
    assert step["extras"]["admissible_actions"] == admissible
    assert "_pending_reasoning_blocks" not in state
    assert "_executed_block_idx" not in state


def test_pi_hat_single_scoring_request():
    """Phase 9 consolidation: all m x |A| scoring prompts go in ONE request, not
    one per block (the m x request multiplier that flooded the vLLM queue)."""
    admissible = ["go", "look", "take"]  # |A| = 3
    calls = {"n": 0, "n_prompts": 0}

    async def counting_scorer(client, prompts, spans, model, temperature=1.0):
        calls["n"] += 1
        calls["n_prompts"] = len(prompts)
        return [0.0] * len(prompts)  # uniform; value irrelevant to this test

    env = _env(4, fake_scorer=counting_scorer)
    state = {
        "trajectory": [],
        "_last_admissible_commands": admissible,
        "_pending_reasoning_blocks": [_Block("<think>r</think><action>look</action>") for _ in range(4)],
        "_executed_block_idx": 0,
        "client": object(),
        "model": "m",
    }
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert calls["n"] == 1  # ONE request for the whole decision point
    assert calls["n_prompts"] == 4 * 3  # m x |A|
    assert "pi_hat" in step["extras"]


class _FakeTokenClient:
    def __init__(self):
        self.posts = []

    async def post(self, path, body, cast_to=None):
        self.posts.append(body)

        class _Empty:
            choices: list = []

        return _Empty()


class _FakeClientWithToken:
    def __init__(self):
        self.token_client = _FakeTokenClient()


def test_pi_hat_prewarms_prefixes_once():
    """Phase 9 fix: each [o, R_j] prefix is pre-warmed into the cache in ONE batched
    request (m prefixes) before the action scoring, so the batched scoring reuses it
    instead of recomputing the ~2000-token prefix per action."""
    admissible = ["go", "look", "take"]
    score_calls = {"n": 0}

    async def scorer(client, prompts, spans, model, temperature=1.0):
        score_calls["n"] += 1
        return [0.0] * len(prompts)

    fake_client = _FakeClientWithToken()
    env = _env(4, fake_scorer=scorer)
    state = {
        "trajectory": [],
        "_last_admissible_commands": admissible,
        "_pending_reasoning_blocks": [_Block("<think>r</think><action>look</action>") for _ in range(4)],
        "_executed_block_idx": 0,
        "client": fake_client,
        "model": "m",
    }
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    # One pre-warm request carrying the m=4 [o, R_j] prefixes, then one scoring call.
    assert len(fake_client.token_client.posts) == 1
    assert len(fake_client.token_client.posts[0]["prompt"]) == 4
    assert "prompt_logprobs" not in fake_client.token_client.posts[0]  # cheap, no logprobs
    assert score_calls["n"] == 1


def test_pi_hat_prewarm_disabled():
    """prewarm_pi_hat_prefixes=False skips the pre-warm (A/B knob)."""
    score_calls = {"n": 0}

    async def scorer(client, prompts, spans, model, temperature=1.0):
        score_calls["n"] += 1
        return [0.0] * len(prompts)

    fake_client = _FakeClientWithToken()
    env = _env(3, fake_scorer=scorer)
    env.prewarm_pi_hat_prefixes = False
    state = {
        "trajectory": [],
        "_last_admissible_commands": ["go", "look"],
        "_pending_reasoning_blocks": [_Block("<think>r</think><action>look</action>") for _ in range(3)],
        "_executed_block_idx": 0,
        "client": fake_client,
        "model": "m",
    }
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert len(fake_client.token_client.posts) == 0  # no pre-warm
    assert score_calls["n"] == 1


def test_m1_no_scoring():
    env = _env(1)
    state = {"trajectory": [], "_last_admissible_commands": ["look"]}
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert "pi_hat" not in step["extras"]
    assert step["extras"]["executed_action"] == "look"


def test_inadmissible_astar_unioned_and_scored():
    admissible = ["go", "take"]  # a* "look" absent -> appended at idx 2
    per_block = [
        [0.2, 0.2, 0.6],
        [0.1, 0.1, 0.8],
        [0.3, 0.3, 0.4],
    ]
    env = _env(3, fake_scorer=_scorer_from_conditionals(per_block))
    state = {
        "trajectory": [],
        "_last_admissible_commands": admissible,
        "_pending_reasoning_blocks": [_Block("<think>r</think><action>look</action>") for _ in range(3)],
        "_executed_block_idx": 1,  # exclude block 1
        "client": object(),
        "model": "m",
    }
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    expected = (0.6 + 0.4) / 2  # blocks 0 and 2, cond[a*]=probs[2]
    assert step["extras"]["pi_hat"] == pytest.approx(expected)


def test_error_turn_skips_scoring():
    env = _env(4, fake_scorer=_scorer_from_conditionals([[1.0]] * 4))
    state = {
        "trajectory": [],
        "_last_admissible_commands": ["look"],
        "_pending_reasoning_blocks": [_Block("<think>r</think><action>look</action>") for _ in range(4)],
        "_executed_block_idx": 0,
        "client": object(),
        "model": "m",
        "error": RuntimeError("boom"),
    }
    step = {"completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert "pi_hat" not in step["extras"]
    # stash still cleaned even on the skip path
    assert "_pending_reasoning_blocks" not in state
    assert "_executed_block_idx" not in state
