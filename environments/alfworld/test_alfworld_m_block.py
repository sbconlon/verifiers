"""Phase 3 (action-level ARM) -- m-reasoning-block generation in get_model_response.

Drives the override with a fake base (counts generations, records prompts,
returns distinct responses). No vLLM: the real concurrency / prefix-cache win is
first exercised at Phase 9. Verifies passthrough at m=1, m generations + stash at
m>1, j* selection/determinism/uniformity, the shared-prompt property, and that
the transient stash is cleaned up.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_alfworld_m_block.py
"""
import asyncio
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

# Mock verifiers before importing alfworld_env (mirrors test_prompting.py), with a
# real fake base providing get_model_response and an identity cleanup decorator so
# cleanup_alf_env stays a callable coroutine.
_vf_mock = MagicMock()


class _Resp:
    def __init__(self, i: int):
        self.id = i


class _FakeMultiTurnEnv:
    async def get_model_response(self, state, prompt, *args, **kwargs):
        self._calls = getattr(self, "_calls", 0) + 1
        self._prompts = getattr(self, "_prompts", [])
        self._prompts.append(prompt)
        return _Resp(self._calls)


_vf_mock.MultiTurnEnv = _FakeMultiTurnEnv
_vf_mock.cleanup = lambda f: f  # identity decorator -> cleanup_alf_env stays callable
sys.modules["verifiers"] = _vf_mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force a fresh import of alfworld_env bound to THIS file's mock (see the matching
# note in test_alfworld_extras.py): the module is cached after the first alfworld
# test module imports it, and other modules install a base class without
# get_model_response. Popping isolates this file's binding.
sys.modules.pop("environments.alfworld.alfworld_env", None)

from environments.alfworld.alfworld_env import ALFWorldEnvironment  # noqa: E402


def _env(m: int) -> ALFWorldEnvironment:
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.num_reasoning_blocks = m
    # These tests exercise the m-separate fallback path (the fake base provides
    # get_model_response). n=m fusion is covered separately in
    # test_alfworld_nm_fusion.py with a fake client.
    env.use_nm_fusion = False
    return env


def _state(trajectory_id="t0", turn=0):
    # turn == len(trajectory) at get_model_response time (step appended afterwards)
    return {"trajectory_id": trajectory_id, "trajectory": [None] * turn}


def test_m1_passthrough():
    env = _env(1)
    state = _state()
    resp = asyncio.run(env.get_model_response(state, prompt=["P"]))
    assert isinstance(resp, _Resp)
    assert env._calls == 1
    assert "_pending_reasoning_blocks" not in state
    assert "_executed_block_idx" not in state


def test_m_block_generates_m():
    env = _env(4)
    state = _state()
    asyncio.run(env.get_model_response(state, prompt=["P"]))
    assert env._calls == 4
    assert len(state["_pending_reasoning_blocks"]) == 4


def test_executed_is_jstar():
    env = _env(4)
    state = _state()
    resp = asyncio.run(env.get_model_response(state, prompt=["P"]))
    j_star = state["_executed_block_idx"]
    assert resp is state["_pending_reasoning_blocks"][j_star]


def test_jstar_seeded_deterministic():
    # Same (trajectory_id, turn) -> same j* across independent calls.
    j_first = []
    for _ in range(2):
        env = _env(8)
        state = _state(trajectory_id="abc", turn=3)
        asyncio.run(env.get_model_response(state, prompt=["P"]))
        j_first.append(state["_executed_block_idx"])
    assert j_first[0] == j_first[1]


def test_jstar_varies_across_turns():
    # Different turns within a rollout draw independently (not pinned to one value).
    picks = set()
    for turn in range(20):
        env = _env(8)
        state = _state(trajectory_id="same", turn=turn)
        asyncio.run(env.get_model_response(state, prompt=["P"]))
        picks.add(state["_executed_block_idx"])
    assert len(picks) > 1


def test_jstar_uniform():
    m = 4
    counts = Counter()
    for i in range(400):
        env = _env(m)
        state = _state(trajectory_id=f"traj-{i}", turn=0)
        asyncio.run(env.get_model_response(state, prompt=["P"]))
        counts[state["_executed_block_idx"]] += 1
    # All buckets hit, none wildly off (loose bound: every bucket within 2x of mean).
    assert set(counts) == set(range(m))
    mean = 400 / m
    for b in range(m):
        assert 0.5 * mean < counts[b] < 1.5 * mean


def test_same_prompt_each_call():
    env = _env(4)
    state = _state()
    prompt = ["the shared observation o"]
    asyncio.run(env.get_model_response(state, prompt=prompt))
    assert len(env._prompts) == 4
    for p in env._prompts:
        assert p is prompt  # identical object -> basis for prefix caching


def test_cleanup_pops_pending_stash():
    env = _env(4)
    state = _state()
    asyncio.run(env.get_model_response(state, prompt=["P"]))
    assert "_pending_reasoning_blocks" in state
    asyncio.run(env.cleanup_alf_env(state))
    assert "_pending_reasoning_blocks" not in state
    assert "_executed_block_idx" not in state


# ---------------------------------------------------------------------------
# Phase 8/9: n=m generation fusion (one request, m choices)
# ---------------------------------------------------------------------------


class _FakeNative:
    """Stand-in for the openai ChatCompletion returned by get_native_response."""

    def __init__(self, choices):
        self.choices = choices

    def model_copy(self, update):
        return _FakeNative(update["choices"])


class _FakeMsg:
    def __init__(self):
        self.tokens = object()  # non-None -> passes the per-block token check


class _FakeResp:
    def __init__(self, tag):
        self.message = _FakeMsg()
        self.tag = tag


class _FakeClient:
    """Mimics the verifiers client methods _generate_m_blocks_fused calls."""

    def __init__(self, n_returned=None):
        self.calls = 0
        self.last_sampling = None
        self._n_returned = n_returned  # None -> honor n; else force this many choices

    def _build_state_headers(self, state):
        return None

    async def to_native_prompt(self, prompt):
        return prompt, {}

    async def to_native_tools(self, tools):
        return tools

    async def get_native_response(self, native_prompt, model, sampling_args, native_tools, **kw):
        self.calls += 1
        self.last_sampling = dict(sampling_args)
        n = self._n_returned if self._n_returned is not None else sampling_args.get("n", 1)
        return _FakeNative([f"c{k}" for k in range(n)])

    async def raise_from_native_response(self, single):
        return None

    async def from_native_response(self, single):
        return _FakeResp(single.choices[0])


def _fusion_state(client, **overrides):
    s = _state()
    s.update({"client": client, "model": "m", "sampling_args": {"temperature": 1.0}})
    s.update(overrides)
    return s


def test_nm_fusion_single_request():
    """m>1 with fusion on issues ONE n=m request (not m), stashes m blocks."""
    env = _env(4)
    env.use_nm_fusion = True
    client = _FakeClient()
    state = _fusion_state(client)
    resp = asyncio.run(env.get_model_response(state, prompt=["P"]))
    assert client.calls == 1  # ONE request, not m
    assert client.last_sampling["n"] == 4
    assert getattr(env, "_calls", 0) == 0  # the m-separate fallback was NOT used
    assert len(state["_pending_reasoning_blocks"]) == 4
    assert resp is state["_pending_reasoning_blocks"][state["_executed_block_idx"]]


def test_nm_fusion_falls_back_when_server_ignores_n():
    """If the server returns < m choices (n not honored), fall back to m separate
    generations -- correct, just slow."""
    env = _env(4)
    env.use_nm_fusion = True
    client = _FakeClient(n_returned=1)  # only 1 choice -> fusion bails
    state = _fusion_state(client)
    asyncio.run(env.get_model_response(state, prompt=["P"]))
    assert client.calls == 1  # tried fusion once
    assert env._calls == 4  # then fell back to m separate (fake base called 4x)
    assert len(state["_pending_reasoning_blocks"]) == 4
