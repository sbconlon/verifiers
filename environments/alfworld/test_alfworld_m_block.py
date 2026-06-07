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
