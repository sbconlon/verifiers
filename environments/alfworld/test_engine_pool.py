"""Engine reuse (engine_reuse=True): the per-process pool of TextWorld engines.

No TextWorld needed: the engines are fakes. What is pinned here is the pool's
contract: a pooled engine is pointed at the next game, an engine with a call in
flight is never reused, the pool is keyed by oracle mode, and with the option off
nothing changes (a fresh engine per rollout, closed at the end).
"""

import asyncio
import itertools

import pytest

try:
    import alfworld_env as ae
except ImportError:  # in-repo layout
    from environments.alfworld import alfworld_env as ae

ALFWorldEnvironment = ae.ALFWorldEnvironment


class _Engine:
    """Stands in for textworld's gym env: reset() loads the next file of its iterator."""

    made = 0

    def __init__(self, game_file):
        type(self).made += 1
        self.serial = type(self).made
        self._gamefiles_iterator = itertools.cycle([game_file])
        self.loaded = None
        self.resets = 0
        self.closed = False
        self.fail_next = None

    def reset(self):
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        self.loaded = next(self._gamefiles_iterator)
        self.resets += 1
        return [f"obs of {self.loaded}"], {"admissible_commands": [["look"]]}

    def step(self, commands):
        if commands == ["boom"]:
            raise RuntimeError("planner died")
        return [f"did {commands[0]} in {self.loaded}"], [0], [False], {"admissible_commands": [["look"]]}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    monkeypatch.setattr(ae, "_TW_ENGINE_POOL", {})
    monkeypatch.setattr(ae, "_TW_ENGINE_STATS", {"made": 0, "reused": 0, "returned": 0, "discarded": 0})
    _Engine.made = 0


def _env(engine_reuse, oracle="off"):
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.engine_reuse = engine_reuse
    env.oracle = oracle
    env._make_tw_env = lambda game_file: _Engine(game_file)
    return env


def _rollout(env, game_file, steps=("look",)):
    """What a rollout does to its engine: acquire + reset, some steps, release."""

    async def go():
        tw_env, obs, _infos = await env._reset_engine(game_file)
        for a in steps:
            env._engine_call(tw_env, "step", [a])
        await env.cleanup_alf_env({"alf_env": tw_env})
        return tw_env, obs[0]

    return asyncio.run(go())


def test_off_by_default_a_fresh_engine_per_rollout_closed_at_the_end():
    env = _env(engine_reuse=False)
    e1, obs1 = _rollout(env, "g1")
    e2, obs2 = _rollout(env, "g1")
    assert e1 is not e2 and e1.closed and e2.closed
    assert (obs1, obs2) == ("obs of g1", "obs of g1")
    assert ae._TW_ENGINE_POOL == {} and ae._TW_ENGINE_STATS["made"] == 0


def test_env_built_without_init_behaves_as_off():
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env._make_tw_env = lambda game_file: _Engine(game_file)
    e1, _ = _rollout(env, "g1")
    assert e1.closed and ae._TW_ENGINE_POOL == {}


def test_reuse_points_the_pooled_engine_at_the_next_game():
    env = _env(engine_reuse=True)
    e1, obs1 = _rollout(env, "g1")
    assert not e1.closed and ae._TW_ENGINE_POOL["off"] == [e1]
    e2, obs2 = _rollout(env, "g2")
    assert e2 is e1 and (obs1, obs2) == ("obs of g1", "obs of g2")
    e3, obs3 = _rollout(env, "g1")
    assert e3 is e1 and obs3 == "obs of g1" and e1.resets == 3
    assert _Engine.made == 1
    assert ae._TW_ENGINE_STATS == {"made": 1, "reused": 2, "returned": 3, "discarded": 0}


def test_concurrent_rollouts_get_distinct_engines_and_the_pool_keeps_them():
    env = _env(engine_reuse=True)

    async def go():
        a, _, _ = await env._reset_engine("g1")
        b, _, _ = await env._reset_engine("g2")
        assert a is not b
        await env.cleanup_alf_env({"alf_env": a})
        await env.cleanup_alf_env({"alf_env": b})
        c, obs, _ = await env._reset_engine("g3")
        assert c in (a, b) and obs[0] == "obs of g3"
        return a, b

    a, b = asyncio.run(go())
    assert _Engine.made == 2 and not a.closed and not b.closed


def test_an_engine_whose_step_raised_is_closed_not_reused():
    env = _env(engine_reuse=True)

    async def go():
        tw_env, _, _ = await env._reset_engine("g1")
        with pytest.raises(RuntimeError):
            env._engine_call(tw_env, "step", ["boom"])
        await env.cleanup_alf_env({"alf_env": tw_env})
        return tw_env

    bad = asyncio.run(go())
    assert bad.closed and ae._TW_ENGINE_POOL.get("off", []) == []
    assert ae._TW_ENGINE_STATS["discarded"] == 1
    fresh, _ = _rollout(env, "g1")
    assert fresh is not bad


def test_an_engine_with_a_call_still_in_flight_is_closed_not_reused():
    """A step abandoned by a timeout leaves its thread inside the engine."""
    env = _env(engine_reuse=True)
    tw_env = _Engine("g1")
    tw_env._arm_in_call = True
    env._release_tw_env(tw_env)
    assert tw_env.closed and ae._TW_ENGINE_POOL.get("off", []) == []


def test_reset_retry_after_oserror_clears_the_mark(monkeypatch):
    env = _env(engine_reuse=True)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(ae.asyncio, "sleep", no_sleep)
    pooled, _ = _rollout(env, "g1")
    pooled.fail_next = OSError("failed to map segment from shared object")
    again, obs = _rollout(env, "g2")
    assert again is pooled and obs == "obs of g2" and not pooled.closed
    assert ae._TW_ENGINE_POOL["off"] == [pooled]


def test_pool_is_keyed_by_oracle_mode():
    plain, probe = _env(engine_reuse=True), _env(engine_reuse=True, oracle="handcoded")
    e_plain, _ = _rollout(plain, "g1")
    e_probe, _ = _rollout(probe, "g1")
    assert e_plain is not e_probe
    assert ae._TW_ENGINE_POOL == {"off": [e_plain], "handcoded": [e_probe]}
    again, _ = _rollout(probe, "g2")
    assert again is e_probe


def test_pool_is_bounded(monkeypatch):
    monkeypatch.setattr(ae, "_TW_ENGINE_POOL_MAX_IDLE", 2)
    env = _env(engine_reuse=True)
    engines = [_Engine(f"g{i}") for i in range(3)]
    for e in engines:
        env._release_tw_env(e)
    assert [e.closed for e in engines] == [False, False, True]
    assert len(ae._TW_ENGINE_POOL["off"]) == 2 and ae._TW_ENGINE_STATS["discarded"] == 1


def test_release_of_nothing_is_a_no_op():
    _env(engine_reuse=True)._release_tw_env(None)
    assert ae._TW_ENGINE_POOL == {}
