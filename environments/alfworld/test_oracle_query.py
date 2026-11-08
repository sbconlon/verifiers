"""Phase 1 (training-time probe) -- the in-env oracle (D-6).

alfworld is not installed on the laptop, so the handcoded agent is a scripted
fake whose policy.act returns a sequence of proposals or raises exceptions
named like alfworld's (HandCodedAgentTimeout / HandCodedAgentFailed); the env
maps them by class name. Also covers the planner record, the missing-'facts'
guard, the setup degrade without alfworld, and the env_response hook.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_oracle_query.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()


class _FakeMultiTurnEnv:
    def __init__(self, **kwargs):
        pass


class _AssistantMessage:
    def __init__(self, content=""):
        self.role = "assistant"
        self.content = content


class _UserMessage:
    def __init__(self, content=""):
        self.role = "user"
        self.content = content


_vf_mock.MultiTurnEnv = _FakeMultiTurnEnv
_vf_mock.AssistantMessage = _AssistantMessage
_vf_mock.UserMessage = _UserMessage
_vf_mock.cleanup = lambda f: f
sys.modules["verifiers"] = _vf_mock
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.modules.pop("environments.alfworld.alfworld_env", None)

from environments.alfworld.alfworld_env import ALFWorldEnvironment  # noqa: E402


class HandCodedAgentTimeout(NameError):
    pass


class HandCodedAgentFailed(NameError):
    pass


class _ScriptedPolicy:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def act(self, game_state, last_action):
        self.calls.append((dict(game_state), last_action))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _ScriptedAgent:
    def __init__(self, script):
        self.policy = _ScriptedPolicy(script)


def _env(oracle="handcoded"):
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.oracle = oracle
    return env


ADM = ["go to fridge 1", "open fridge 1", "look", "take apple 1 from table 1"]


def _gs(last_obs="You are in the kitchen.", admissible=ADM):
    return {"feedback": last_obs, "admissible_commands": list(admissible), "facts": [], "won": False}


def test_status_expert_random_look_fallback_failed_timeout_error():
    env = _env()
    agent = _ScriptedAgent(
        [
            "open fridge 1",  # admissible, != last -> expert
            "go to fridge 1",  # == last_action -> random
            "examine shelf 9",  # last == "look", not admissible -> random
            "examine shelf 9",  # not admissible, last != look -> look_fallback
            HandCodedAgentFailed(),
            HandCodedAgentTimeout(),
            ValueError("off-path state"),
        ]
    )
    out = [
        env._oracle_query(agent, _gs(), "go to fridge 1"),
        env._oracle_query(agent, _gs(), "go to fridge 1"),
        env._oracle_query(agent, _gs(), "look"),
        env._oracle_query(agent, _gs(), "go to fridge 1"),
        env._oracle_query(agent, _gs(), "open fridge 1"),
        env._oracle_query(agent, _gs(), "open fridge 1"),
        env._oracle_query(agent, _gs(), "open fridge 1"),
    ]
    assert [o["status"] for o in out] == [
        "expert",
        "random",
        "random",
        "look_fallback",
        "failed",
        "timeout",
        "error",
    ]
    assert out[0]["action"] == "open fridge 1"
    assert out[1]["action"] == "go to fridge 1"  # raw proposal recorded even when random would fire
    assert out[3]["action"] == "examine shelf 9"
    for o in out[4:]:
        assert o["action"] is None
    for o in out:
        assert set(o) == {"action", "status", "plan", "steps_left"}
        assert o["plan"] is None and o["steps_left"] is None
    # policy.act called exactly once per query, with the game_state and last action
    assert len(agent.policy.calls) == 7
    assert agent.policy.calls[2][1] == "look"
    assert agent.policy.calls[0][0]["admissible_commands"] == ADM


def test_oracle_after_step_builds_game_state_and_queries():
    env = _env()
    agent = _ScriptedAgent(["open fridge 1"])
    state = {"_probe_oracle_agent": agent}
    infos = {"admissible_commands": [ADM], "facts": [["fact-obj"]], "won": [False]}
    rec = env._oracle_after_step(state, "go to fridge 1", "You arrive at fridge 1. It is closed.", infos)
    assert rec == {"action": "open fridge 1", "status": "expert", "plan": None, "steps_left": None}
    gs, last = agent.policy.calls[0]
    assert last == "go to fridge 1"
    assert gs["feedback"] == "You arrive at fridge 1. It is closed."
    assert gs["admissible_commands"] == ADM
    assert gs["facts"] == ["fact-obj"]
    assert gs["won"] is False


def test_oracle_after_step_missing_facts_is_error():
    env = _env()
    agent = _ScriptedAgent(["open fridge 1"])
    state = {"_probe_oracle_agent": agent}
    infos = {"admissible_commands": [ADM], "won": [False]}  # no 'facts'
    rec = env._oracle_after_step(state, "go to fridge 1", "obs", infos)
    assert rec["status"] == "error" and rec["action"] is None
    assert agent.policy.calls == []  # never queried


def test_oracle_after_step_without_agent_is_error():
    env = _env()
    rec = env._oracle_after_step({"_probe_oracle_agent": None}, "look", "obs", {"admissible_commands": [ADM], "facts": [[]]})
    assert rec["status"] == "error"


def test_planner_oracle_record():
    env = _env("planner")
    infos = {"policy_commands": [["go to fridge 1", "open fridge 1", "take apple 1 from fridge 1"]]}
    rec = env._oracle_after_step({}, "look", "obs", infos)
    assert rec == {
        "action": "go to fridge 1",
        "status": "planner",
        "plan": ["go to fridge 1", "open fridge 1", "take apple 1 from fridge 1"],
        "steps_left": 3,
    }
    empty = env._planner_oracle({"policy_commands": [[]]})
    assert empty == {"action": None, "status": "planner", "plan": [], "steps_left": 0}
    missing = env._planner_oracle({"admissible_commands": [ADM]})
    assert missing["status"] == "error"


def test_oracle_setup_handcoded_degrades_without_game():
    """Whether alfworld is importable or not, a setup failure (no module here; no
    traj_data.json on a cluster) yields status=error, no agent, and no raise."""
    env = _env("handcoded")
    state = {}
    env._oracle_setup(state, "/nonexistent/game.tw-pddl", "-= Welcome to TextWorld, ALFRED! =-", {})
    assert state["_probe_oracle_agent"] is None
    assert state["_probe_oracle_last"]["status"] == "error"


def test_oracle_setup_planner_reads_policy_commands():
    env = _env("planner")
    state = {}
    env._oracle_setup(state, "/g/game.tw-pddl", "intro", {"policy_commands": [["look", "go to desk 1"]]})
    assert state["_probe_oracle_agent"] is None
    assert state["_probe_oracle_last"] == {
        "action": "look",
        "status": "planner",
        "plan": ["look", "go to desk 1"],
        "steps_left": 2,
    }


def test_oracle_setup_handcoded_with_injected_agent_class(monkeypatch):
    """Exercise the success path by injecting a fake alfworld.agents.expert module."""
    import types

    created = {}

    class _FakeTWAgent:
        def __init__(self, max_steps=150):
            created["max_steps"] = max_steps
            self.reset_game = None
            self.observed = None

        def reset(self, game):
            self.reset_game = game

        def observe(self, obs):
            self.observed = obs

    pkg = types.ModuleType("alfworld")
    agents = types.ModuleType("alfworld.agents")
    expert = types.ModuleType("alfworld.agents.expert")
    expert.HandCodedTWAgent = _FakeTWAgent
    monkeypatch.setitem(sys.modules, "alfworld", pkg)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents)
    monkeypatch.setitem(sys.modules, "alfworld.agents.expert", expert)

    env = _env("handcoded")
    state = {}
    env._oracle_setup(state, "/g/trial/game.tw-pddl", "-= Welcome =- You see a fridge 1.", {})
    agent = state["_probe_oracle_agent"]
    assert isinstance(agent, _FakeTWAgent)
    assert created["max_steps"] == 200
    assert agent.reset_game == "/g/trial/game.tw-pddl"
    assert agent.observed == "-= Welcome =- You see a fridge 1."
    assert state["_probe_oracle_last"] == {"action": None, "status": "init", "plan": None, "steps_left": None}


def test_cleanup_pops_oracle_state():
    import asyncio

    env = _env("handcoded")
    state = {"_probe_oracle_agent": object(), "_probe_oracle_last": {"status": "init"}, "alf_env": None}
    asyncio.run(env.cleanup_alf_env(state))
    assert "_probe_oracle_agent" not in state
    assert "_probe_oracle_last" not in state


def test_tw_env_id_cache_key_per_oracle():
    """Oracle modes register under their own key so request_infos differ per mode;
    the training path's key is the bare game file."""
    # Other alfworld test modules pop + re-import alfworld_env, so resolve the
    # cache through THIS class's defining module (not a fresh import).
    cache = ALFWorldEnvironment._get_tw_env_id.__globals__["_TW_ENV_ID_CACHE"]
    monkey = {"game.tw-pddl": "tw-train", "game.tw-pddl::oracle=handcoded": "tw-oracle"}
    saved = dict(cache)
    try:
        cache.clear()
        cache.update(monkey)
        assert _env("off")._get_tw_env_id("game.tw-pddl") == "tw-train"
        assert _env("handcoded")._get_tw_env_id("game.tw-pddl") == "tw-oracle"
    finally:
        cache.clear()
        cache.update(saved)
