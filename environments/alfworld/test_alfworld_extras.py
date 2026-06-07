"""Phase 1 (action-level ARM) -- ALFWorldEnvironment.add_trajectory_step override.

The override records, per turn, the *raw* admissible set the model saw and the
executed action text into step["extras"]. prime-rl's interleave_rollout applies
the union invariant and builds the DecisionPoint from these keys; here we only
verify the env-side extras population, driven by a synthetic state (no TextWorld
load, no GPU).

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_alfworld_extras.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock verifiers before importing alfworld_env (mirrors test_prompting.py), but
# provide a real base class with add_trajectory_step (the override calls super())
# and a real AssistantMessage class (the override's _parse_action isinstance-checks it).
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
_vf_mock.cleanup = lambda f: f  # identity decorator -> cleanup_alf_env stays callable
sys.modules["verifiers"] = _vf_mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force a fresh import of alfworld_env bound to THIS file's mock: other alfworld
# test modules install their own verifiers mock and import alfworld_env too, and
# the module is cached after the first import. Popping it isolates this file's
# binding (otherwise we'd reuse a base class missing the methods we exercise).
sys.modules.pop("environments.alfworld.alfworld_env", None)

from environments.alfworld.alfworld_env import ALFWorldEnvironment  # noqa: E402


def _env() -> ALFWorldEnvironment:
    # Bypass the heavy __init__ (TextWorld/alfworld); the override only needs
    # _parse_action (stateless) and super().add_trajectory_step.
    return ALFWorldEnvironment.__new__(ALFWorldEnvironment)


def _step(content: str | None) -> dict:
    completion = [] if content is None else [_AssistantMessage(content)]
    return {"completion": completion, "extras": {}}


def test_add_trajectory_step_populates_extras():
    env = _env()
    state = {"trajectory": [], "_last_admissible_commands": ["go to cabinet 1", "look"]}
    step = _step("<think>reason</think><action>go to cabinet 1</action>")
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["admissible_actions"] == ["go to cabinet 1", "look"]
    assert step["extras"]["executed_action"] == "go to cabinet 1"
    # super() was called: the step is appended to the trajectory.
    assert state["trajectory"] == [step]


def test_add_trajectory_step_inadmissible_action_recorded_raw():
    """An action absent from the admissible set is recorded as-is; the union
    (and its index) is applied later in interleave_rollout, not here."""
    env = _env()
    state = {"trajectory": [], "_last_admissible_commands": ["look", "go north"]}
    step = _step("<think>x</think><action>take apple</action>")
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["admissible_actions"] == ["look", "go north"]
    assert step["extras"]["executed_action"] == "take apple"


def test_add_trajectory_step_unparsable_action_falls_back_to_look():
    """No assistant message at all -> _parse_action's "look" fallback. This is the
    format-failure path (later visible via admissible_action_rate / format metrics)."""
    env = _env()
    state = {"trajectory": [], "_last_admissible_commands": ["look", "go north"]}
    step = _step(None)
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["executed_action"] == "look"


def test_add_trajectory_step_missing_admissible_defaults_empty():
    """Defensive: if _last_admissible_commands is absent, write an empty set
    (interleave's union still guarantees a* is present downstream)."""
    env = _env()
    state = {"trajectory": []}
    step = _step("<think>x</think><action>look</action>")
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["admissible_actions"] == []
    assert step["extras"]["executed_action"] == "look"
