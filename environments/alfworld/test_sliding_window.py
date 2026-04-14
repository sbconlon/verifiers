"""Unit test for ALFWorldEnvironment._apply_sliding_window eviction logic.

Runs without a live vLLM server by mocking _count_tokens.
"""
import sys
import types
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out problematic imports before loading the module
# ---------------------------------------------------------------------------

# Stub 'verifiers' package
vf_mod = types.ModuleType("verifiers")

class _FakeMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    def model_dump(self):
        return {"role": self.role, "content": self.content}

class _FakeUserMessage(_FakeMessage):
    def __init__(self, content=""):
        super().__init__("user", content)

class _FakeAssistantMessage(_FakeMessage):
    def __init__(self, content=""):
        super().__init__("assistant", content)

class _FakeSystemMessage(_FakeMessage):
    def __init__(self, content=""):
        super().__init__("system", content)

class _FakeMultiTurnEnv:
    def __init__(self, **kwargs):
        pass
    async def get_prompt_messages(self, state):
        return state.get("_messages", [])

class _FakeRubric:
    def add_reward_func(self, fn):
        pass

def _cleanup(fn):
    return fn

vf_mod.MultiTurnEnv = _FakeMultiTurnEnv
vf_mod.UserMessage = _FakeUserMessage
vf_mod.AssistantMessage = _FakeAssistantMessage
vf_mod.SystemMessage = _FakeSystemMessage
vf_mod.Rubric = _FakeRubric
vf_mod.State = dict
vf_mod.Messages = list
vf_mod.Error = Exception
vf_mod.cleanup = _cleanup
vf_mod.Environment = _FakeMultiTurnEnv

sys.modules["verifiers"] = vf_mod

# Stub 'datasets'
ds_mod = types.ModuleType("datasets")
ds_mod.Dataset = type("Dataset", (), {"from_list": staticmethod(lambda rows: rows)})
sys.modules["datasets"] = ds_mod

# Stub 'httpx'
httpx_mod = types.ModuleType("httpx")
class _AsyncClient:
    def __init__(self, **kwargs): self.is_closed = False
    async def post(self, *a, **kw): pass
httpx_mod.AsyncClient = _AsyncClient
sys.modules["httpx"] = httpx_mod

# ---------------------------------------------------------------------------
# Now we can import the env module
# ---------------------------------------------------------------------------
import importlib, importlib.util, pathlib

# Load alfworld_env.py from WSL via the Windows temp copy we already have
ALFWORLD_ENV_PATH = "/mnt/c/Users/SBC98/AppData/Local/Temp/alfworld_env_new.py"

spec = importlib.util.spec_from_file_location("alfworld_env", ALFWORLD_ENV_PATH)
mod = importlib.util.module_from_spec(spec)
# Stub alfworld imports that would fail
sys.modules.setdefault("alfworld", types.ModuleType("alfworld"))
sys.modules.setdefault("alfworld.agents", types.ModuleType("alfworld.agents"))
sys.modules.setdefault("alfworld.agents.environment", types.ModuleType("alfworld.agents.environment"))
sys.modules.setdefault("alfworld.agents.environment.alfred_tw_env", types.ModuleType("alfworld.agents.environment.alfred_tw_env"))
sys.modules["alfworld.agents.environment.alfred_tw_env"].AlfredDemangler = MagicMock
sys.modules["alfworld.agents.environment.alfred_tw_env"].AlfredInfos = MagicMock
spec.loader.exec_module(mod)

ALFWorldEnvironment = mod.ALFWorldEnvironment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env():
    """Return an ALFWorldEnvironment with max_context_tokens=100."""
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.max_context_tokens = 100
    env._http_client = None
    return env

def make_state(**extra):
    state = {
        "trajectory": [],
        "model": "test-model",
        "client": MagicMock(),
    }
    state.update(extra)
    return state

def make_messages(*specs):
    """Build a message list from (role, content) pairs."""
    mapping = {"system": _FakeSystemMessage, "user": _FakeUserMessage, "assistant": _FakeAssistantMessage}
    return [mapping[role](content) for role, content in specs]

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplySlidingWindow(unittest.TestCase):

    def _make_count_sequence(self, env, counts):
        """Patch _count_tokens to return successive values from counts list."""
        call_iter = iter(counts)
        async def fake_count(messages, state):
            return next(call_iter)
        env._count_tokens = fake_count

    # ------------------------------------------------------------------
    # 1. No eviction when within budget
    # ------------------------------------------------------------------
    def test_no_eviction_within_budget(self):
        env = make_env()
        env.max_context_tokens = 1000
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),
            ("assistant", "act1"),
        )
        state = make_state()
        self._make_count_sequence(env, [50])  # well within budget

        result = run(env._apply_sliding_window(list(msgs), state))

        self.assertEqual(len(result), 3)
        self.assertFalse(state.get("context_truncated"))

    # ------------------------------------------------------------------
    # 2. Single eviction: oldest non-system message removed
    # ------------------------------------------------------------------
    def test_single_eviction(self):
        env = make_env()
        env.max_context_tokens = 100
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),      # messages[1] — should be evicted
            ("assistant", "act1"),
            ("user", "obs2"),
        )
        state = make_state()
        # First count: over budget; second count (after pop): under budget
        self._make_count_sequence(env, [150, 80])

        result = run(env._apply_sliding_window(list(msgs), state))

        # messages[0] always kept; messages[1] (obs1) evicted
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].role, "system")
        self.assertEqual(result[1].content, "act1")
        self.assertEqual(result[2].content, "obs2")

    # ------------------------------------------------------------------
    # 3. Multiple evictions until budget satisfied
    # ------------------------------------------------------------------
    def test_multiple_evictions(self):
        env = make_env()
        env.max_context_tokens = 100
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),
            ("assistant", "act1"),
            ("user", "obs2"),
            ("assistant", "act2"),
        )
        state = make_state()
        # Needs 3 count calls: initial over, after 1st pop still over, after 2nd pop ok
        self._make_count_sequence(env, [300, 200, 80])

        result = run(env._apply_sliding_window(list(msgs), state))

        self.assertEqual(len(result), 3)  # system + obs2 + act2
        self.assertEqual(result[0].role, "system")
        self.assertEqual(result[1].content, "obs2")
        self.assertEqual(result[2].content, "act2")

    # ------------------------------------------------------------------
    # 4. Truncation metrics set correctly on first eviction
    # ------------------------------------------------------------------
    def test_truncation_metrics_first_eviction(self):
        env = make_env()
        env.max_context_tokens = 100
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),
            ("assistant", "act1"),
        )
        state = make_state()
        state["trajectory"] = ["step0", "step1", "step2"]  # len = 3
        self._make_count_sequence(env, [150, 80])

        run(env._apply_sliding_window(list(msgs), state))

        self.assertTrue(state["context_truncated"])
        self.assertEqual(state["context_truncated_at_turn"], 3)
        self.assertEqual(state["context_evictions"], 1)

    # ------------------------------------------------------------------
    # 5. Multiple evictions accumulate eviction count
    # ------------------------------------------------------------------
    def test_eviction_count_accumulates(self):
        env = make_env()
        env.max_context_tokens = 100
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),
            ("assistant", "act1"),
            ("user", "obs2"),
        )
        state = make_state()
        self._make_count_sequence(env, [300, 200, 80])

        run(env._apply_sliding_window(list(msgs), state))

        self.assertEqual(state["context_evictions"], 2)

    # ------------------------------------------------------------------
    # 6. context_truncated_at_turn NOT overwritten on second call
    # ------------------------------------------------------------------
    def test_truncated_at_turn_not_overwritten(self):
        env = make_env()
        env.max_context_tokens = 100
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),
            ("assistant", "act1"),
        )
        state = make_state()
        state["context_truncated"] = True
        state["context_truncated_at_turn"] = 2
        state["context_evictions"] = 3
        self._make_count_sequence(env, [150, 80])

        run(env._apply_sliding_window(list(msgs), state))

        # Should NOT overwrite the original turn
        self.assertEqual(state["context_truncated_at_turn"], 2)
        # But should increment evictions
        self.assertEqual(state["context_evictions"], 4)

    # ------------------------------------------------------------------
    # 7. System prompt never evicted (cannot go below 1 message)
    # ------------------------------------------------------------------
    def test_system_prompt_always_preserved(self):
        env = make_env()
        env.max_context_tokens = 10
        msgs = make_messages(
            ("system", "sys"),
        )
        state = make_state()
        # Always over budget — would loop forever without the len<=1 guard
        call_count = 0
        async def count_always_over(messages, state):
            nonlocal call_count
            call_count += 1
            return 9999
        env._count_tokens = count_always_over

        result = run(env._apply_sliding_window(list(msgs), state))

        # Only system prompt remains; we should have called count exactly once
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "system")
        self.assertEqual(call_count, 1)

    # ------------------------------------------------------------------
    # 8. Exact number of /tokenize calls
    # ------------------------------------------------------------------
    def test_tokenize_call_count(self):
        env = make_env()
        env.max_context_tokens = 100
        msgs = make_messages(
            ("system", "sys"),
            ("user", "obs1"),
            ("assistant", "act1"),
            ("user", "obs2"),
        )
        state = make_state()

        call_count = 0
        async def counting_count(messages, st):
            nonlocal call_count
            call_count += 1
            # First call: over; second: over; third: under
            return [300, 200, 80][call_count - 1]
        env._count_tokens = counting_count

        run(env._apply_sliding_window(list(msgs), state))

        self.assertEqual(call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
