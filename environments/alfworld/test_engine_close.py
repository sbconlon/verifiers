"""Engine teardown unloads the planner library (job 750438: /tmp exhaustion).

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_engine_close.py
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()
_vf_mock.MultiTurnEnv = type("MultiTurnEnv", (), {"__init__": lambda self, **kw: None})
_vf_mock.cleanup = lambda f: f
_vf_mock.stop = lambda f: f
sys.modules.setdefault("verifiers", _vf_mock)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.alfworld import alfworld_env as ae  # noqa: E402


class _Pddl:
    def __init__(self, lib):
        self.downward_lib = lib


class _Wrapper:
    def __init__(self, env):
        self._wrapped_env = env


class _Batch:
    def __init__(self, envs):
        self.envs = envs

    def close(self):
        self.closed = True


class _Gym:
    """textworld's gym env: close() drops batch_env, as the real one does."""

    def __init__(self, envs):
        self.batch_env = _Batch(envs)
        self.closed = False

    def close(self):
        self.batch_env.close()
        self.batch_env = None
        self.closed = True


@pytest.fixture
def fake_fd(monkeypatch):
    closed = []
    mod = types.ModuleType("fast_downward")
    mod.close_lib = lambda lib: closed.append(lib)
    monkeypatch.setitem(sys.modules, "fast_downward", mod)
    return closed


def test_planner_libs_walks_the_wrapper_chain():
    lib = object()
    pddl = _Pddl(lib)
    gym = _Gym([_Wrapper(_Wrapper(pddl))])
    assert ae._planner_libs(gym) == [(pddl, lib)]


def test_close_engine_closes_then_unloads_once(fake_fd):
    lib = object()
    pddl = _Pddl(lib)
    gym = _Gym([_Wrapper(pddl)])
    ae._close_engine(gym)
    assert gym.closed is True
    assert fake_fd == [lib]
    assert pddl.downward_lib is None  # no dangling handle left on the engine
    # a second close finds nothing to unload
    gym2 = _Gym([_Wrapper(pddl)])
    ae._close_engine(gym2)
    assert fake_fd == [lib]


def test_close_engine_unloads_even_if_close_raises(fake_fd):
    lib = object()
    pddl = _Pddl(lib)
    gym = _Gym([pddl])
    gym.close = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        ae._close_engine(gym)
    assert fake_fd == [lib]


def test_close_engine_tolerates_test_doubles(fake_fd):
    class _Plain:
        closed = False

        def close(self):
            self.closed = True

    plain = _Plain()
    ae._close_engine(plain)
    assert plain.closed and fake_fd == []
    mock = MagicMock()
    ae._close_engine(mock)
    mock.close.assert_called_once()
    assert fake_fd == []


def test_unload_failure_is_swallowed(monkeypatch):
    mod = types.ModuleType("fast_downward")

    def bad(lib):
        raise OSError("dlclose failed")

    mod.close_lib = bad
    monkeypatch.setitem(sys.modules, "fast_downward", mod)
    gym = _Gym([_Pddl(object())])
    ae._close_engine(gym)  # must not raise
    assert gym.closed
