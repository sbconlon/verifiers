"""pi_hat_renorm flag (training-probe D-13, 2026-11-07).

With pi_hat_renorm=False the per-block conditionals are the raw sequence
probabilities exp(summed logp) instead of the softmax over the admissible
union. Pins: the raw helper; the training path's leave-one-out over raw
probabilities; the probe record (cond rows = exp(raw), marginal = column
mean, Z unchanged, the flag written); and that the default is the old
convention (golden value from the pre-refactor code).

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_pi_hat_renorm.py
"""
import asyncio
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()


class _FakeMultiTurnEnv:
    def __init__(self, **kwargs):
        pass

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
from environments.alfworld.rbmc import PiHatMatrix, conditional_renorm, raw_probs  # noqa: E402


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
        self.message = _Msg(content, _Toks([1, 2], list(range(8))))


def _completion(action):
    return [_AssistantMessage(f"<think>r</think><action>{action}</action>")]


def _scorer_from_raw(raw_rows):
    async def fake(client, prompts, spans, model, temperature=1.0):
        out = []
        for row in raw_rows:
            out.extend(row)
        assert len(out) == len(prompts)
        return out

    return fake


def _env(m, probe, raw_rows, renorm):
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.num_reasoning_blocks = m
    env.probe_capture = probe
    env.probe_block_temperature = 0.8
    env.probe_block_min_p = 0.1
    env.store_raw_logps = False
    env.oracle = "off"
    env.use_nm_fusion = False
    env.prewarm_pi_hat_prefixes = False
    env._tokenizer = _FakeTok()
    env._score_continuations = _scorer_from_raw(raw_rows)
    if renorm is not None:
        env.pi_hat_renorm = renorm
    return env


RAW4 = [
    [-1.3, -0.2, -2.7, -0.9],
    [-0.4, -1.1, -0.6, -3.2],
    [-2.2, -2.0, -0.1, -1.5],
    [-0.7, -0.7, -1.9, -0.3],
]
GOLDEN_LOO_J2 = 0.2262275884744491  # softmax convention, j*=2, a* appended at idx 3
ADM3 = ["go to desk 1", "look", "take pen 1 from desk 1"]


def test_raw_probs_helper():
    assert raw_probs([math.log(0.2), math.log(0.5)]) == pytest.approx([0.2, 0.5])
    assert sum(raw_probs([-1.0, -2.0])) < 1.0
    with pytest.raises(ValueError):
        raw_probs([])
    # the two conventions agree up to the normalizer
    row = [-1.3, -0.2, -2.7]
    z = math.log(sum(math.exp(v) for v in row))
    assert conditional_renorm(row) == pytest.approx([p / math.exp(z) for p in raw_probs(row)])


def test_pi_hat_matrix_default_is_renormalized():
    mat = PiHatMatrix(candidates=["a"], a_star_idx=0, cond=[[1.0]], Z=[0.0], raw=[[0.0]])
    assert mat.renormalized is True


def _train_state(blocks, j_star):
    return {
        "trajectory": [],
        "_last_admissible_commands": list(ADM3),
        "_pending_reasoning_blocks": blocks,
        "_executed_block_idx": j_star,
        "client": object(),
        "model": "m",
    }


def test_training_path_default_unchanged_golden():
    env = _env(4, probe=False, raw_rows=RAW4, renorm=None)  # attribute absent -> getattr default True
    state = _train_state([_Block(f"<think>r{j}</think><action>open drawer 1</action>") for j in range(4)], 2)
    step = {"completion": _completion("open drawer 1"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["pi_hat"] == pytest.approx(GOLDEN_LOO_J2, abs=1e-12)


def test_training_path_raw_loo():
    # pi_hat_renorm=False scores only <action>a*</action> under the m-1 held-out
    # blocks (_score_pi_hat_astar; a* at idx 3, j*=2 skipped), so the positional
    # fake sees 3 prompts and answers the a* column of RAW4 for blocks 0, 1, 3.
    # The request shape and the equality with the matrix path are pinned in
    # test_pi_hat_score_astar.py.
    env = _env(4, probe=False, raw_rows=[[RAW4[j][3]] for j in (0, 1, 3)], renorm=False)
    state = _train_state([_Block(f"<think>r{j}</think><action>open drawer 1</action>") for j in range(4)], 2)
    step = {"completion": _completion("open drawer 1"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    # a* appended at idx 3; leave out block 2; raw probabilities, no softmax
    expected = sum(math.exp(RAW4[j][3]) for j in (0, 1, 3)) / 3
    assert step["extras"]["pi_hat"] == pytest.approx(expected, abs=1e-12)
    assert step["extras"]["pi_hat"] != pytest.approx(GOLDEN_LOO_J2)  # a different estimator
    assert "probe" not in step["extras"]


# Physically valid rows: each row's probability mass exp(Z_j) is below 1, as a
# real sequence distribution's admissible mass must be.
RAW3 = [
    [-0.5, -1.5, -2.5],  # mass 0.912
    [-2.0, -0.5, -1.6],  # mass 0.944
    [-1.3, -1.3, -0.9],  # mass 0.952
]


def _probe_state(blocks):
    return {
        "trajectory": [],
        "_last_admissible_commands": ["go north", "look", "take apple 1 from table 1"],
        "_pending_reasoning_blocks": blocks,
        "_executed_block_idx": None,
        "client": object(),
        "model": "m",
    }


def test_probe_record_raw_convention():
    env = _env(3, probe=True, raw_rows=RAW3, renorm=False)
    blocks = [_Block(f"<think>b{j}</think><action>look</action>") for j in range(3)]
    state = _probe_state(blocks)
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    p = step["extras"]["probe"]
    assert p["pi_hat_renorm"] is False
    for j, b in enumerate(p["blocks"]):
        assert b["cond"] == pytest.approx([math.exp(v) for v in RAW3[j]])
        assert sum(b["cond"]) < 1.0
        # Z is still the log admissible mass, so cond sums to exp(Z)
        assert sum(b["cond"]) == pytest.approx(math.exp(p["Z"][j]), abs=1e-12)
    marginal = [sum(math.exp(RAW3[j][a]) for j in range(3)) / 3 for a in range(3)]
    assert p["pi_hat_marginal"] == pytest.approx(marginal)
    assert step["extras"]["pi_hat"] == pytest.approx(marginal[1])


def test_probe_record_renormalized_convention_flagged():
    env = _env(3, probe=True, raw_rows=RAW3, renorm=True)
    blocks = [_Block(f"<think>b{j}</think><action>look</action>") for j in range(3)]
    state = _probe_state(blocks)
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    p = step["extras"]["probe"]
    assert p["pi_hat_renorm"] is True
    for b in p["blocks"]:
        assert sum(b["cond"]) == pytest.approx(1.0, abs=1e-9)


def test_probe_record_flag_written_on_scoring_failure():
    async def boom(client, prompts, spans, model, temperature=1.0):
        raise RuntimeError("down")

    env = _env(2, probe=True, raw_rows=RAW3, renorm=False)
    env._score_continuations = boom
    blocks = [_Block("<think>b</think><action>look</action>") for _ in range(2)]
    state = _probe_state(blocks)
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    p = step["extras"]["probe"]
    assert p["pi_hat_renorm"] is False and p["blocks"] == [] and p["pi_hat_marginal"] is None


def test_constructor_flag_and_default():
    assert ALFWorldEnvironment().pi_hat_renorm is True
    assert ALFWorldEnvironment(pi_hat_renorm=False).pi_hat_renorm is False
