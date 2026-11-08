"""Phase 1 (training-time probe) -- probe capture mode of ALFWorldEnvironment.

Drives get_model_response / add_trajectory_step with fake clients and a fake
scorer (no vLLM, no TextWorld). Pins: the training path is numerically
unchanged by the pi_hat refactor (golden value captured from the pre-refactor
code), two requests per probe turn with the block sampling override, the
matrix shapes / marginal / Z, the per-block action parse, the window fields,
the scoring-failure degrade, the admissible-union order vs prime-rl, and the
probe-set dataset rows.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_probe_capture.py
"""
import asyncio
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()


class _Resp:
    """Fake vf.Response: carries the content + tokens the env reads."""

    def __init__(self, content, prompt_ids=(1, 2)):
        self.message = _Msg(content, _Toks(list(prompt_ids), list(range(8))))


class _Toks:
    def __init__(self, prompt_ids, completion_ids):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids


class _Msg:
    def __init__(self, content, tokens):
        self.content = content
        self.tokens = tokens


class _FakeMultiTurnEnv:
    """Fake base: records get_model_response calls (prompt + sampling_args)."""

    def __init__(self, **kwargs):
        pass

    async def get_model_response(self, state, prompt, *args, **kwargs):
        self._calls = getattr(self, "_calls", [])
        self._calls.append({"prompt": prompt, "sampling_args": kwargs.get("sampling_args")})
        return _Resp(f"<think>base{len(self._calls)}</think><action>look</action>")

    async def add_trajectory_step(self, state, trajectory_step):
        state["trajectory"].append(trajectory_step)


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
from environments.alfworld.probe_sets import load_probe_set  # noqa: E402


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _block(content):
    return _Resp(content)


def _completion(action):
    return [_AssistantMessage(f"<think>r</think><action>{action}</action>")]


def _scorer_from_raw(raw_rows):
    """Fake _score_continuations: one call carrying all m x |A| prompts (blocks
    concatenated in order); return the flat concatenation of the raw rows."""

    async def fake(client, prompts, spans, model, temperature=1.0):
        out = []
        for row in raw_rows:
            out.extend(row)
        assert len(out) == len(prompts), (len(out), len(prompts))
        return out

    return fake


def _env(m, probe=False, scorer=None, **attrs):
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
    if scorer is not None:
        env._score_continuations = scorer
    for k, v in attrs.items():
        setattr(env, k, v)
    return env


def _state(**over):
    s = {
        "trajectory": [],
        "trajectory_id": "t0",
        "client": object(),
        "model": "m",
        "sampling_args": {"temperature": 0.0, "max_tokens": 16, "extra_body": {"top_k": -1, "min_p": 0.0}},
    }
    s.update(over)
    return s


# Irregular golden fixture (also used by scratch golden_pi_hat.py on the
# pre-refactor code): m=4 blocks x |A|=4 candidates (3 admissible + a*).
RAW4 = [
    [-1.3, -0.2, -2.7, -0.9],
    [-0.4, -1.1, -0.6, -3.2],
    [-2.2, -2.0, -0.1, -1.5],
    [-0.7, -0.7, -1.9, -0.3],
]
GOLDEN_LOO_J2 = 0.2262275884744491  # pre-refactor _compute_pi_hat, j*=2, a* appended at idx 3


def _softmax(row):
    mx = max(row)
    ex = [math.exp(v - mx) for v in row]
    s = sum(ex)
    return [e / s for e in ex]


# ---------------------------------------------------------------------------
# D-12: training path unchanged
# ---------------------------------------------------------------------------


def test_training_path_unchanged_golden():
    env = _env(4, probe=False, scorer=_scorer_from_raw(RAW4))
    state = _state(
        _last_admissible_commands=["go to desk 1", "look", "take pen 1 from desk 1"],
        _pending_reasoning_blocks=[_block(f"<think>r{j}</think><action>open drawer 1</action>") for j in range(4)],
        _executed_block_idx=2,
    )
    step = {"completion": _completion("open drawer 1"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["pi_hat"] == pytest.approx(GOLDEN_LOO_J2, abs=1e-12)
    assert "probe" not in step["extras"]
    assert step["extras"]["admissible_actions"] == ["go to desk 1", "look", "take pen 1 from desk 1"]
    assert step["extras"]["executed_action"] == "open drawer 1"
    assert "_pending_reasoning_blocks" not in state and "_executed_block_idx" not in state


def test_training_path_get_model_response_unchanged():
    """probe_capture=False: m separate generations, j* from the stash, no sampling override."""
    env = _env(3, probe=False)
    state = _state(trajectory=[])
    resp = asyncio.run(env.get_model_response(state, ["P"]))
    assert len(env._calls) == 3
    assert all(c["sampling_args"] is None for c in env._calls)
    assert state["_executed_block_idx"] in (0, 1, 2)
    assert resp is state["_pending_reasoning_blocks"][state["_executed_block_idx"]]


# ---------------------------------------------------------------------------
# Probe generation: two requests per turn (D-4)
# ---------------------------------------------------------------------------


class _FakeNative:
    def __init__(self, choices):
        self.choices = choices

    def model_copy(self, update):
        return _FakeNative(update["choices"])


class _FakeClient:
    """Mimics the verifiers client surface _generate_m_blocks_fused uses."""

    def __init__(self, n_returned=None):
        self.calls = []
        self._n_returned = n_returned

    def _build_state_headers(self, state):
        return None

    async def to_native_prompt(self, prompt):
        return prompt, {}

    async def to_native_tools(self, tools):
        return tools

    async def get_native_response(self, native_prompt, model, sampling_args, native_tools, **kw):
        self.calls.append(dict(sampling_args))
        n = self._n_returned if self._n_returned is not None else sampling_args.get("n", 1)
        return _FakeNative([f"c{k}" for k in range(n)])

    async def raise_from_native_response(self, single):
        return None

    async def from_native_response(self, single):
        return _Resp(f"<think>{single.choices[0]}</think><action>go north</action>")


def test_probe_mode_two_requests_per_turn():
    env = _env(3, probe=True, use_nm_fusion=True)
    client = _FakeClient()
    state = _state(client=client)
    resp = asyncio.run(env.get_model_response(state, ["P"]))

    # 1) one greedy executed response through the base with the state's args
    assert len(env._calls) == 1
    assert env._calls[0]["sampling_args"] is None  # resolves to state["sampling_args"] (T=0)
    assert "base1" in resp.message.content
    # 2) one fused n=m request at the block temperature / min_p
    assert len(client.calls) == 1
    blk = client.calls[0]
    assert blk["n"] == 3
    assert blk["temperature"] == pytest.approx(0.8)
    assert blk["extra_body"]["min_p"] == pytest.approx(0.1)
    assert blk["extra_body"]["top_k"] == -1  # other extra_body keys preserved
    assert blk["max_tokens"] == 16
    # stash: m sampled blocks, no executed block index
    assert len(state["_pending_reasoning_blocks"]) == 3
    assert state["_executed_block_idx"] is None
    assert "_executed_block_idx" in state
    # the state's own sampling args are untouched
    assert state["sampling_args"]["temperature"] == 0.0
    assert state["sampling_args"]["extra_body"]["min_p"] == 0.0


def test_probe_mode_fallback_uses_block_args():
    """Server ignores n -> m separate generations, each with the block override."""
    env = _env(3, probe=True, use_nm_fusion=True)
    client = _FakeClient(n_returned=1)
    state = _state(client=client)
    asyncio.run(env.get_model_response(state, ["P"]))
    assert len(client.calls) == 1  # fusion attempted once
    assert len(env._calls) == 1 + 3  # greedy + m fallbacks through the base
    assert env._calls[0]["sampling_args"] is None
    for c in env._calls[1:]:
        assert c["sampling_args"]["temperature"] == pytest.approx(0.8)
        assert c["sampling_args"]["extra_body"]["min_p"] == pytest.approx(0.1)
    assert state["_executed_block_idx"] is None


def test_probe_block_sampling_args_without_extra_body():
    env = _env(2, probe=True)
    args = env._probe_block_sampling_args({"sampling_args": {"temperature": 0.0, "n": 7}})
    assert args == {"temperature": 0.8, "extra_body": {"min_p": 0.1}}


# ---------------------------------------------------------------------------
# Probe record (probe-schema.md §1)
# ---------------------------------------------------------------------------

RAW3 = [
    [-0.5, -1.5, -2.5],
    [-2.0, -0.1, -1.0],
    [-1.0, -1.0, -0.2],
]


def _probe_state(blocks, admissible=("go north", "look", "take apple 1 from table 1"), **over):
    return _state(
        _last_admissible_commands=list(admissible),
        _pending_reasoning_blocks=blocks,
        _executed_block_idx=None,
        **over,
    )


def test_probe_matrix_shapes_and_marginal():
    env = _env(3, probe=True, scorer=_scorer_from_raw(RAW3))
    blocks = [_block(f"<think>b{j}</think><action>look</action>") for j in range(3)]
    state = _probe_state(blocks)
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))

    p = step["extras"]["probe"]
    assert p["schema"] == 1
    assert p["admissible_union"] == ["go north", "look", "take apple 1 from table 1"]
    assert p["executed_idx"] == 1
    assert p["n_admissible"] == 3
    assert p["parsed_ok"] is True
    assert p["exec_block"] == {
        "reasoning": "<think>r</think>",
        "action": "look",
        "completion": "<think>r</think><action>look</action>",
    }
    assert len(p["blocks"]) == 3
    expected_cond = [_softmax(r) for r in RAW3]
    for j, b in enumerate(p["blocks"]):
        assert len(b["cond"]) == 3
        assert sum(b["cond"]) == pytest.approx(1.0, abs=1e-9)
        assert b["cond"] == pytest.approx(expected_cond[j])
        assert b["reasoning"] == f"<think>b{j}</think>"
        assert b["action"] == "look"
    marginal = [sum(expected_cond[j][a] for j in range(3)) / 3 for a in range(3)]
    assert p["pi_hat_marginal"] == pytest.approx(marginal)
    assert step["extras"]["pi_hat"] == pytest.approx(marginal[1])
    assert p["raw_logps"] is None  # store_raw_logps off
    # the three training-path keys are still written as before
    assert step["extras"]["admissible_actions"] == ["go north", "look", "take apple 1 from table 1"]
    assert step["extras"]["executed_action"] == "look"
    # stash consumed
    assert "_pending_reasoning_blocks" not in state and "_executed_block_idx" not in state
    assert state["trajectory"] == [step]


def test_Z_is_logsumexp_of_raw_and_raw_kept_when_flagged():
    env = _env(3, probe=True, scorer=_scorer_from_raw(RAW3), store_raw_logps=True)
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(3)]
    state = _probe_state(blocks)
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    p = step["extras"]["probe"]
    for j, row in enumerate(RAW3):
        mx = max(row)
        lse = mx + math.log(sum(math.exp(v - mx) for v in row))
        assert p["Z"][j] == pytest.approx(lse, abs=1e-12)
        # cond == exp(raw - Z)
        assert p["blocks"][j]["cond"] == pytest.approx([math.exp(v - lse) for v in row])
    assert p["raw_logps"] == RAW3


def test_inadmissible_astar_appended_in_probe_record():
    raw = [[-1.0, -2.0, -0.5]] * 2  # |A|=2 admissible + a* appended
    env = _env(2, probe=True, scorer=_scorer_from_raw(raw))
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(2)]
    state = _probe_state(blocks, admissible=("go north", "look"))
    step = {"prompt": [], "completion": _completion("open fridge 1"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    p = step["extras"]["probe"]
    assert p["admissible_union"] == ["go north", "look", "open fridge 1"]
    assert p["executed_idx"] == 2
    assert p["n_admissible"] == 2  # the appended a* sits at index >= n_admissible
    assert step["extras"]["pi_hat"] == pytest.approx(p["pi_hat_marginal"][2])


def test_union_order_matches_prime_rl():
    adm_mod = pytest.importorskip("prime_rl.orchestrator.admissible")
    helper = adm_mod.substitute_executed_into_admissible
    cases = [
        (["go", "look", "take"], "look"),
        (["go", "take"], "look"),  # not in set -> appended
        ([], "look"),
        (["look", "look", "go"], "look"),  # duplicate -> first occurrence
    ]
    for admissible, a_star in cases:
        assert ALFWorldEnvironment._admissible_union(admissible, a_star) == helper(list(admissible), a_star)


def test_block_actions_parsed_per_block():
    raw = [[-1.0, -1.0, -1.0]] * 3
    env = _env(3, probe=True, scorer=_scorer_from_raw(raw))
    blocks = [
        _block("<think>a</think><action>go north</action>"),
        _block("<think>b</think><action> look </action>"),
        _block("<think>c</think>"),  # no tag: _parse_action_text fallback
    ]
    state = _probe_state(blocks)
    # executed completion WITHOUT a tag: parsed_ok False, a* = fallback text
    step = {"prompt": [], "completion": [_AssistantMessage("<think>x</think> take apple 1 from table 1")], "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    p = step["extras"]["probe"]
    assert p["parsed_ok"] is False
    assert p["exec_block"]["action"] == "take apple 1 from table 1"
    assert step["extras"]["executed_action"] == "take apple 1 from table 1"
    assert [b["action"] for b in p["blocks"]] == [
        "go north",
        "look",
        ALFWorldEnvironment._parse_action_text("<think>c</think>"),
    ]
    assert p["blocks"][2]["reasoning"] == "<think>c</think>"


def test_window_fields():
    raw = [[0.0, 0.0]] * 2
    env = _env(2, probe=True, scorer=_scorer_from_raw(raw))

    async def fake_count(messages, state):
        return 1234

    env._count_tokens = fake_count
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(2)]
    prompt = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "obs 2"},
        {"role": "assistant", "content": "a 2"},
        {"role": "user", "content": "obs 3"},
    ]
    # two steps already taken, one paired eviction so far
    state = _probe_state(blocks, admissible=("go north", "look"), trajectory=[{}, {}], context_evictions=1)
    step = {"prompt": prompt, "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    w = step["extras"]["probe"]["window"]
    assert w["labels"] == ALFWorldEnvironment._label_prompt(prompt, 2)
    assert w["labels"] == ["System Prompt", "User (Turn 2)", "Assistant (Turn 2)", "User (Turn 3)"]
    assert w["token_count"] == 1234
    assert w["evictions_so_far"] == 1


def test_scoring_failure_degrades():
    async def boom(client, prompts, spans, model, temperature=1.0):
        raise RuntimeError("score route down")

    env = _env(2, probe=True, scorer=boom)
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(2)]
    state = _probe_state(blocks, _probe_oracle_last={"action": "go north", "status": "expert", "plan": None, "steps_left": None})
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))  # must not raise
    p = step["extras"]["probe"]
    assert p["blocks"] == []
    assert p["pi_hat_marginal"] is None
    assert p["Z"] == []
    assert "pi_hat" not in step["extras"]
    # the rest of the record is still there
    assert p["admissible_union"] == ["go north", "look", "take apple 1 from table 1"]
    assert p["executed_idx"] == 1
    assert p["oracle"] == {"action": "go north", "status": "expert", "plan": None, "steps_left": None}
    assert "_pending_reasoning_blocks" not in state
    assert state["trajectory"] == [step]


def test_error_turn_skips_scoring_but_writes_record():
    calls = {"n": 0}

    async def counting(client, prompts, spans, model, temperature=1.0):
        calls["n"] += 1
        return [0.0] * len(prompts)

    env = _env(2, probe=True, scorer=counting)
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(2)]
    state = _probe_state(blocks, error=RuntimeError("boom"))
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert calls["n"] == 0
    assert step["extras"]["probe"]["blocks"] == []
    assert "pi_hat" not in step["extras"]


def test_oracle_default_off_in_record():
    env = _env(2, probe=True, scorer=_scorer_from_raw([[0.0, 0.0, 0.0]] * 2))
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(2)]
    state = _probe_state(blocks)  # no _probe_oracle_last
    step = {"prompt": [], "completion": _completion("look"), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    assert step["extras"]["probe"]["oracle"] == {"action": None, "status": "off", "plan": None, "steps_left": None}


def test_compute_pi_hat_plain_mean_when_no_executed_block():
    env = _env(3, probe=True, scorer=_scorer_from_raw(RAW3))
    blocks = [_block("<think>b</think><action>look</action>") for _ in range(3)]
    state = _state()
    admissible = ["go north", "look", "take apple 1 from table 1"]
    val = asyncio.run(env._compute_pi_hat(state, blocks, None, admissible, "look"))
    expected = sum(_softmax(r)[1] for r in RAW3) / 3
    assert val == pytest.approx(expected)
    val_loo = asyncio.run(env._compute_pi_hat(state, blocks, 0, admissible, "look"))
    expected_loo = sum(_softmax(r)[1] for r in RAW3[1:]) / 2
    assert val_loo == pytest.approx(expected_loo)


# ---------------------------------------------------------------------------
# Constructor guards + probe-set dataset rows
# ---------------------------------------------------------------------------


def test_constructor_guards():
    with pytest.raises(ValueError):
        ALFWorldEnvironment(probe_capture=True, num_reasoning_blocks=1)
    with pytest.raises(ValueError):
        ALFWorldEnvironment(oracle="bogus")
    env = ALFWorldEnvironment(probe_capture=True, num_reasoning_blocks=2, oracle="handcoded")
    assert env.probe_capture is True and env.oracle == "handcoded"
    assert env.probe_block_temperature == 0.8 and env.probe_block_min_p == 0.1
    default = ALFWorldEnvironment()
    assert default.probe_capture is False and default.oracle == "off" and default.store_raw_logps is False


def test_probe_set_dataset_rows():
    from environments.alfworld.alfworld_env import build_dataset

    games = load_probe_set("alfworld-24-v1")
    ds = build_dataset("/data/alf", "valid_seen", probe_games=games)
    assert len(ds) == 24
    rows = list(ds)
    for row, g in zip(rows, games):
        info = row["info"]
        assert info["probe_id"] == g["probe_id"]
        assert info["game_id"] == g["game_id"]
        assert info["split"] == g["split"]
        assert info["task_type"] == g["task_type"]
        assert info["role"] == g["role"]
        assert info["game_file"].startswith("/data/alf/" + g["split"] + "/")
        assert info["game_file"].endswith("/game.tw-pddl")
        assert row["prompt"][0]["role"] == "system"
    assert [r["info"]["probe_id"] for r in rows] == [g["probe_id"] for g in games]


def test_probe_set_dataset_ignores_curriculum():
    from environments.alfworld.alfworld_env import build_dataset

    games = load_probe_set("alfworld-24-v1")[:2]
    ds = build_dataset("/data/alf", "train", curriculum=[{"default": 1.0}], probe_games=games)
    rows = list(ds)  # works for the real Dataset and for the list stub other test modules install
    assert len(rows) == 2
    assert all("curriculum_weights" not in r for r in rows)
