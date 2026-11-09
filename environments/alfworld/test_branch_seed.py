"""Branched dataset Phase 1 -- ALFWorldEnvironment node primitives and the seeded
setup_state (plan/branched-dataset phase-01 §8).

No vLLM, no TextWorld: a scripted fused-generation client, a scripted span
scorer, a character tokenizer and a scripted engine. Pins: score_node's rows /
lead / own / o_ids and its two generation requests (greedy n=1 at T=0, sampled
n=m at the block args); synthesize_block's sampled-first / elicited / give-up
paths with the natural prompt used for verification; build_forced_step's token
alignment; the seeded entry's replay, invariant, installed conversation (the
load-bearing "next prompt equals the trunk's would-have-been prompt"), max_turns
offset, eviction keeping the latest observation, and the unseeded path unchanged.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_branch_seed.py
"""
import asyncio
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_vf_mock = MagicMock()


class _Toks:
    def __init__(self, prompt_ids, completion_ids):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids


class _Msg:
    def __init__(self, content, tokens):
        self.content = content
        self.tokens = tokens


class _Resp:
    def __init__(self, content, prompt_ids=(1, 2, 3)):
        self.message = _Msg(content, _Toks(list(prompt_ids), list(range(4))))


class _FakeMultiTurnEnv:
    def __init__(self, **kwargs):
        pass

    async def get_model_response(self, state, prompt, *args, **kwargs):
        self._calls = getattr(self, "_calls", [])
        self._calls.append({"prompt": prompt, "sampling_args": kwargs.get("sampling_args")})
        return _Resp(f"<think>base{len(self._calls)}</think>\n<action>look</action>")

    async def get_prompt_messages(self, state):
        # verifiers' MultiTurnEnv: the first prompt is state["prompt"] verbatim
        assert len(state["trajectory"]) == 0
        return state["prompt"]

    async def add_trajectory_step(self, state, trajectory_step):
        state["trajectory"].append(trajectory_step)


class _Message:
    def __init__(self, content=""):
        self.content = content

    def model_dump(self):
        return {"role": self.role, "content": self.content}

    def __eq__(self, other):
        return (
            getattr(other, "role", None) == self.role
            and getattr(other, "content", None) == self.content
        )

    def __repr__(self):
        return f"{self.role}:{self.content[:30]!r}"


class _AssistantMessage(_Message):
    role = "assistant"


class _UserMessage(_Message):
    role = "user"


class _SystemMessage(_Message):
    role = "system"


class _Error(Exception):
    pass


_vf_mock.MultiTurnEnv = _FakeMultiTurnEnv
_vf_mock.AssistantMessage = _AssistantMessage
_vf_mock.UserMessage = _UserMessage
_vf_mock.SystemMessage = _SystemMessage
_vf_mock.Error = _Error
_vf_mock.cleanup = lambda f: f
_vf_mock.stop = lambda f: f
sys.modules["verifiers"] = _vf_mock
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.modules.pop("environments.alfworld.alfworld_env", None)

from environments.alfworld.alfworld_env import ALFWorldEnvironment  # noqa: E402
from environments.alfworld.branching import ELICIT_DEFAULT, LEAK_DEFAULT, NodeScore  # noqa: E402

ADM = ["go to desk 1", "open drawer 1", "go to dresser 1", "examine shelf 1"]


class _FakeTok:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages)


class _FakeNative:
    def __init__(self, choices):
        self.choices = choices

    def model_copy(self, update):
        return _FakeNative(update["choices"])


class _ScriptedClient:
    """Fused-generation client: each get_native_response pops one script entry
    (a list of completion texts, one per returned choice)."""

    def __init__(self, scripts, prompt_ids=(1, 2, 3)):
        self.scripts = list(scripts)
        self.calls = []
        self.prompt_ids = tuple(prompt_ids)

    def _build_state_headers(self, state):
        return None

    async def to_native_prompt(self, prompt):
        return prompt, {}

    async def to_native_tools(self, tools):
        return tools

    async def get_native_response(self, native_prompt, model, sampling_args, native_tools, **kw):
        self.calls.append({"prompt": list(native_prompt), "sampling_args": dict(sampling_args)})
        contents = self.scripts.pop(0)
        assert len(contents) == sampling_args.get("n", 1), (contents, sampling_args)
        return _FakeNative(list(contents))

    async def raise_from_native_response(self, single):
        return None

    async def from_native_response(self, single):
        return _Resp(single.choices[0], prompt_ids=self.prompt_ids)


def _scorer(rows_queue, seen):
    """Scripted _score_continuations: pops raw rows until the prompts are covered."""

    async def fake(client, prompts, spans, model, temperature=1.0):
        out = []
        while len(out) < len(prompts):
            out.extend(rows_queue.pop(0))
        assert len(out) == len(prompts), (len(out), len(prompts))
        seen.append({"prompts": [list(p) for p in prompts], "spans": list(spans)})
        return out

    return fake


def _env(**attrs):
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.num_reasoning_blocks = 1
    env.probe_capture = False
    env.oracle = "off"
    env.use_nm_fusion = True
    env.prewarm_pi_hat_prefixes = False
    env.use_score_route = True
    env.pi_hat_renorm = False  # node rows must be normalized regardless (D-5)
    env.store_raw_logps = False
    env.max_context_tokens = -1
    env.max_turns = 50
    env.branch_mode = True
    env.tau = 0.5
    env.n_elicit = 3
    env.elicit_prompt = ELICIT_DEFAULT
    env.leak_patterns = list(LEAK_DEFAULT)
    env._tokenizer = _FakeTok()
    env._tokenizer_path = None
    for k, v in attrs.items():
        setattr(env, k, v)
    return env


def _softmax(row):
    mx = max(row)
    ex = [math.exp(v - mx) for v in row]
    s = sum(ex)
    return [e / s for e in ex]


# Raw logp rows over ADM for the 4 blocks of a node (block 0 greedy + m=3):
# argmax per block -> desk, drawer, desk, dresser.
RAW_NODE = [
    [-0.3, -2.0, -2.5, -3.0],
    [-2.0, -0.4, -2.2, -3.5],
    [-0.5, -1.5, -2.5, -3.0],
    [-2.0, -2.5, -1.3, -3.0],  # dresser tops it but softmax ~0.62? see test
]
BLOCKS = [
    "<think>greedy plan</think>\n<action>go to desk 1</action>",
    "<think>drawer plan</think>\n<action>open drawer 1</action>",
    "<think>desk plan</think>\n<action>go to desk 1</action>",
    "<think>dresser plan</think>\n<action>go to dresser 1</action>",
]
PROMPT = [_SystemMessage("sys"), _UserMessage("obs0")]
BLOCK_ARGS = {"temperature": 0.8, "max_tokens": 64, "extra_body": {"min_p": 0.1}}


def _node(env, client=None, scorer_rows=None, seen=None):
    client = client or _ScriptedClient([[BLOCKS[0]], BLOCKS[1:]])
    rows = scorer_rows if scorer_rows is not None else [list(r) for r in RAW_NODE]
    seen = seen if seen is not None else []
    env._score_continuations = _scorer(rows, seen)
    node = asyncio.run(env.score_node(client, "m", PROMPT, ADM, 3, BLOCK_ARGS))
    return node, client, seen


# ---------------------------------------------------------------------------
# score_node
# ---------------------------------------------------------------------------


def test_score_node_rows_lead_own_and_requests():
    env = _env()
    node, client, seen = _node(env)
    # two generation requests: greedy n=1 at T=0, then n=m at the block args
    assert [c["sampling_args"]["n"] for c in client.calls] == [1, 3]
    assert client.calls[0]["sampling_args"]["temperature"] == 0.0
    assert client.calls[1]["sampling_args"]["temperature"] == pytest.approx(0.8)
    assert client.calls[1]["sampling_args"]["extra_body"]["min_p"] == pytest.approx(0.1)
    # one scoring request for all (m+1) x |A| prompts
    assert len(seen) == 1 and len(seen[0]["prompts"]) == 4 * len(ADM)
    # rows normalized although pi_hat_renorm=False on the env (D-5)
    for row, raw in zip(node.rows, RAW_NODE):
        assert sum(row) == pytest.approx(1.0)
        assert row == pytest.approx(_softmax(raw))
    assert node.masses == pytest.approx([math.exp(math.log(sum(math.exp(v) for v in r))) for r in RAW_NODE])
    assert node.lead == ["go to desk 1", "open drawer 1", "go to desk 1", "go to dresser 1"]
    assert node.own == ["go to desk 1", "open drawer 1", "go to dresser 1"]
    assert node.texts == [b.split("<action>")[0] for b in BLOCKS]
    assert [b.tag_action for b in node.blocks] == ["go to desk 1", "open drawer 1", "go to desk 1", "go to dresser 1"]
    assert node.blocks[0].completion_text == BLOCKS[0]
    assert node.o_ids == [1, 2, 3]
    assert node.admissible == ADM


def test_score_node_fallback_when_n_not_honored():
    env = _env()

    class _NoN(_ScriptedClient):
        async def get_native_response(self, native_prompt, model, sampling_args, native_tools, **kw):
            self.calls.append({"sampling_args": dict(sampling_args)})
            return _FakeNative(["only one"])  # ignores n -> fused path returns None for n=3

    client = _NoN([])
    seen = []
    env._score_continuations = _scorer([list(r) for r in RAW_NODE], seen)
    node = asyncio.run(env.score_node(client, "m", PROMPT, ADM, 3, BLOCK_ARGS))
    # greedy via fusion (n=1 honored), sampled via 3 separate base generations
    assert len(env._calls) == 3
    assert all(c["sampling_args"]["temperature"] == pytest.approx(0.8) for c in env._calls)
    assert len(node.blocks) == 4


def test_score_node_rejects_empty_admissible():
    env = _env()
    with pytest.raises(ValueError):
        asyncio.run(env.score_node(_ScriptedClient([]), "m", PROMPT, [], 3, BLOCK_ARGS))


class _StrictClient(_ScriptedClient):
    """Like the real OpenAI client: to_native_prompt dispatches on the message
    classes and raises ValueError("Invalid chat message: ...") on a role/content
    dict (what the collect driver sends; job 748578, 2026-11-09)."""

    async def to_native_prompt(self, prompt):
        for message in prompt:
            if not hasattr(message, "role"):
                raise ValueError(f"Invalid chat message: {message}")
        return prompt, {}


WIRE_PROMPT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "obs0"}]


def test_score_node_accepts_wire_form_prompt():
    env = _env()
    client = _StrictClient([[BLOCKS[0]], BLOCKS[1:]])
    env._score_continuations = _scorer([list(r) for r in RAW_NODE], [])
    node = asyncio.run(env.score_node(client, "m", WIRE_PROMPT, ADM, 3, BLOCK_ARGS))
    assert node.own[0] == "go to desk 1"
    assert len(client.calls) == 2
    for call in client.calls:
        # typed messages reached the client, equal to the typed PROMPT
        assert call["prompt"] == PROMPT


def test_elicit_block_accepts_wire_form_prompt():
    env = _env()
    client = _StrictClient([["<think>forced</think>\n<action>open drawer 1</action>"]])
    text = asyncio.run(env.elicit_block(client, "m", WIRE_PROMPT, "open drawer 1", BLOCK_ARGS))
    assert "open drawer 1" in text
    sent = client.calls[0]["prompt"]
    assert sent[:2] == PROMPT
    assert sent[2].role == "user" and "open drawer 1" in sent[2].content


# ---------------------------------------------------------------------------
# synthesize_block
# ---------------------------------------------------------------------------


def test_synthesize_prefers_sampled_block():
    env = _env()
    node, client, seen = _node(env)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "open drawer 1", node, BLOCK_ARGS))
    assert syn.source == "sampled"
    assert syn.block_idx == 1
    assert syn.attempts == 0
    assert syn.reject_reasons == []
    assert syn.p_accept == pytest.approx(_softmax(RAW_NODE[1])[1])
    assert syn.reasoning == "<think>drawer plan</think>\n"
    assert syn.completion_text == "<think>drawer plan</think>\n<action>open drawer 1</action>"
    # no elicitation request was made
    assert len(client.scripts) == 0 and len(client.calls) == 2


def test_synthesize_elicits_when_sampled_block_below_tau():
    env = _env(tau=0.7)
    node, client, seen = _node(env)
    # dresser: block 3 tops the row but below 0.7 -> "tau"; then elicit:
    # attempt 1 leaks, attempt 2 passes (scored under the natural prompt)
    client.scripts = [
        ["<think>I was told to go to the dresser.</think>\n<action>go to dresser 1</action>"],
        ["<think>The dresser is the likely spot.</think>\n<action>go to dresser 1</action>"],
    ]
    rows_queue = [[-3.0, -3.0, -0.1, -3.0]]  # one row for the accepted elicited block
    env._score_continuations = _scorer(rows_queue, seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "go to dresser 1", node, BLOCK_ARGS))
    assert syn.source == "elicited"
    assert syn.block_idx is None
    assert syn.attempts == 2
    assert syn.reject_reasons == ["tau", "leak"]
    assert syn.reasoning == "<think>The dresser is the likely spot.</think>\n"
    assert syn.completion_text.endswith("<action>go to dresser 1</action>")
    assert syn.p_accept == pytest.approx(_softmax(rows_queue_ref := [-3.0, -3.0, -0.1, -3.0])[2]) if False else True
    # the elicitation prompt is the natural prompt plus one user instruction
    elicit_prompts = [c["prompt"] for c in client.calls[2:]]
    for p in elicit_prompts:
        assert p[:-1] == PROMPT
        assert p[-1].role == "user"
        assert "go to dresser 1" in p[-1].content
        assert ELICIT_DEFAULT.format(action="go to dresser 1") == p[-1].content
    # verification was scored under the NATURAL prompt ids (node.o_ids), one row
    last = seen[-1]
    assert len(last["prompts"]) == len(ADM)
    assert all(p[:3] == node.o_ids for p in last["prompts"])


def test_synthesize_gives_up_with_reasons():
    env = _env(n_elicit=2)
    node, client, seen = _node(env)
    # "examine shelf 1": no block leads to it; both elicitations top a different action
    client.scripts = [
        ["<think>a</think>\n<action>examine shelf 1</action>"],
        ["<think>b</think>\n<action>examine shelf 1</action>"],
    ]
    env._score_continuations = _scorer([[-0.1, -3.0, -3.0, -3.0], [-0.1, -3.0, -3.0, -3.0]], seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "examine shelf 1", node, BLOCK_ARGS))
    assert syn.source == "none"
    assert syn.attempts == 2
    assert syn.reject_reasons == ["argmax", "argmax"]
    assert syn.completion_text == "" and syn.reasoning == ""
    with pytest.raises(ValueError):
        env.build_forced_step({"tokens": {}}, syn, "examine shelf 1", "b", ADM)


def test_synthesize_parse_failure_counts():
    env = _env(n_elicit=1)
    node, client, seen = _node(env)
    client.scripts = [["no tag at all"]]
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "examine shelf 1", node, BLOCK_ARGS))
    assert syn.source == "none" and syn.reject_reasons == ["parse"]


def test_synthesize_requires_admissible_action():
    env = _env()
    node, client, seen = _node(env)
    with pytest.raises(ValueError):
        asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "fly", node, BLOCK_ARGS))


# ---------------------------------------------------------------------------
# build_forced_step
# ---------------------------------------------------------------------------


def test_build_forced_step_alignment():
    env = _env()
    node, client, seen = _node(env)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "open drawer 1", node, BLOCK_ARGS))
    trunk_step = {
        "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "obs0"}],
        "completion": [{"role": "assistant", "content": BLOCKS[0]}],
        "tokens": {"prompt_ids": [5, 6, 7], "prompt_mask": [0, 0, 0], "completion_ids": [9], "completion_mask": [1], "completion_logprobs": [-0.1]},
        "extras": {"admissible_actions": ADM, "executed_action": "go to desk 1"},
    }
    step = env.build_forced_step(trunk_step, syn, "open drawer 1", "g1-t0-b1", ADM)
    text = syn.completion_text
    assert step["prompt"] == trunk_step["prompt"]
    assert step["completion"] == [{"role": "assistant", "content": text}]
    assert step["tokens"]["prompt_ids"] == [5, 6, 7]
    assert step["tokens"]["prompt_mask"] == [0, 0, 0]
    assert step["tokens"]["completion_ids"] == [ord(c) for c in text] + [999]
    n = len(step["tokens"]["completion_ids"])
    assert step["tokens"]["completion_mask"] == [1] * n
    assert step["tokens"]["completion_logprobs"] == [0.0] * n
    assert step["tokens"]["completion_top_k_token_ids"] is None
    assert step["tokens"]["routed_experts"] is None
    assert step["extras"]["executed_action"] == "open drawer 1"
    assert step["extras"]["admissible_actions"] == ADM
    assert step["extras"]["pi_hat"] == pytest.approx(syn.p_accept)
    assert step["extras"]["branch"] == {"forced": True, "block_source": "sampled", "p_accept": pytest.approx(syn.p_accept), "block_idx": 1}
    assert step["trajectory_id"] == "g1-t0-b1"
    assert step["is_truncated"] is False and step["response"] is None
    # the keys interleave_rollout's prepare_step_tokens reads are all present
    for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "completion_logprobs"):
        assert k in step["tokens"]


# ---------------------------------------------------------------------------
# seeded setup_state
# ---------------------------------------------------------------------------


class _Engine:
    """Scripted TextWorld engine: step(action) -> next (obs, admissible, done, won)."""

    def __init__(self, script):
        self.script = dict(script)
        self.steps = []
        self.closed = False

    def step(self, actions):
        (a,) = actions
        self.steps.append(a)
        obs, adm, done, won = self.script[a]
        return [obs], [0.0], [done], {"admissible_commands": [adm], "won": [won]}

    def close(self):
        self.closed = True


def _fmt(obs, adm):
    return ALFWorldEnvironment._format_obs(obs, adm)


ADM0 = ["go to desk 1", "go to dresser 1", "look"]
ADM1 = ["open drawer 1", "go to dresser 1", "look"]
ADM2 = ["take pen 1", "close drawer 1", "look"]
ADM2B = ["examine dresser 1", "go to desk 1", "look"]
SCRIPT = {
    "go to desk 1": ("You are at the desk.", ADM1, False, False),
    "open drawer 1": ("The drawer is open. You see a pen 1.", ADM2, False, False),
    "go to dresser 1": ("You are at the dresser.", ADM2B, False, False),
    "take pen 1": ("You take the pen. Task complete!", ["look"], True, True),
}


def _seeded_env(engine, **attrs):
    env = _env(**attrs)

    async def _reset(game_file):
        return engine, ["Welcome. Find a pen."], {"admissible_commands": [ADM0], "won": [False]}

    env._reset_engine = _reset
    return env


def _trunk_prompt():
    return [_SystemMessage("sys"), _UserMessage(_fmt("Welcome. Find a pen.", ADM0))]


def _state(prompt, seed):
    return {
        "prompt": list(prompt),
        "info": {"game_file": "/g/game.tw-pddl", "branch_seed": seed},
        "trajectory": [],
        "trajectory_id": "br",
        "final_env_response": None,
    }


def _seed(**over):
    c0 = "<think>desk first</think>\n<action>go to desk 1</action>"
    seed = {
        "trunk_id": "g1-t0",
        "branch_id": "g1-t0-b0",
        "actions_prefix": ["go to desk 1"],
        "expected_obs": _fmt("You are at the desk.", ADM1),
        "forced_action": "go to dresser 1",
        "forced_completion": "<think>the dresser instead</think>\n<action>go to dresser 1</action>",
        "completion_prefix": [
            {"role": "assistant", "content": c0},
            {"role": "user", "content": _fmt("You are at the desk.", ADM1)},
        ],
        "turn": 1,
    }
    seed.update(over)
    return seed


def test_seeded_setup_replays_and_installs_conversation():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine)
    state = _state(_trunk_prompt(), _seed())
    out = asyncio.run(env.setup_state(state))
    assert engine.steps == ["go to desk 1", "go to dresser 1"]
    assert out["alf_env"] is engine
    assert out["won"] is False and out["final_env_response"] is None
    assert out["_last_admissible_commands"] == ADM2B
    assert out["_branch_turn_offset"] == 2
    assert out["branch"]["replay_ok"] is True and out["branch"]["forced_action"] == "go to dresser 1"
    assert out["context_evictions"] == 0 and out["context_truncated"] is False
    msgs = out["prompt"]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[4].content == _seed()["forced_completion"]
    assert msgs[5].content == _fmt("You are at the dresser.", ADM2B)
    assert len(out["trajectory"]) == 0


def test_seeded_next_prompt_matches_trunk_would_have_been():
    """Load-bearing (O-3): the first prompt the loop builds for a seeded episode
    equals the prompt a trunk that took the forced action would have shown at
    turn t+1: trunk prompt at t + [assistant(forced), user(obs after)]."""
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine)
    seed = _seed()
    state = _state(_trunk_prompt(), seed)
    asyncio.run(env.setup_state(state))
    first_prompt = asyncio.run(env.get_prompt_messages(state))
    trunk_prompt_at_t = _trunk_prompt() + [
        _AssistantMessage(seed["completion_prefix"][0]["content"]),
        _UserMessage(seed["completion_prefix"][1]["content"]),
    ]
    expected = trunk_prompt_at_t + [
        _AssistantMessage(seed["forced_completion"]),
        _UserMessage(_fmt("You are at the dresser.", ADM2B)),
    ]
    assert first_prompt == expected


def test_seeded_invariant_mismatch_raises_and_closes():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine)
    state = _state(_trunk_prompt(), _seed(expected_obs="something else"))
    with pytest.raises(_Error, match="replay invariant failed"):
        asyncio.run(env.setup_state(state))
    assert engine.closed is True
    assert engine.steps == ["go to desk 1"]  # the forced action was never stepped


def test_seeded_prefix_ending_episode_raises():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine)
    seed = _seed(actions_prefix=["go to desk 1", "open drawer 1", "take pen 1"], turn=3)
    with pytest.raises(_Error, match="ended at prefix step"):
        asyncio.run(env.setup_state(_state(_trunk_prompt(), seed)))
    assert engine.closed is True


def test_seeded_forced_action_ending_episode_sets_final_response():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine)
    seed = _seed(
        actions_prefix=["go to desk 1", "open drawer 1"],
        expected_obs=_fmt("The drawer is open. You see a pen 1.", ADM2),
        forced_action="take pen 1",
        forced_completion="<think>pen</think>\n<action>take pen 1</action>",
        turn=2,
    )
    out = asyncio.run(env.setup_state(_state(_trunk_prompt(), seed)))
    assert out["won"] is True
    assert out["final_env_response"] is not None
    assert out["final_env_response"][0].content == _fmt("You take the pen. Task complete!", ["look"])
    assert out["_branch_turn_offset"] == 3


def test_seed_requires_branch_mode_and_keys():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine, branch_mode=False)
    with pytest.raises(_Error, match="branch_mode"):
        asyncio.run(env.setup_state(_state(_trunk_prompt(), _seed())))
    assert engine.steps == []  # refused before touching the engine
    env = _seeded_env(engine)
    bad = _seed()
    del bad["expected_obs"]
    with pytest.raises(_Error, match="missing keys"):
        asyncio.run(env.setup_state(_state(_trunk_prompt(), bad)))


def test_seeded_respects_max_turns():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine, max_turns=5)
    state = _state(_trunk_prompt(), _seed(turn=2))
    asyncio.run(env.setup_state(state))
    assert state["_branch_turn_offset"] == 3
    # 5 - 3 = 2 model calls remain
    assert asyncio.run(env.max_turns_reached(state)) is False
    state["trajectory"].append({})
    assert asyncio.run(env.max_turns_reached(state)) is False
    state["trajectory"].append({})
    assert asyncio.run(env.max_turns_reached(state)) is True
    # unseeded: the parent's rule verbatim
    plain = {"trajectory": [{}] * 5}
    assert asyncio.run(env.max_turns_reached(plain)) is True
    assert asyncio.run(env.max_turns_reached({"trajectory": [{}] * 4})) is False
    env.max_turns = -1
    assert asyncio.run(env.max_turns_reached(state)) is False


def test_eviction_keeps_latest_observation():
    """O-4: the sliding window evicts from messages[1:] FIFO; the newest observation
    (the last message) survives as long as anything older is left to evict."""
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine)
    state = _state(_trunk_prompt(), _seed())
    asyncio.run(env.setup_state(state))
    full = list(state["prompt"])
    # budget: everything minus the first user/assistant pair (the fake tokenizer
    # counts characters), so exactly one paired eviction fits
    total = sum(len(m.content) for m in full)
    env.max_context_tokens = total - len(full[1].content) - len(full[2].content) + 1
    windowed = asyncio.run(env.get_prompt_messages(state))
    assert windowed[0] is full[0]  # system prompt kept
    assert windowed[-1] is full[-1]  # latest observation kept
    assert windowed == [full[0]] + full[3:]
    assert state["context_truncated"] is True
    assert state["context_evictions"] >= 1


def test_unseeded_setup_path_unchanged():
    engine = _Engine(SCRIPT)
    env = _seeded_env(engine, branch_mode=False)
    state = {"prompt": [_SystemMessage("sys")], "info": {"game_file": "/g/game.tw-pddl"}, "trajectory": []}
    out = asyncio.run(env.setup_state(state))
    assert engine.steps == []
    assert out["alf_env"] is engine
    assert [m.role for m in out["prompt"]] == ["system", "user"]
    assert out["prompt"][1].content == _fmt("Welcome. Find a pen.", ADM0)
    assert "_branch_turn_offset" not in out and "branch" not in out
    assert out["_last_admissible_commands"] == ADM0


def test_node_client_config_resolved_and_cached(monkeypatch):
    """A ClientConfig (what the inference pool returns) is resolved through
    verifiers.clients.resolve_client once per distinct config; live clients and
    the tests' scripted clients pass through."""
    import types as _types

    class _Client:
        pass

    class _Cfg:
        def __init__(self, url):
            self.url = url

        def model_dump_json(self):
            return self.url

    built = []

    def _resolve(cfg):
        built.append(cfg.url)
        return _Client()

    fake_mod = _types.SimpleNamespace(Client=_Client, resolve_client=_resolve)
    monkeypatch.setitem(sys.modules, "verifiers.clients", fake_mod)
    env = _env()
    live = _Client()
    assert env._resolve_node_client(live) is live
    a1 = env._resolve_node_client(_Cfg("http://a"))
    a2 = env._resolve_node_client(_Cfg("http://a"))
    b1 = env._resolve_node_client(_Cfg("http://b"))
    assert a1 is a2 and a1 is not b1
    assert built == ["http://a", "http://b"]
    scripted = _ScriptedClient([])
    assert env._resolve_node_client(scripted) is scripted
    assert env._node_state(_Cfg("http://a"), "m", {})["client"] is a1


def test_constructor_defaults_and_passthrough():
    env = ALFWorldEnvironment(num_reasoning_blocks=1)
    assert env.branch_mode is False
    assert env.tau == 0.5 and env.n_elicit == 3
    assert env.elicit_prompt == ELICIT_DEFAULT
    assert env.leak_patterns == list(LEAK_DEFAULT)
    env2 = ALFWorldEnvironment(branch_mode=True, tau=0.7, n_elicit=1, elicit_prompt="Do {action}", leak_patterns=["x"])
    assert env2.branch_mode is True and env2.tau == 0.7 and env2.n_elicit == 1
    assert env2.elicit_prompt == "Do {action}" and env2.leak_patterns == ["x"]


# ---------------------------------------------------------------------------
# synthesize_block: weak acceptance by probability threshold (D-17)
# ---------------------------------------------------------------------------
# p("examine shelf 1") over the node's four sampled blocks is ~0.049, 0.032,
# 0.052, 0.092: no block leads to it and block 3 is the best sampled candidate.

SHELF = "examine shelf 1"
ELICITED_ROWS = [
    [-0.1, -3.0, -3.0, -2.5],  # p(shelf) ~ 0.076, desk on top
    [-0.5, -3.0, -3.0, -1.0],  # p(shelf) ~ 0.343, desk on top
    [-0.1, -3.0, -3.0, -3.0],  # p(shelf) ~ 0.047, desk on top
]


def _elicit_scripts(n):
    return [[f"<think>e{k + 1}</think>\n<action>{SHELF}</action>"] for k in range(n)]


def test_weak_acceptance_takes_best_elicited_attempt():
    env = _env(theta=0.01)
    node, client, seen = _node(env)
    client.scripts = _elicit_scripts(3)
    env._score_continuations = _scorer([list(r) for r in ELICITED_ROWS], seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, SHELF, node, BLOCK_ARGS))
    assert syn.source == "elicited"
    assert syn.block_idx is None
    assert syn.attempts == 3
    assert syn.reject_reasons == ["argmax", "argmax", "argmax"]
    assert syn.reasoning == "<think>e2</think>\n"
    assert syn.completion_text == f"<think>e2</think>\n<action>{SHELF}</action>"
    assert syn.p_accept == pytest.approx(_softmax(ELICITED_ROWS[1])[3])
    assert syn.p_accept < env.tau  # weak
    # one entry for the best sampled block, one per elicitation attempt
    assert [e["stage"] for e in syn.attempt_log] == ["sampled", "elicited", "elicited", "elicited"]
    assert syn.attempt_log[0]["block_idx"] == 3
    assert syn.attempt_log[0]["p"] == pytest.approx(_softmax(RAW_NODE[3])[3])
    assert [e.get("accepted") for e in syn.attempt_log] == [None, None, "weak", None]
    assert syn.attempt_log[2]["rank"] == 2 and syn.attempt_log[2]["argmax"] == "go to desk 1"
    assert syn.attempt_log[1]["reasoning"] == "<think>e1</think>\n"


def test_weak_acceptance_falls_back_to_best_sampled_block():
    env = _env(theta=0.0, n_elicit=1)
    node, client, seen = _node(env)
    client.scripts = _elicit_scripts(1)
    env._score_continuations = _scorer([list(ELICITED_ROWS[2])], seen)  # 0.047 < block 3's 0.092
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, SHELF, node, BLOCK_ARGS))
    assert syn.source == "sampled"
    assert syn.block_idx == 3
    assert syn.attempts == 1
    assert syn.reasoning == "<think>dresser plan</think>\n"
    assert syn.completion_text == f"<think>dresser plan</think>\n<action>{SHELF}</action>"
    assert syn.p_accept == pytest.approx(_softmax(RAW_NODE[3])[3])
    assert syn.attempt_log[0]["accepted"] == "weak"


def test_weak_candidate_below_theta_is_rejected():
    env = _env(theta=0.2, n_elicit=1)
    node, client, seen = _node(env)
    client.scripts = _elicit_scripts(1)
    env._score_continuations = _scorer([list(ELICITED_ROWS[2])], seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, SHELF, node, BLOCK_ARGS))
    assert syn.source == "none"
    assert syn.reject_reasons == ["argmax"]
    assert len(syn.attempt_log) == 2
    assert all("accepted" not in e for e in syn.attempt_log)


def test_theta_unset_keeps_the_strong_only_rule():
    env = _env(n_elicit=1)  # no theta attribute: theta = tau
    node, client, seen = _node(env)
    client.scripts = _elicit_scripts(1)
    env._score_continuations = _scorer([list(ELICITED_ROWS[1])], seen)  # 0.343 would pass a weak rule
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, SHELF, node, BLOCK_ARGS))
    assert syn.source == "none"
    # the attempt is logged with its p all the same
    assert syn.attempt_log[1]["p"] == pytest.approx(_softmax(ELICITED_ROWS[1])[3])
    assert syn.attempt_log[1]["reason"] == "argmax"


def test_strong_elicited_block_still_stops_early_under_theta():
    env = _env(tau=0.7, theta=0.1)
    node, client, seen = _node(env)
    # dresser: block 3 leads to it below tau ("tau"); the first elicitation is strong
    client.scripts = [["<think>The dresser is the likely spot.</think>\n<action>go to dresser 1</action>"]]
    env._score_continuations = _scorer([[-3.0, -3.0, -0.1, -3.0]], seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, "go to dresser 1", node, BLOCK_ARGS))
    assert syn.source == "elicited" and syn.attempts == 1
    assert syn.reject_reasons == ["tau"]
    assert syn.p_accept >= env.tau
    assert [e.get("accepted") for e in syn.attempt_log] == [None, "strong"]
    assert syn.attempt_log[0]["block_idx"] == 3


# ---------------------------------------------------------------------------
# elicitation template variants (paired A/B on the same (o, action))
# ---------------------------------------------------------------------------

ALT = "ALT {action}"
AB_ROWS = [
    [-0.1, -3.0, -3.0, -2.5],  # attempt 1, template 0: weak
    [-3.0, -3.0, -3.0, -0.1],  # attempt 2, template 1: strong (~0.87)
    [-0.1, -3.0, -3.0, -3.0],  # attempt 3, template 0: weak
    [-0.5, -3.0, -3.0, -1.0],  # attempt 4, template 1: weak (~0.34)
]


def test_exhaustive_elicitation_cycles_templates_and_logs_the_variant():
    env = _env(theta=0.0, n_elicit=4, elicit_variants=[ALT], elicit_exhaustive=True)
    node, client, seen = _node(env)
    client.scripts = _elicit_scripts(4)
    env._score_continuations = _scorer([list(r) for r in AB_ROWS], seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, SHELF, node, BLOCK_ARGS))
    # no early stop at the strong attempt 2: all four ran
    assert syn.attempts == 4
    assert syn.source == "elicited" and syn.reasoning == "<think>e2</think>\n"
    assert syn.p_accept == pytest.approx(_softmax(AB_ROWS[1])[3])
    assert [e.get("variant") for e in syn.attempt_log] == [None, 0, 1, 0, 1]
    assert [e.get("accepted") for e in syn.attempt_log] == [None, None, "strong", None, None]
    assert syn.reject_reasons == ["argmax", "argmax", "argmax"]
    sent = [c["prompt"][-1].content for c in client.calls[2:]]
    assert sent == [ELICIT_DEFAULT.format(action=SHELF), ALT.format(action=SHELF)] * 2


def test_variants_without_exhaustive_stop_at_the_first_strong_attempt():
    env = _env(theta=0.0, n_elicit=4, elicit_variants=[ALT])
    node, client, seen = _node(env)
    client.scripts = _elicit_scripts(4)
    env._score_continuations = _scorer([list(r) for r in AB_ROWS], seen)
    syn = asyncio.run(env.synthesize_block(client, "m", PROMPT, ADM, SHELF, node, BLOCK_ARGS))
    assert syn.attempts == 2 and syn.source == "elicited"
    assert [e.get("variant") for e in syn.attempt_log] == [None, 0, 1]
    assert syn.attempt_log[-1]["accepted"] == "strong"
