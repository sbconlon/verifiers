"""a*-only pi_hat scoring on the training path (pi_hat_renorm=False, D-13).

With pi_hat_renorm=False the per-block value at a* is the raw sequence
probability exp(summed logp) and depends on no other candidate, so
_compute_pi_hat scores only "<action>a*</action>" under the m-1 held-out blocks
(_score_pi_hat_astar) instead of the full m x |A| matrix. Pins: numeric
equality with the matrix path (1e-12) over m in {2, 4}, |A| in {2, 5}, a* in /
not in the admissible set, every j*; the request shape (m-1 prompts, ONE call,
the a* suffix, j* absent, block order); the pre-warm (m-1 prefixes, none when
disabled); the scoring-failure degrade; and that pi_hat_renorm=True, probe
capture, and a direct probe_capture=True call still issue m x |A| prompts and
keep the full matrix.

The fake scorer answers by prompt CONTENT (block index + action decoded from the
ord-encoded prompt), so the same (block, action) pair scores the same number
whichever prompt set a path builds -- that is what makes the two paths
comparable on one fixture.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_pi_hat_score_astar.py
"""
import asyncio
import math
import random
import re
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
from environments.alfworld.rbmc import rbmc_marginal, rbmc_marginal_loo  # noqa: E402


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _enc(text):
    return [ord(c) for c in text]


def _dec(ids):
    return "".join(chr(t) for t in ids)


O_IDS = [1, 2]  # the fake observation prefix every _Block carries


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
        self.message = _Msg(content, _Toks(list(O_IDS), list(range(8))))


def _reasoning(j):
    return f"<think>r{j}</think>"


def _blocks(m, action="look"):
    return [_Block(f"{_reasoning(j)}<action>{action}</action>") for j in range(m)]


def _completion(action):
    return [_AssistantMessage(f"<think>r</think><action>{action}</action>")]


_PROMPT_RE = re.compile(r"<think>r(\d+)</think><action>(.*)</action>", re.DOTALL)


def _scorer_by_content(raw, candidates):
    """Fake _score_continuations keyed on prompt content: prompt = O_IDS +
    ord("<think>r{j}</think><action>{a}</action>") scores raw[j][candidates.index(a)].
    Records every call (prompts + spans)."""
    calls = []

    async def fake(client, prompts, spans, model, temperature=1.0):
        calls.append({"prompts": [list(p) for p in prompts], "spans": list(spans)})
        out = []
        for p in prompts:
            assert list(p[: len(O_IDS)]) == O_IDS
            m = _PROMPT_RE.fullmatch(_dec(p[len(O_IDS) :]))
            assert m is not None, _dec(p[len(O_IDS) :])
            out.append(raw[int(m.group(1))][candidates.index(m.group(2))])
        return out

    return fake, calls


async def _must_not_run(*args, **kwargs):
    raise AssertionError("_score_pi_hat_astar must not run on this path")


def _env(m, *, probe=False, renorm=False, scorer=None, prewarm=False):
    env = ALFWorldEnvironment.__new__(ALFWorldEnvironment)
    env.num_reasoning_blocks = m
    env.probe_capture = probe
    env.pi_hat_renorm = renorm
    env.prewarm_pi_hat_prefixes = prewarm
    env.probe_block_temperature = 0.8
    env.probe_block_min_p = 0.1
    env.store_raw_logps = False
    env.oracle = "off"
    env.use_nm_fusion = False
    env._tokenizer = _FakeTok()
    if scorer is not None:
        env._score_continuations = scorer
    return env


def _state(admissible, blocks, j_star, client=None):
    return {
        "trajectory": [],
        "_last_admissible_commands": list(admissible),
        "_pending_reasoning_blocks": blocks,
        "_executed_block_idx": j_star,
        "client": client if client is not None else object(),
        "model": "m",
    }


def _bare_state():
    return {"client": object(), "model": "m"}


ACTIONS = ["go to desk 1", "look", "take pen 1 from desk 1", "open drawer 1", "examine desk 1"]
A_STAR_IN = "look"  # in every admissible prefix of ACTIONS of size >= 2
A_STAR_OUT = "open fridge 1"  # never admissible -> appended to the union


def _fixture(m, n_adm, a_star, seed):
    """Admissible set ACTIONS[:n_adm], the union the env builds for a*, and an
    irregular m x |union| raw logp matrix in [-4, -0.05] (physically valid rows
    are not needed: both paths exponentiate the same numbers)."""
    admissible = ACTIONS[:n_adm]
    candidates, _ = ALFWorldEnvironment._admissible_union(admissible, a_star)
    rng = random.Random(seed)
    raw = [[rng.uniform(-4.0, -0.05) for _ in candidates] for _ in range(m)]
    return admissible, candidates, raw


def _run_training(env, admissible, blocks, j_star, a_star, client=None):
    state = _state(admissible, blocks, j_star, client)
    step = {"completion": _completion(a_star), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))
    return state, step


def _matrix_path_value(env, admissible, blocks, j_star, a_star):
    """The pre-shortcut _compute_pi_hat, verbatim: full matrix via the untouched
    _score_pi_hat_matrix, then the LOO (or plain) mean of the a* column."""
    mat = asyncio.run(env._score_pi_hat_matrix(_bare_state(), blocks, admissible, a_star))
    assert mat.renormalized is False
    col = [row[mat.a_star_idx] for row in mat.cond]
    return rbmc_marginal(col) if j_star is None else rbmc_marginal_loo(col, exclude_idx=j_star)


def _softmax(row):
    mx = max(row)
    ex = [math.exp(v - mx) for v in row]
    s = sum(ex)
    return [e / s for e in ex]


# ---------------------------------------------------------------------------
# (a) numeric equality with the matrix path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_star", [A_STAR_IN, A_STAR_OUT])
@pytest.mark.parametrize("n_adm", [2, 5])
@pytest.mark.parametrize("m", [2, 4])
def test_astar_path_equals_matrix_path(m, n_adm, a_star):
    seed = 100 * m + 10 * n_adm + (a_star == A_STAR_IN)
    admissible, candidates, raw = _fixture(m, n_adm, a_star, seed)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=False, scorer=scorer)
    a_idx = candidates.index(a_star)

    for j_star in range(m):
        state, step = _run_training(env, admissible, _blocks(m, a_star), j_star, a_star)
        got = step["extras"]["pi_hat"]
        expected = _matrix_path_value(env, admissible, _blocks(m, a_star), j_star, a_star)
        assert got == pytest.approx(expected, abs=1e-12)
        # independent closed form (no env code): raw-probability LOO mean at a*
        closed = sum(math.exp(raw[j][a_idx]) for j in range(m) if j != j_star) / (m - 1)
        assert got == pytest.approx(closed, abs=1e-12)
        # the rest of the training-path contract is untouched
        assert "probe" not in step["extras"]
        assert step["extras"]["admissible_actions"] == admissible
        assert step["extras"]["executed_action"] == a_star
        assert "_pending_reasoning_blocks" not in state and "_executed_block_idx" not in state
        assert state["trajectory"] == [step]

    # j_star None (direct call, the probe estimator): plain mean over all m blocks
    val = asyncio.run(env._compute_pi_hat(_bare_state(), _blocks(m, a_star), None, admissible, a_star))
    assert val == pytest.approx(_matrix_path_value(env, admissible, _blocks(m, a_star), None, a_star), abs=1e-12)
    assert val == pytest.approx(sum(math.exp(raw[j][a_idx]) for j in range(m)) / m, abs=1e-12)


def test_golden_fixture_from_renorm_tests():
    """Same RAW4 / j*=2 / a* appended at idx 3 as test_pi_hat_renorm.py::test_training_path_raw_loo."""
    raw4 = [
        [-1.3, -0.2, -2.7, -0.9],
        [-0.4, -1.1, -0.6, -3.2],
        [-2.2, -2.0, -0.1, -1.5],
        [-0.7, -0.7, -1.9, -0.3],
    ]
    admissible = ["go to desk 1", "look", "take pen 1 from desk 1"]
    a_star = "open drawer 1"
    candidates = admissible + [a_star]
    scorer, calls = _scorer_by_content(raw4, candidates)
    env = _env(4, renorm=False, scorer=scorer)
    _, step = _run_training(env, admissible, _blocks(4, a_star), 2, a_star)
    expected = sum(math.exp(raw4[j][3]) for j in (0, 1, 3)) / 3
    assert step["extras"]["pi_hat"] == pytest.approx(expected, abs=1e-12)
    assert len(calls) == 1 and len(calls[0]["prompts"]) == 3


# ---------------------------------------------------------------------------
# (b) request shape: m-1 prompts, one call, a* suffix, j* absent, block order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,j_star", [(2, 0), (2, 1), (4, 0), (4, 2), (4, 3)])
def test_training_request_is_m_minus_1_astar_prompts(m, j_star):
    admissible, candidates, raw = _fixture(m, 5, A_STAR_IN, seed=7)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=False, scorer=scorer)
    _run_training(env, admissible, _blocks(m), j_star, A_STAR_IN)

    assert len(calls) == 1  # ONE scoring request per decision point, as before
    prompts, spans = calls[0]["prompts"], calls[0]["spans"]
    assert len(prompts) == m - 1  # not m x |A|
    suffix = _enc("<action>" + A_STAR_IN + "</action>")
    kept = [j for j in range(m) if j != j_star]
    for p, (s, e), j in zip(prompts, spans, kept):
        assert p[-len(suffix) :] == suffix  # every prompt ends with <action>a*</action>
        assert p == O_IDS + _enc(_reasoning(j)) + suffix  # block order, j* skipped
        # span = the action continuation only (no BPE seam with the ord tokenizer)
        assert (s, e) == (len(O_IDS) + len(_enc(_reasoning(j))), len(p))
    assert all(_reasoning(j_star) not in _dec(p) for p in prompts)
    # no other admissible action was ever scored
    assert all(_dec(p).endswith("<action>look</action>") for p in prompts)


def test_m2_scores_a_single_prompt():
    admissible, candidates, raw = _fixture(2, 2, A_STAR_OUT, seed=2)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(2, renorm=False, scorer=scorer)
    _, step = _run_training(env, admissible, _blocks(2, A_STAR_OUT), 1, A_STAR_OUT)
    assert len(calls[0]["prompts"]) == 1
    assert step["extras"]["pi_hat"] == pytest.approx(math.exp(raw[0][2]), abs=1e-12)


def test_direct_call_without_j_star_scores_all_m():
    m = 4
    admissible, candidates, raw = _fixture(m, 3, A_STAR_IN, seed=4)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=False, scorer=scorer)
    asyncio.run(env._compute_pi_hat(_bare_state(), _blocks(m), None, admissible, A_STAR_IN))
    assert len(calls) == 1 and len(calls[0]["prompts"]) == m
    assert [_dec(p[len(O_IDS) :]) for p in calls[0]["prompts"]] == [
        f"{_reasoning(j)}<action>look</action>" for j in range(m)
    ]


# ---------------------------------------------------------------------------
# pre-warm: same gating, only the scored prefixes
# ---------------------------------------------------------------------------


class _FakeTokenClient:
    def __init__(self):
        self.posts = []

    async def post(self, path, body, cast_to=None):
        self.posts.append({"path": path, "body": body})

        class _Empty:
            choices: list = []

        return _Empty()


class _FakeClient:
    def __init__(self):
        self.token_client = _FakeTokenClient()


def test_prewarm_only_the_scored_prefixes():
    m, j_star = 4, 1
    admissible, candidates, raw = _fixture(m, 3, A_STAR_IN, seed=3)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=False, scorer=scorer, prewarm=True)
    client = _FakeClient()
    _, step = _run_training(env, admissible, _blocks(m), j_star, A_STAR_IN, client=client)
    # one pre-warm request carrying the m-1 [o, R_j] prefixes of the scored blocks
    assert len(client.token_client.posts) == 1
    post = client.token_client.posts[0]
    assert post["path"] == "/v1/completions"
    assert post["body"]["prompt"] == [O_IDS + _enc(_reasoning(j)) for j in range(m) if j != j_star]
    assert post["body"]["max_tokens"] == 1
    assert "prompt_logprobs" not in post["body"]
    # then the single scoring call
    assert len(calls) == 1 and len(calls[0]["prompts"]) == m - 1
    assert "pi_hat" in step["extras"]


def test_prewarm_disabled():
    m = 3
    admissible, candidates, raw = _fixture(m, 3, A_STAR_IN, seed=3)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=False, scorer=scorer, prewarm=False)
    client = _FakeClient()
    _run_training(env, admissible, _blocks(m), 0, A_STAR_IN, client=client)
    assert client.token_client.posts == []
    assert len(calls) == 1


def test_prewarm_failure_is_best_effort():
    class _BoomTokenClient:
        async def post(self, path, body, cast_to=None):
            raise RuntimeError("prewarm down")

    class _BoomClient:
        token_client = _BoomTokenClient()

    m = 3
    admissible, candidates, raw = _fixture(m, 3, A_STAR_IN, seed=6)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=False, scorer=scorer, prewarm=True)
    _, step = _run_training(env, admissible, _blocks(m), 0, A_STAR_IN, client=_BoomClient())
    assert len(calls) == 1
    a_idx = candidates.index(A_STAR_IN)
    assert step["extras"]["pi_hat"] == pytest.approx(sum(math.exp(raw[j][a_idx]) for j in (1, 2)) / 2, abs=1e-12)


# ---------------------------------------------------------------------------
# degrade: a scoring failure leaves pi_hat unset, exactly like the matrix path
# ---------------------------------------------------------------------------


def test_astar_path_scoring_failure_degrades():
    async def boom(client, prompts, spans, model, temperature=1.0):
        raise RuntimeError("score route down")

    env = _env(4, renorm=False, scorer=boom)
    state, step = _run_training(env, ACTIONS[:3], _blocks(4), 0, A_STAR_IN)
    assert "pi_hat" not in step["extras"]
    assert step["extras"]["executed_action"] == A_STAR_IN
    assert step["extras"]["admissible_actions"] == ACTIONS[:3]
    assert "_pending_reasoning_blocks" not in state and "_executed_block_idx" not in state
    assert state["trajectory"] == [step]


# ---------------------------------------------------------------------------
# (c) pi_hat_renorm=True still scores the full m x |A| matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_star", [A_STAR_IN, A_STAR_OUT])
def test_renorm_true_still_scores_full_matrix(a_star):
    m, n_adm, j_star = 4, 5, 2
    admissible, candidates, raw = _fixture(m, n_adm, a_star, seed=11)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=True, scorer=scorer)
    env._score_pi_hat_astar = _must_not_run
    _, step = _run_training(env, admissible, _blocks(m, a_star), j_star, a_star)
    assert len(calls) == 1
    assert len(calls[0]["prompts"]) == m * len(candidates)  # m x |A| (union)
    a_idx = candidates.index(a_star)
    expected = sum(_softmax(raw[j])[a_idx] for j in range(m) if j != j_star) / (m - 1)
    assert step["extras"]["pi_hat"] == pytest.approx(expected, abs=1e-12)


def test_renorm_attribute_absent_defaults_to_matrix():
    """Envs built without the attribute (older fixtures) keep the old convention."""
    m = 3
    admissible, candidates, raw = _fixture(m, 3, A_STAR_IN, seed=12)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, renorm=True, scorer=scorer)
    del env.pi_hat_renorm
    del env.probe_capture
    _, step = _run_training(env, admissible, _blocks(m), 0, A_STAR_IN)
    assert len(calls[0]["prompts"]) == m * len(candidates)
    assert "pi_hat" in step["extras"]


# ---------------------------------------------------------------------------
# (d) probe capture keeps the full matrix with pi_hat_renorm=False
# ---------------------------------------------------------------------------


def test_probe_capture_keeps_full_matrix():
    m, n_adm = 4, 5
    admissible, candidates, raw = _fixture(m, n_adm, A_STAR_IN, seed=5)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, probe=True, renorm=False, scorer=scorer)
    env._score_pi_hat_astar = _must_not_run
    state = _state(admissible, _blocks(m), None)
    step = {"prompt": [], "completion": _completion(A_STAR_IN), "extras": {}}
    asyncio.run(env.add_trajectory_step(state, step))

    assert len(calls) == 1 and len(calls[0]["prompts"]) == m * len(candidates)
    p = step["extras"]["probe"]
    assert p["pi_hat_renorm"] is False
    assert p["admissible_union"] == candidates
    assert len(p["blocks"]) == m
    for j, b in enumerate(p["blocks"]):
        assert len(b["cond"]) == len(candidates)
        assert b["cond"] == pytest.approx([math.exp(v) for v in raw[j]], abs=1e-12)
    assert len(p["Z"]) == m
    a_idx = candidates.index(A_STAR_IN)
    marginal = [sum(math.exp(raw[j][a]) for j in range(m)) / m for a in range(len(candidates))]
    assert p["pi_hat_marginal"] == pytest.approx(marginal, abs=1e-12)
    assert step["extras"]["pi_hat"] == pytest.approx(marginal[a_idx], abs=1e-12)


def test_compute_pi_hat_with_probe_capture_set_uses_matrix():
    """The guard in _compute_pi_hat itself: probe_capture=True + pi_hat_renorm=False
    on a direct call still goes through the full matrix."""
    m, j_star = 3, 1
    admissible, candidates, raw = _fixture(m, 3, A_STAR_IN, seed=9)
    scorer, calls = _scorer_by_content(raw, candidates)
    env = _env(m, probe=True, renorm=False, scorer=scorer)
    env._score_pi_hat_astar = _must_not_run
    val = asyncio.run(env._compute_pi_hat(_bare_state(), _blocks(m), j_star, admissible, A_STAR_IN))
    assert len(calls) == 1 and len(calls[0]["prompts"]) == m * len(candidates)
    a_idx = candidates.index(A_STAR_IN)
    assert val == pytest.approx(sum(math.exp(raw[j][a_idx]) for j in (0, 2)) / 2, abs=1e-12)
