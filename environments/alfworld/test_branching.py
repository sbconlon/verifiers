"""Branched dataset Phase 1 -- the pure helpers in branching.py (D-4, D-7, D-8).

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_branching.py
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.alfworld.branching import (  # noqa: E402
    ELICIT_DEFAULT,
    LEAK_DEFAULT,
    congruent,
    format_forced_completion,
    lead_actions,
    leaks,
    pick_actions,
    ranked_distinct,
    reject_reason,
    strip_to_reasoning,
)

ADM = ["go to desk 1", "open drawer 1", "go to dresser 1", "examine shelf 1", "take pen 1"]


def test_lead_actions_argmax_first_on_ties():
    rows = [[0.1, 0.7, 0.2], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    assert lead_actions(rows) == [1, 0, 2]
    with pytest.raises(ValueError):
        lead_actions([[]])


def test_ranked_distinct_by_count_then_block_order():
    # 7 blocks -> desk, 1 -> drawer (block 2), 1 -> dresser (block 4)
    lead = [0, 0, 1, 0, 2, 0, 0, 0, 0]
    assert ranked_distinct(lead, ADM) == ["go to desk 1", "open drawer 1", "go to dresser 1"]
    # tie between two singletons: the earlier block wins
    assert ranked_distinct([2, 1], ADM) == ["go to dresser 1", "open drawer 1"]


def test_pick_actions_fills_prior_then_tail():
    own = ["go to desk 1", "open drawer 1", "go to dresser 1"]
    picks = pick_actions(own, ADM, a_taken="go to desk 1", k=4, n_unif=2, rng=random.Random(0))
    assert len(picks) == 4
    actions = [a for a, _ in picks]
    sources = [s for _, s in picks]
    assert actions[:2] == ["open drawer 1", "go to dresser 1"]
    assert sources == ["own", "own", "tail", "tail"]
    # tail: admissible, not an own candidate, not the trunk's action
    for a in actions[2:]:
        assert a in {"examine shelf 1", "take pen 1"}
    assert len(set(actions)) == 4


def test_pick_actions_falls_through_to_tail():
    picks = pick_actions(["open drawer 1"], ADM, a_taken="go to desk 1", k=4, n_unif=2, rng=random.Random(1))
    assert [s for _, s in picks] == ["own", "tail", "tail", "tail"]
    assert "go to desk 1" not in [a for a, _ in picks]


def test_pick_actions_filters_inadmissible_and_dedups():
    own = ["fly away", "open drawer 1", "open drawer 1"]
    picks = pick_actions(own, ADM, a_taken="examine shelf 1", k=2, n_unif=0, rng=random.Random(0))
    # one own candidate survives; the unfilled own slot falls through to the tail (D-8)
    assert picks[0] == ("open drawer 1", "own")
    assert len(picks) == 2 and picks[1][1] == "tail"
    assert picks[1][0] in {"go to desk 1", "go to dresser 1", "take pen 1"}


def test_pick_actions_deterministic_and_bounded():
    a = pick_actions([], ADM, "go to desk 1", 4, 4, random.Random(7))
    b = pick_actions([], ADM, "go to desk 1", 4, 4, random.Random(7))
    assert a == b
    assert all(s == "tail" for _, s in a)
    # k larger than the tail: every tail action once, no repeats
    c = pick_actions([], ADM, "go to desk 1", 10, 10, random.Random(0))
    assert sorted(x for x, _ in c) == sorted(set(ADM) - {"go to desk 1"})
    with pytest.raises(ValueError):
        pick_actions([], ADM, "go to desk 1", 2, 3, random.Random(0))


def test_congruent_and_reject_reason():
    row = [0.55, 0.30, 0.10, 0.03, 0.02]
    ok, p = congruent(row, ADM, "go to desk 1", tau=0.5)
    assert ok and p == pytest.approx(0.55)
    ok, p = congruent(row, ADM, "open drawer 1", tau=0.5)
    assert not ok and p == pytest.approx(0.30)
    assert reject_reason(row, ADM, "open drawer 1", 0.5) == "argmax"
    ok, _ = congruent(row, ADM, "go to desk 1", tau=0.6)
    assert not ok
    assert reject_reason(row, ADM, "go to desk 1", 0.6) == "tau"


def test_leak_filter():
    assert leaks("<think>I was told to open the drawer.</think>\n", LEAK_DEFAULT)
    assert leaks("<think>The action has been decided for me</think>", LEAK_DEFAULT)
    assert leaks("<think>Following the INSTRUCTION, I go</think>", LEAK_DEFAULT)
    # a plain plan does not leak
    assert not leaks("<think>The pen is probably in the drawer, so I open it.</think>\n", LEAK_DEFAULT)
    # documented behavior: "decide" in another sense is NOT caught (\bdecided\b only)
    assert not leaks("<think>I decide to look in the drawer.</think>", LEAK_DEFAULT)
    assert leaks("anything", [r".*"])


def test_strip_to_reasoning_is_byte_exact_prefix():
    text = "<think>plan</think>\n<action>go to desk 1</action>"
    assert strip_to_reasoning(text) == "<think>plan</think>\n"
    assert strip_to_reasoning("no tag here") == ""
    assert strip_to_reasoning("  \n<action>look</action>") == ""
    # the first tag splits, later text is dropped
    assert strip_to_reasoning("<think>a</think><action>x</action> trailing <action>y</action>") == "<think>a</think>"


def test_format_forced_completion_matches_scored_string():
    reasoning = "<think>plan</think>\n"
    out = format_forced_completion(reasoning, "open drawer 1")
    assert out == "<think>plan</think>\n<action>open drawer 1</action>"
    # round trip: stripping the formatted completion gives the reasoning back
    assert strip_to_reasoning(out) == reasoning


def test_elicit_default_has_action_slot():
    assert "{action}" in ELICIT_DEFAULT
    assert "open drawer 1" in ELICIT_DEFAULT.format(action="open drawer 1")
