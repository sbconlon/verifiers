"""Pure helpers for the branched value dataset (plan/branched-dataset, Phase 1).

No infrastructure: everything here is arithmetic and string handling over the
rows the ALFWorld env's pi_hat scorer already produces. The env wraps these in
score_node / synthesize_block / build_forced_step; the prime-rl driver
(branch-collect) orchestrates. Decisions referenced: D-4 (synthesis), D-5
(normalized acceptance rows), D-7 (a block leads to its row argmax), D-8
(prior slots + uniform tail).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Instruction appended as a user message when a forced action has no congruent
# sampled block (D-4). {action} is substituted. Never stored in a record.
ELICIT_DEFAULT = (
    "Your next action has been decided: {action}. Write your reasoning for this "
    "step in your usual format and end with exactly that action."
)

# Elicitation template marker: "<instruction><<prefill>><assistant prefix>".
# The part before it is appended as a user message (empty = no instruction);
# the part after it is prefilled as the start of the assistant turn and the
# model continues it (vLLM continue_final_message). Both take {action}.
PREFILL_MARK = "<<prefill>>"

# Case-insensitive regexes; an elicited block matching any is rejected ("leak").
LEAK_DEFAULT = [
    r"\bdecided\b",
    r"\btold\b",
    r"\binstruct",
    r"\bgiven action\b",
    r"\basked to\b",
    r"\bchosen for me\b",
    r"\bmust take\b",
    r"\bhas been decided\b",
]


@dataclass
class Block:
    """One response at a node: its reasoning (text before <action>), the action
    its own generation emitted (diagnostic, D-7), and the full completion."""

    reasoning: str
    tag_action: str
    completion_text: str


@dataclass
class NodeScore:
    """Output of ALFWorldEnvironment.score_node (branch-schema.md §1)."""

    blocks: list[Block]
    # Normalized conditional over `admissible` per block (D-5).
    rows: list[list[float]]
    # Admissible mass e^Z per block (recorded, not applied).
    masses: list[float]
    # Row argmax per block, as action text (D-7).
    lead: list[str]
    # Distinct lead actions, most blocks first, ties by block order.
    own: list[str]
    # Reasoning texts, one per block (what the scorer teacher-forces after).
    texts: list[str]
    admissible: list[str]
    # Natural-prompt token ids (from the node's own generations); elicited
    # blocks are verified under these.
    o_ids: list[int] = field(default_factory=list)


@dataclass
class Synthesis:
    """Output of ALFWorldEnvironment.synthesize_block (D-4)."""

    reasoning: str
    source: str  # "sampled" | "elicited" | "none"
    # Row value at the action for the returned block; < tau = weak (D-17).
    p_accept: float
    attempts: int
    reject_reasons: list[str]
    block_idx: int | None
    completion_text: str
    # One entry for the best sampled block and one per elicitation attempt,
    # whatever the outcome: stage, p (row value at the action), rank, argmax,
    # the reasoning text for elicited attempts, and "accepted" / "reason".
    attempt_log: list[dict] = field(default_factory=list)


def lead_actions(rows: list[list[float]]) -> list[int]:
    """Argmax index per row (first maximum on ties)."""
    out: list[int] = []
    for row in rows:
        if not row:
            raise ValueError("lead_actions: empty row")
        best = 0
        for i, v in enumerate(row):
            if v > row[best]:
                best = i
        out.append(best)
    return out


def ranked_distinct(lead_idx: list[int], admissible: list[str]) -> list[str]:
    """Distinct led-to actions, most frequent first; ties broken by the first
    block that led to them (so block 0, the greedy response, wins a tie)."""
    counts = Counter(lead_idx)
    first_pos = {}
    for pos, i in enumerate(lead_idx):
        first_pos.setdefault(i, pos)
    order = sorted(counts, key=lambda i: (-counts[i], first_pos[i]))
    return [admissible[i] for i in order]


def pick_actions(
    own: list[str],
    admissible: list[str],
    a_taken: str,
    k: int,
    n_unif: int,
    rng,
) -> list[tuple[str, str]]:
    """D-8: fill k - n_unif slots from the policy's own candidates, then draw
    uniformly from the admissible tail (actions no block led to). Unfilled own
    slots fall through to the tail. Returns (action, "own" | "tail") pairs."""
    if n_unif > k:
        raise ValueError("n_unif must be <= k")
    own_ok = []
    for a in own:
        if a in admissible and a != a_taken and a not in own_ok:
            own_ok.append(a)
    chosen = own_ok[: max(k - n_unif, 0)]
    # the tail excludes every own candidate (not just the chosen ones) so
    # the exploration term never re-draws a prior-term action
    tail = [a for a in admissible if a not in own_ok and a != a_taken]
    n_tail = min(k - len(chosen), len(tail))
    picks = rng.sample(tail, n_tail) if n_tail > 0 else []
    out = [(a, "own") for a in chosen]
    out += [(a, "tail") for a in picks]
    return out


def congruent(
    row: list[float], admissible: list[str], action: str, tau: float
) -> tuple[bool, float]:
    """D-4 acceptance under the natural prompt: the action must be the row's
    argmax and its normalized value must reach tau. Returns (ok, row[action])."""
    i = admissible.index(action)
    p = float(row[i])
    is_argmax = lead_actions([row])[0] == i
    return (is_argmax and p >= tau), p


def reject_reason(row: list[float], admissible: list[str], action: str, tau: float) -> str:
    """Why `congruent` failed: "argmax" when another action tops the row,
    else "tau" when the action tops it below the threshold."""
    i = admissible.index(action)
    return "tau" if lead_actions([row])[0] == i else "argmax"


def row_stats(row: list[float], admissible: list[str], action: str) -> tuple[float, int, str]:
    """(p, rank, argmax action) of ``action`` in a normalized row. Rank 1 is
    the top: 1 + the number of entries strictly above the action's value."""
    i = admissible.index(action)
    p = float(row[i])
    rank = 1 + sum(1 for v in row if v > p)
    return p, rank, admissible[lead_actions([row])[0]]


_LEAK_CACHE: dict[tuple[str, ...], list[re.Pattern]] = {}


def leaks(text: str, patterns: list[str]) -> bool:
    """True when any pattern matches the text (case-insensitive search)."""
    key = tuple(patterns)
    compiled = _LEAK_CACHE.get(key)
    if compiled is None:
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        _LEAK_CACHE[key] = compiled
    return any(p.search(text) for p in compiled)


def strip_to_reasoning(completion_text: str) -> str:
    """The completion's text before its first <action> tag, byte-exact (the
    scorer teacher-forces "<action>a</action>" right after it, so whitespace at
    the seam must be preserved). "" when there is no tag or the prefix is blank."""
    if "<action>" not in completion_text:
        return ""
    prefix = completion_text.split("<action>", 1)[0]
    return prefix if prefix.strip() else ""


def format_forced_completion(reasoning: str, action: str) -> str:
    """The stored assistant turn for a forced action: the reasoning followed by
    the action tag, exactly the string the scorer verified (D-3)."""
    return f"{reasoning}<action>{action}</action>"
