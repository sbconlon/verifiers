"""Rao-Blackwellized Monte Carlo (RBMC) marginal estimation for action-level ARM.

Pure functions, no infrastructure. The ALFWorld env computes π̂(a|o) at rollout
time (D-1): for each of m sampled reasoning blocks R_j it teacher-forces the
admissible actions to get the conditional π(·|o, R_j), then averages those
conditionals to estimate the marginal π(a|o) = E_{R~π}[π(a|o, R)]. For the
*executed* action a* the average is leave-one-out (excludes the block j* that
generated a*), removing the upward bias of including a*'s own reasoning.

These functions are defined and unit-tested in Phase 2 and wired into the env's
add_trajectory_step in Phase 4. Diary §6 is the source of truth for the math.
"""

from __future__ import annotations

import math


def conditional_renorm(action_logprobs: list[float]) -> list[float]:
    """π(·|o, R_j) over A(o) from summed action-token logprobs.

    action_logprobs[a] = Σ_t log π(a_t | o, R_j, a_<t), the (vocab-unnormalized)
    log-probability of admissible action a as a token continuation of [o, R_j].
    Returns the softmax over A(o): exp(lp_a) / Σ_a' exp(lp_a'). Numerically
    stable (subtract the max before exp). The result has the same length as the
    input and sums to 1.
    """
    if not action_logprobs:
        raise ValueError("action_logprobs must be non-empty")
    m = max(action_logprobs)
    exps = [math.exp(lp - m) for lp in action_logprobs]
    total = sum(exps)
    return [e / total for e in exps]


def rbmc_marginal(conditionals_per_block: list[float]) -> float:
    """Full RBMC marginal (1/m) Σ_j c_j for one target action.

    conditionals_per_block[j] = π(target | o, R_j), already renormalized over
    A(o). Used for non-executed actions (all m blocks contribute).
    """
    m = len(conditionals_per_block)
    if m == 0:
        raise ValueError("conditionals_per_block must be non-empty")
    return sum(conditionals_per_block) / m


def rbmc_marginal_loo(conditionals_per_block: list[float], exclude_idx: int) -> float:
    """Leave-one-out RBMC marginal (1/(m-1)) Σ_{j != exclude_idx} c_j.

    Produces π̂(a*|o) with exclude_idx = j* (the reasoning block that generated
    the executed action a*). Excluding R_{j*} removes the upward bias of the
    block whose sampling produced a* -- the same hold-out structure as RLOO.
    Requires m >= 2 (the marginal of a* is undefined from a single block).
    """
    m = len(conditionals_per_block)
    assert m >= 2, "leave-one-out RBMC requires m >= 2 reasoning blocks"
    if not 0 <= exclude_idx < m:
        raise IndexError(f"exclude_idx {exclude_idx} out of range for m={m}")
    held_out_sum = sum(conditionals_per_block) - conditionals_per_block[exclude_idx]
    return held_out_sum / (m - 1)
