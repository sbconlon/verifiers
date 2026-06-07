"""Phase 2 (action-level ARM) -- RBMC marginal math.

Pure functions; no env/GPU/vLLM. Hand-verifiable softmax, mean, and
leave-one-out-mean examples, plus numerical stability and the m>=2 guard.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_rbmc.py
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.alfworld.rbmc import (  # noqa: E402
    conditional_renorm,
    rbmc_marginal,
    rbmc_marginal_loo,
)


def test_conditional_renorm_softmax():
    # logprobs [log 1, log 3] -> softmax [0.25, 0.75]
    out = conditional_renorm([math.log(1.0), math.log(3.0)])
    assert out == pytest.approx([0.25, 0.75])
    assert sum(out) == pytest.approx(1.0)


def test_conditional_renorm_uniform():
    out = conditional_renorm([0.0, 0.0, 0.0])
    assert out == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_conditional_renorm_single_action():
    assert conditional_renorm([-12.34]) == pytest.approx([1.0])


def test_conditional_renorm_numerical_stability():
    # Large magnitudes must not overflow; equal logprobs -> uniform.
    out = conditional_renorm([1000.0, 1001.0])
    assert all(math.isfinite(x) for x in out)
    assert sum(out) == pytest.approx(1.0)
    # exp(1000)/(exp(1000)+exp(1001)) = 1/(1+e) ; exp(1001)/... = e/(1+e)
    e = math.e
    assert out == pytest.approx([1 / (1 + e), e / (1 + e)])


def test_conditional_renorm_empty_raises():
    with pytest.raises(ValueError):
        conditional_renorm([])


def test_rbmc_marginal_mean():
    assert rbmc_marginal([0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_rbmc_marginal_empty_raises():
    with pytest.raises(ValueError):
        rbmc_marginal([])


def test_rbmc_marginal_loo_excludes_index():
    # exclude idx 0 -> mean of [0.1, 0.2] = 0.15
    assert rbmc_marginal_loo([0.9, 0.1, 0.2], exclude_idx=0) == pytest.approx(0.15)


def test_rbmc_marginal_loo_excludes_middle():
    assert rbmc_marginal_loo([0.3, 0.9, 0.6], exclude_idx=1) == pytest.approx(0.45)


def test_rbmc_marginal_loo_m2():
    # two blocks, exclude one -> the other's value exactly
    assert rbmc_marginal_loo([0.8, 0.2], exclude_idx=0) == pytest.approx(0.2)
    assert rbmc_marginal_loo([0.8, 0.2], exclude_idx=1) == pytest.approx(0.8)


def test_rbmc_marginal_loo_m1_raises():
    with pytest.raises(AssertionError):
        rbmc_marginal_loo([0.5], exclude_idx=0)


def test_rbmc_marginal_loo_bad_index_raises():
    with pytest.raises(IndexError):
        rbmc_marginal_loo([0.5, 0.3], exclude_idx=2)
