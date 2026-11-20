"""Tests for VaR / CVaR / Kupiec against analytic ground truth."""
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from risk_metrics import (  # noqa: E402
    expected_shortfall,
    kupiec_test,
    normal_var_cvar,
    value_at_risk,
)


def test_var_cvar_on_known_normal():
    """For N(0,1) losses, VaR99 = 2.3263 and CVaR99 = phi(z)/0.01 = 2.6652."""
    rng = np.random.default_rng(0)
    losses = rng.standard_normal(2_000_000)
    var, cvar = normal_var_cvar(0.99)
    assert var == pytest.approx(2.3263478740408408, rel=1e-9)
    assert cvar == pytest.approx(2.6652142208453746, rel=1e-9)
    assert value_at_risk(losses, 0.99) == pytest.approx(var, abs=0.01)
    assert expected_shortfall(losses, 0.99) == pytest.approx(cvar, abs=0.02)
    assert cvar > var


def test_var_cvar_small_sample_by_hand():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # 80th percentile by linear interpolation: 4 + 0.2*(100-4) = 23.2
    assert value_at_risk(losses, 0.80) == pytest.approx(23.2)
    assert expected_shortfall(losses, 0.80) == pytest.approx(100.0)


def test_kupiec_accepts_well_specified_model():
    rng = np.random.default_rng(123)
    losses = rng.standard_normal(1000)
    res = kupiec_test(losses, norm.ppf(0.99), alpha=0.99)
    assert res["violation_rate"] == pytest.approx(0.01, abs=0.01)
    assert res["p_value"] > 0.05
    assert res["reject_h0_5pct"] is False


def test_kupiec_rejects_heavy_tails():
    rng = np.random.default_rng(123)
    losses = rng.standard_t(3, size=1000) / np.sqrt(3)  # same variance, fat tails
    res = kupiec_test(losses, norm.ppf(0.99), alpha=0.99)
    assert res["reject_h0_5pct"] is True
    assert res["p_value"] < 0.05
