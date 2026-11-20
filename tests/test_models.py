"""Tests for Merton / Vasicek / GBM / Heston and Cholesky."""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fortran_backend import cholesky  # noqa: E402
from models import (  # noqa: E402
    gbm_terminal_mean,
    merton_pd,
    simulate_gbm,
    simulate_heston,
    simulate_merton_losses,
    simulate_vasicek_losses,
    vasicek_expected_loss,
    vasicek_var_analytic,
)
from risk_metrics import value_at_risk  # noqa: E402


def test_cholesky_reconstructs_correlation():
    rng = np.random.default_rng(0)
    # Build a correlation matrix that is strictly positive definite:
    # Cov = B B^T + 5I, then normalise to unit diagonal. (np.corrcoef of
    # only 5 observations would be singular, rank <= 4 — Cholesky
    # correctly refuses it, so it must not be used as the test input.)
    b = rng.standard_normal((5, 5))
    cov = b @ b.T + 5.0 * np.eye(5)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr = (corr + corr.T) / 2
    L = cholesky(corr)
    assert np.allclose(L @ L.T, corr, atol=1e-10)
    assert np.allclose(L, np.tril(L))  # lower-triangular


def test_cholesky_rejects_non_positive_definite():
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_merton_pd_matches_closed_form():
    # Deep in-the-money firm (V0 >> Debt) -> tiny PD; near Debt -> ~50%.
    # d2 = (ln(100/80) + 0.01875)/0.25 = 0.968 -> PD = Phi(-0.968) = 0.167
    assert merton_pd(100, 80, 0.05, 0.25) == pytest.approx(0.1666, abs=0.001)
    assert merton_pd(80, 80, 0.05, 0.25) == pytest.approx(0.47, abs=0.02)
    assert 0 < merton_pd(100, 80, 0.05, 0.25) < 1


def test_vasicek_expected_loss_identity():
    """EL = N * PD * LGD * EAD: simulation must match within MC error."""
    losses = simulate_vasicek_losses(100_000, 1000, pd=0.02, rho=0.15,
                                     lgd=0.45, ead=1.0, seed=42, antithetic=True)
    theory = vasicek_expected_loss(0.02, 0.45, 1.0, 1000)
    assert theory == pytest.approx(9.0)
    se = losses.std(ddof=1) / np.sqrt(len(losses))
    assert abs(losses.mean() - theory) < 5 * se + 0.1


def test_vasicek_var_converges_to_analytic_limit():
    losses = simulate_vasicek_losses(100_000, 1000, pd=0.02, rho=0.15,
                                     lgd=0.45, ead=1.0, seed=42)
    limit = vasicek_var_analytic(0.02, 0.15, 0.45, 1.0, 1000, alpha=0.99)
    assert value_at_risk(losses, 0.99) == pytest.approx(limit, rel=0.10)


def test_gbm_terminal_mean_and_schemes():
    theory = gbm_terminal_mean(100.0, 0.05, 1.0)
    assert theory == pytest.approx(105.127, abs=0.01)
    for scheme in ["exact", "euler", "milstein"]:
        paths = simulate_gbm(n_paths=50_000, n_steps=252, seed=7, scheme=scheme)
        assert paths.shape == (50_000, 253)
        assert np.all(paths[:, 0] == 100.0)
        # Monte Carlo error of the mean ~ sigma/sqrt(N) ~ 0.09; allow 1.0.
        assert abs(paths[:, -1].mean() - theory) < 1.0, scheme


def test_heston_variance_reverts_to_theta():
    _, var = simulate_heston(n_paths=20_000, n_steps=252, seed=7)
    assert var.shape == (20_000, 253)
    assert np.all(var >= 0)  # full truncation: no negative variance
    assert abs(var[:, -1].mean() - 0.04) < 0.005  # long-run mean = theta


def test_merton_losses_match_vasicek_scale():
    # Same PD (2%) and rho: Merton and Vasicek ELs must agree.
    # V0 = 131.2 gives Merton PD = 2%: ln(131.2/80)=0.495, d2=2.054.
    losses = simulate_merton_losses(20_000, 200, rho=0.15, v0=131.2, debt=80.0,
                                    mu=0.05, sigma=0.25, lgd=0.45, ead=1.0, seed=1)
    assert abs(losses.mean() - 200 * 0.02 * 0.45) < 0.5
