"""Stochastic models: Merton, Vasicek, GBM, Heston.

All simulators share conventions:
- ``rng`` is a ``numpy.random.Generator`` (``np.random.default_rng(seed)``);
  passing an explicit seed makes every result reproducible.
- Losses are in currency units: loss = defaults * LGD * EAD.
- Returns are arrays; no function prints or writes files (callers do that).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Merton (1974) — structural model
# ---------------------------------------------------------------------------
def merton_pd(v0: float, debt: float, mu: float, sigma: float, t: float = 1.0) -> float:
    """Probability of default from the Merton structural model.

    Firm asset value follows GBM; default iff V_T < Debt, i.e.
        PD = Phi(-d2),  d2 = [ln(V0/D) + (mu - sigma^2/2) T] / (sigma sqrt(T))
    """
    if sigma <= 0 or t <= 0 or v0 <= 0 or debt <= 0:
        raise ValueError("v0, debt, sigma, t must be positive")
    d2 = (np.log(v0 / debt) + (mu - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return float(norm.cdf(-d2))


def merton_d2(v0: float, debt: float, mu: float, sigma: float, t: float = 1.0) -> float:
    """Distance-to-default d2 (same formula as :func:`merton_pd`)."""
    if sigma <= 0 or t <= 0 or v0 <= 0 or debt <= 0:
        raise ValueError("v0, debt, sigma, t must be positive")
    d2 = (np.log(v0 / debt) + (mu - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return float(d2)


def simulate_merton_losses(
    n_scenarios: int,
    n_obligors: int,
    corr: np.ndarray | None = None,
    rho: float = 0.15,
    v0: float = 100.0,
    debt: float = 80.0,
    mu: float = 0.05,
    sigma: float = 0.25,
    t: float = 1.0,
    lgd: float = 0.45,
    ead: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate portfolio losses under Merton with correlated asset returns.

    Standardised asset returns X ~ N(0, Corr); obligor i defaults iff
    X_i < Phi^{-1}(PD_i). If ``corr`` is None, an equicorrelation matrix
    with off-diagonal ``rho`` is used (Cholesky factorisation).
    """
    rng = np.random.default_rng(seed)
    pd = merton_pd(v0, debt, mu, sigma, t)
    threshold = norm.ppf(pd)
    if corr is None:
        # Equicorrelation fast path — mathematically identical to the
        # Cholesky product, O(N_scen * N) instead of O(N_scen * N^2):
        #   X_i = sqrt(rho) * Y + sqrt(1 - rho) * eps_i,
        #   Y, eps_i i.i.d. N(0,1).  Corr(X_i, X_j) = rho for i != j.
        # (This is also why Merton with equicorrelation and Vasicek
        #  agree: both are single-factor models.)
        y = rng.standard_normal((n_scenarios, 1))
        eps = rng.standard_normal((n_scenarios, n_obligors))
        x = np.sqrt(rho) * y + np.sqrt(1.0 - rho) * eps
    else:
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (n_obligors, n_obligors):
            raise ValueError(f"corr must be ({n_obligors}, {n_obligors})")
        chol = np.linalg.cholesky(corr)  # raises LinAlgError if not PD
        z = rng.standard_normal((n_scenarios, n_obligors))
        x = z @ chol.T
    defaults = (x < threshold).sum(axis=1)
    return defaults.astype(float) * lgd * ead


# ---------------------------------------------------------------------------
# Vasicek (1987/2002) — single-factor correlated defaults
# ---------------------------------------------------------------------------
def vasicek_conditional_pd(pd: float, rho: float, y: np.ndarray) -> np.ndarray:
    """p(Y) = Phi((Phi^{-1}(PD) - sqrt(rho) Y) / sqrt(1 - rho))."""
    return norm.cdf((norm.ppf(pd) - np.sqrt(rho) * y) / np.sqrt(1.0 - rho))


def vasicek_expected_loss(pd: float, lgd: float, ead: float, n_obligors: int) -> float:
    """EL = N * PD * LGD * EAD (unconditional mean of the Vasicek loss)."""
    return float(n_obligors * pd * lgd * ead)


def vasicek_var_analytic(
    pd: float, rho: float, lgd: float, ead: float, n_obligors: int, alpha: float = 0.99
) -> float:
    """Large-homogeneous-portfolio VaR (Vasicek limit, N -> infinity).

    VaR_alpha = N * LGD * EAD * Phi((Phi^{-1}(PD) + sqrt(rho) Phi^{-1}(alpha))
                                     / sqrt(1 - rho))
    This is the Basel II IRB asymptotic formula; finite-N simulation
    should converge to it as N grows — used as a validation target.
    """
    q = norm.cdf(
        (norm.ppf(pd) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1.0 - rho)
    )
    return float(n_obligors * lgd * ead * q)


def simulate_vasicek_losses(
    n_scenarios: int,
    n_obligors: int,
    pd: float = 0.02,
    rho: float = 0.15,
    lgd: float = 0.45,
    ead: float = 1.0,
    seed: int = 42,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate Vasicek portfolio losses.

    Per scenario: Y ~ N(0,1) systemic factor -> conditional PD p(Y) ->
    defaults ~ Binomial(N, p(Y)). With ``antithetic=True`` the second
    half of scenarios uses -Y, which cancels the first-order Monte
    Carlo error in the systemic factor.
    """
    if not 0 < pd < 1 or not 0 <= rho < 1:
        raise ValueError("need 0 < pd < 1 and 0 <= rho < 1")
    rng = np.random.default_rng(seed)
    n_half = n_scenarios // 2 if antithetic else n_scenarios
    y = rng.standard_normal(n_half)
    if antithetic:
        y = np.concatenate([y, -y])
        if len(y) < n_scenarios:  # odd n_scenarios: one extra draw
            y = np.concatenate([y, rng.standard_normal(1)])
    p_cond = vasicek_conditional_pd(pd, rho, y)
    defaults = rng.binomial(n_obligors, p_cond)
    return defaults.astype(float) * lgd * ead


# ---------------------------------------------------------------------------
# GBM — geometric Brownian motion
# ---------------------------------------------------------------------------
def simulate_gbm(
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.20,
    t: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: int = 42,
    scheme: str = "exact",
) -> np.ndarray:
    """Simulate GBM paths, shape (n_paths, n_steps + 1), column 0 = S0.

    Schemes:
    - ``exact``: log-price is simulated exactly (no discretisation error).
    - ``euler``: Euler-Maruyama,  S += S (mu dt + sigma sqrt(dt) Z).
    - ``milstein``: Euler + 0.5 sigma^2 S ((sqrt(dt) Z)^2 - dt).
      For GBM the Milstein term is exact in the diffusion coefficient,
      so Milstein converges faster (strong order 1.0 vs 0.5 for Euler).
    """
    if scheme not in {"exact", "euler", "milstein"}:
        raise ValueError("scheme must be 'exact', 'euler' or 'milstein'")
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    if scheme == "exact":
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        paths[:, 1:] = s0 * np.exp(np.cumsum(increments, axis=1))
    else:
        s = np.full(n_paths, s0)
        for i in range(n_steps):
            dw = np.sqrt(dt) * z[:, i]
            if scheme == "euler":
                s = s + s * (mu * dt + sigma * dw)
            else:  # milstein
                s = s + s * (mu * dt + sigma * dw) + 0.5 * sigma**2 * s * (dw**2 - dt)
            paths[:, i + 1] = s
    return paths


def gbm_terminal_mean(s0: float, mu: float, t: float) -> float:
    return float(s0 * np.exp(mu * t))


# ---------------------------------------------------------------------------
# Heston (1993) — stochastic volatility
# ---------------------------------------------------------------------------
def simulate_heston(
    s0: float = 100.0,
    v0: float = 0.04,
    mu: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.30,
    rho: float = -0.7,
    t: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: int = 42,
    scheme: str = "euler",
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths (price, variance), each (n_paths, n_steps+1).

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,  corr(dW1, dW2) = rho
    Full-truncation Euler: v is floored at 0 before taking sqrt(v),
    which keeps the scheme stable when the Feller condition
    2 kappa theta > xi^2 is violated. Milstein adds the variance
    correction 0.25 xi^2 ((sqrt(dt) Z2)^2 - dt) to the v-step.
    """
    if scheme not in {"euler", "milstein"}:
        raise ValueError("scheme must be 'euler' or 'milstein'")
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    z1 = rng.standard_normal((n_paths, n_steps))
    z2_indep = rng.standard_normal((n_paths, n_steps))
    z2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2_indep
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)
    price = np.empty((n_paths, n_steps + 1))
    var = np.empty((n_paths, n_steps + 1))
    price[:, 0], var[:, 0] = s0, v0
    for i in range(n_steps):
        dw1 = np.sqrt(dt) * z1[:, i]
        dw2 = np.sqrt(dt) * z2[:, i]
        v_pos = np.maximum(v, 0.0)
        s = s + mu * s * dt + np.sqrt(v_pos) * s * dw1
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dw2
        if scheme == "milstein":
            v = v + 0.25 * xi**2 * (dw2**2 - dt)
        v = np.maximum(v, 0.0)
        price[:, i + 1], var[:, i + 1] = s, v
    return price, var
