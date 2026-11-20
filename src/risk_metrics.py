"""Tail-risk metrics and backtesting: VaR, Expected Shortfall, Kupiec.

Conventions: ``losses`` is a 1-D array of portfolio losses (positive =
loss). VaR_alpha is the alpha-quantile of losses; CVaR_alpha (Expected
Shortfall) is the mean loss conditional on exceeding VaR_alpha.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm


def value_at_risk(losses: np.ndarray, alpha: float = 0.99) -> float:
    """VaR_alpha = quantile_alpha(losses), NumPy 'linear' interpolation."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    losses = np.asarray(losses, dtype=float).ravel()
    if losses.size == 0:
        raise ValueError("losses is empty")
    return float(np.quantile(losses, alpha, method="linear"))


def expected_shortfall(losses: np.ndarray, alpha: float = 0.99) -> float:
    """CVaR_alpha = E[L | L > VaR_alpha]; equals VaR if no tail exceedance."""
    var = value_at_risk(losses, alpha)
    tail = np.asarray(losses, dtype=float).ravel()
    tail = tail[tail > var]
    return float(tail.mean()) if tail.size else var


def var_cvar(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    return value_at_risk(losses, alpha), expected_shortfall(losses, alpha)


def expected_loss(losses: np.ndarray) -> float:
    return float(np.mean(losses))


def kupiec_test(
    losses: np.ndarray, var: float, alpha: float = 0.99
) -> dict[str, float]:
    """Kupiec (1995) unconditional-coverage (POF) test for a VaR model.

    H0: the violation probability equals p = 1 - alpha.
    LR = -2 ln[ (1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x) ] ~ chi2(1)
    under H0, where x = # losses > VaR, n = # observations.
    Returns LR statistic, p-value, violation count/rate and a verdict
    at the 5% level. Edge cases x=0 or x=n are handled in the limit.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    n = losses.size
    x = int((losses > var).sum())
    p = 1.0 - alpha
    if n == 0:
        raise ValueError("losses is empty")
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        x_rate = x / n
        lr = -2.0 * (
            (n - x) * np.log(1.0 - p)
            + x * np.log(p)
            - (n - x) * np.log(1.0 - x_rate)
            - x * np.log(x_rate)
        )
    p_value = float(chi2.sf(max(lr, 0.0), df=1))
    return {
        "n": float(n),
        "violations": float(x),
        "violation_rate": x / n,
        "expected_rate": p,
        "lr_stat": float(lr),
        "p_value": p_value,
        "reject_h0_5pct": bool(p_value < 0.05),
    }


def normal_var_cvar(
    alpha: float = 0.99, mu: float = 0.0, sigma: float = 1.0,
) -> tuple[float, float]:
    """Analytic VaR / CVaR for a Normal(mu, sigma) loss distribution.

    VaR = mu + sigma Phi^{-1}(alpha)
    CVaR = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
    Used as the ground truth in tests and in the convergence script.
    """
    z = norm.ppf(alpha)
    var = mu + sigma * z
    cvar = mu + sigma * norm.pdf(z) / (1.0 - alpha)
    return float(var), float(cvar)


def violation_series(losses: np.ndarray, var: float) -> np.ndarray:
    """1/0 violation indicator series: 1 where loss exceeds VaR."""
    return (np.asarray(losses) > var).astype(int)
