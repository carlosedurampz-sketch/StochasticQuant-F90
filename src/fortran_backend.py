"""Fortran backend with a NumPy fallback — one API, two engines.

If ``fortran/montecarlo`` was compiled with f2py (``make fortran``),
its routines are used. Otherwise this module reproduces the *same*
function signatures with NumPy, so every script, test and benchmark
runs on a machine without gfortran. ``FORTRAN_AVAILABLE`` tells the
caller which engine is actually running; benchmarks label it honestly
("fortran" vs "numpy-fallback") instead of pretending.

Fortran argument order is preserved (f2py exposes the subroutines
positionally), e.g. ``simulate_gbm(n_paths, n_steps, s0, mu, sigma,
t, seed, antithetic)``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Compiled with:  cd fortran && f2py -c -m montecarlo montecarlo.f90
    # (the .so is placed next to this file by the Makefile)
    import montecarlo as _f90  # type: ignore

    FORTRAN_AVAILABLE = True
    BACKEND = "fortran"
except Exception:  # ImportError, missing .so, ABI mismatch, ...
    _f90 = None
    FORTRAN_AVAILABLE = False
    BACKEND = "numpy-fallback"


def cholesky(a: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor L with A = L L^T (same as Fortran cholesky)."""
    a = np.asarray(a, dtype=float)
    if FORTRAN_AVAILABLE:
        l, info = _f90.montecarlo.cholesky(a)
        if info != 0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        return np.asarray(l, dtype=float)
    return np.linalg.cholesky(a)


def simulate_gbm_fortran(
    n_paths: int, n_steps: int, s0: float, mu: float, sigma: float,
    t: float, seed: int, antithetic: bool = False,
) -> tuple[np.ndarray, float, float]:
    """GBM terminal values. Returns (terminal, mean, std)."""
    if FORTRAN_AVAILABLE:
        terminal, mean, std = _f90.montecarlo.simulate_gbm(
            n_paths, n_steps, s0, mu, sigma, t, seed, antithetic
        )
        return np.asarray(terminal, dtype=float), float(mean), float(std)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    if antithetic:
        half = n_paths // 2
        up = s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z[:half])
        dn = s0 * np.exp((mu - 0.5 * sigma**2) * t - sigma * np.sqrt(t) * z[:half])
        terminal = np.empty(n_paths)
        terminal[:half] = 0.5 * (up + dn)
        terminal[half:2 * half] = terminal[:half]
        if n_paths % 2:
            terminal[-1] = s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z[-1])
    else:
        terminal = s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    return terminal, float(terminal.mean()), float(terminal.std(ddof=1))


def simulate_vasicek_losses_fortran(
    n_scenarios: int, n_obligors: int, pd: float, rho: float,
    lgd: float, ead: float, seed: int, antithetic: bool = False,
) -> tuple[np.ndarray, float]:
    """Vasicek losses. Returns (losses, mean_loss)."""
    if FORTRAN_AVAILABLE:
        losses, mean_loss = _f90.montecarlo.simulate_vasicek_losses(
            n_scenarios, n_obligors, pd, rho, lgd, ead, seed, antithetic
        )
        return np.asarray(losses, dtype=float), float(mean_loss)
    # Identical model to src/models.py (exact binomial for every N).
    from models import simulate_vasicek_losses

    losses = simulate_vasicek_losses(
        n_scenarios, n_obligors, pd=pd, rho=rho, lgd=lgd, ead=ead,
        seed=seed, antithetic=antithetic,
    )
    return losses, float(losses.mean())


def gbm_call_control_variate(
    n_paths: int, s0: float, k: float, r: float, mu: float,
    sigma: float, t: float, seed: int,
) -> tuple[float, float, float]:
    """European call: (raw MC price, control-variate price, Black-Scholes).

    Control variate is the terminal stock price, whose expectation
    E[S_T] = S0 exp(mu T) is known analytically.
    """
    if FORTRAN_AVAILABLE:
        mc, cv, bs = _f90.montecarlo.gbm_call_control_variate(
            n_paths, s0, k, r, mu, sigma, t, seed
        )
        return float(mc), float(cv), float(bs)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    st = s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    payoff = np.exp(-r * t) * np.maximum(st - k, 0.0)
    e_st = s0 * np.exp(mu * t)
    beta = np.cov(payoff, st, ddof=1)[0, 1] / np.var(st, ddof=1)
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    bs = float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(payoff.mean()), float(payoff.mean() - beta * (st.mean() - e_st)), bs


def portfolio_var_cvar_fortran(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    if FORTRAN_AVAILABLE:
        var, cvar = _f90.montecarlo.portfolio_var_cvar(
            np.asarray(losses, dtype=float), alpha
        )
        return float(var), float(cvar)
    from risk_metrics import var_cvar

    return var_cvar(losses, alpha)
