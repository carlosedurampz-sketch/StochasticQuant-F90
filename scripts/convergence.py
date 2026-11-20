"""Numerical convergence: VaR/CVaR vs number of scenarios, GBM scheme error.

Checks two things a physicist cares about before trusting a number:
1. Monte Carlo error shrinks like 1/sqrt(N): VaR99 and CVaR99 for the
   Vasicek portfolio at N_scen = 1k … 200k, with the analytic Vasicek
   limit as reference and the Monte Carlo standard error of the mean.
2. Discretisation error: Euler-Maruyama (strong order 0.5) vs Milstein
   (strong order 1.0) vs the exact GBM solution, as n_steps grows.

Outputs: results/convergence.csv, results/convergence.png
Run: python scripts/convergence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models import simulate_gbm, simulate_vasicek_losses, vasicek_var_analytic  # noqa: E402
from risk_metrics import expected_shortfall, value_at_risk  # noqa: E402


def main() -> None:
    os.makedirs("results", exist_ok=True)
    var_limit = vasicek_var_analytic(0.02, 0.15, 0.45, 1.0, 1000, alpha=0.99)
    print(f"Vasicek asymptotic VaR99 (N->inf): {var_limit:.4f}")
    rows = []
    print(f"{'scenarios':>10} {'VaR99':>8} {'CVaR99':>8} {'EL':>7} {'EL err':>7} {'1/sqrt(N) SE':>12}")
    for n in [1_000, 5_000, 20_000, 50_000, 100_000, 200_000]:
        losses = simulate_vasicek_losses(n, 1000, pd=0.02, rho=0.15, lgd=0.45, ead=1.0,
                                         seed=42, antithetic=True)
        var99 = value_at_risk(losses, 0.99)
        cvar99 = expected_shortfall(losses, 0.99)
        el, se = losses.mean(), losses.std(ddof=1) / np.sqrt(n)
        rows.append({"n_scenarios": n, "var99": var99, "cvar99": cvar99,
                     "el": el, "el_se": se, "var99_vs_limit_pct": 100 * (var99 - var_limit) / var_limit})
        print(f"{n:>10,} {var99:>8.3f} {cvar99:>8.3f} {el:>7.3f} {el - 9.0:>+7.3f} {se:>12.4f}")
    import csv
    with open("results/convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # GBM scheme error vs exact terminal mean E[S_T] = S0 exp(mu T)
    theory = 100.0 * np.exp(0.05)
    print(f"\nGBM terminal mean, theory {theory:.4f} (20k paths, seed 7):")
    print(f"{'steps':>6} {'exact err':>10} {'euler err':>10} {'milstein err':>12}")
    for steps in [12, 52, 252]:
        errs = {}
        for scheme in ["exact", "euler", "milstein"]:
            paths = simulate_gbm(n_steps=steps, n_paths=20_000, seed=7, scheme=scheme)
            errs[scheme] = paths[:, -1].mean() - theory
        print(f"{steps:>6} {errs['exact']:>10.3f} {errs['euler']:>10.3f} {errs['milstein']:>12.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r["n_scenarios"] for r in rows]
    ax.errorbar(ns, [r["var99"] for r in rows], fmt="o-", label="VaR99 (simulated)")
    ax.plot(ns, [r["cvar99"] for r in rows], "s-", label="CVaR99 (simulated)")
    ax.axhline(var_limit, color="red", linestyle="--", label=f"Vasicek limit {var_limit:.2f}")
    ax.set_xscale("log"); ax.set_xlabel("Scenarios"); ax.set_ylabel("Loss")
    ax.set_title("Convergence: VaR/CVaR vs number of scenarios (1/sqrt(N) error)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("results/convergence.png", dpi=150)
    print("Saved results/convergence.csv and results/convergence.png")


if __name__ == "__main__":
    main()
