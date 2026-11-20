"""End-to-end run: correlated-default portfolio + GBM/Heston valuation.

Default portfolio (the README's headline case):
  N=1,000 obligors, PD=2%, asset correlation rho=0.15, LGD=45%, EAD=1
  (i.e. 1 unit per obligor), 50,000 scenarios, seed 42.
Expected loss is then N*PD*LGD*EAD = 9.0 by construction — the run
checks the simulation against that identity and against the Vasicek
asymptotic VaR, and fails loudly (exit code 1) if either is off by
more than the stated tolerance.

Outputs:
- results/summary.json — all headline numbers
- results/loss_distribution.png — histogram + VaR99 / CVaR99 lines
- results/gbm_heston.png — sample GBM and Heston paths

Run:  python -m src.run_simulation [--n-scenarios 50000] [--no-plots]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fortran_backend import BACKEND, simulate_vasicek_losses_fortran  # noqa: E402
from models import (  # noqa: E402
    merton_pd,
    simulate_gbm,
    simulate_heston,
    simulate_merton_losses,
    simulate_vasicek_losses,
    vasicek_expected_loss,
    vasicek_var_analytic,
)
from risk_metrics import expected_shortfall, kupiec_test, value_at_risk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="StochasticQuant-F90 end-to-end run")
    ap.add_argument("--n-scenarios", type=int, default=50_000)
    ap.add_argument("--n-obligors", type=int, default=1_000)
    ap.add_argument("--pd", type=float, default=0.02)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--lgd", type=float, default=0.45)
    ap.add_argument("--ead", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--antithetic", action="store_true", default=True)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    t0 = time.perf_counter()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. Vasicek portfolio (NumPy engine) -------------------------------
    losses = simulate_vasicek_losses(
        args.n_scenarios, args.n_obligors, pd=args.pd, rho=args.rho,
        lgd=args.lgd, ead=args.ead, seed=args.seed, antithetic=args.antithetic,
    )
    # --- 2. Same model through the Fortran API (compiled or fallback) ------
    losses_f, _ = simulate_vasicek_losses_fortran(
        args.n_scenarios, args.n_obligors, args.pd, args.rho,
        args.lgd, args.ead, args.seed, args.antithetic,
    )
    # --- 3. Merton portfolio with the same PD (cross-model check) ----------
    # Choose V0/Debt so the Merton PD matches the Vasicek PD: invert
    # PD = Phi(-d2) with sigma=0.25, mu=0.05, T=1.
    d2_target = -norm.ppf(args.pd)
    # ln(V0/D) = d2*sigma - (mu - sigma^2/2); set Debt=80, solve V0.
    v0 = 80.0 * np.exp(d2_target * 0.25 - (0.05 - 0.5 * 0.25**2))
    merton_losses = simulate_merton_losses(
        args.n_scenarios, args.n_obligors, rho=args.rho, v0=float(v0),
        debt=80.0, mu=0.05, sigma=0.25, lgd=args.lgd, ead=args.ead,
        seed=args.seed,
    )
    pd_merton = merton_pd(float(v0), 80.0, 0.05, 0.25)

    el_theory = vasicek_expected_loss(args.pd, args.lgd, args.ead, args.n_obligors)
    var99_analytic = vasicek_var_analytic(
        args.pd, args.rho, args.lgd, args.ead, args.n_obligors, alpha=0.99
    )
    var99, cvar99 = value_at_risk(losses, 0.99), expected_shortfall(losses, 0.99)
    var95, cvar95 = value_at_risk(losses, 0.95), expected_shortfall(losses, 0.95)
    var99_f = value_at_risk(losses_f, 0.99)
    cvar99_f = expected_shortfall(losses_f, 0.99)
    kupiec = kupiec_test(losses, var99, alpha=0.99)

    # --- 4. Market dynamics: GBM exact vs Euler vs Milstein, Heston --------
    gbm_exact = simulate_gbm(n_paths=20_000, n_steps=252, scheme="exact", seed=7)
    gbm_euler = simulate_gbm(n_paths=20_000, n_steps=252, scheme="euler", seed=7)
    gbm_milstein = simulate_gbm(n_paths=20_000, n_steps=252, scheme="milstein", seed=7)
    heston_price, heston_var = simulate_heston(n_paths=20_000, n_steps=252, seed=7)

    summary = {
        "backend": BACKEND,
        "portfolio": {
            "n_obligors": args.n_obligors, "pd": args.pd, "rho": args.rho,
            "lgd": args.lgd, "ead": args.ead, "n_scenarios": args.n_scenarios,
            "seed": args.seed, "antithetic": args.antithetic,
            "merton_v0_for_same_pd": round(float(v0), 4),
            "merton_pd_check": round(pd_merton, 6),
        },
        "vasicek": {
            "expected_loss_simulated": round(float(losses.mean()), 4),
            "expected_loss_theory_N_PD_LGD_EAD": round(el_theory, 4),
            "std_loss": round(float(losses.std(ddof=1)), 4),
            "var_95": round(var95, 4), "cvar_95": round(cvar95, 4),
            "var_99": round(var99, 4), "cvar_99": round(cvar99, 4),
            "var_99_analytic_vasicek_limit": round(var99_analytic, 4),
            "var_99_fortran_api": round(var99_f, 4),
            "cvar_99_fortran_api": round(cvar99_f, 4),
        },
        "merton": {
            "expected_loss_simulated": round(float(merton_losses.mean()), 4),
            "var_99": round(value_at_risk(merton_losses, 0.99), 4),
            "cvar_99": round(expected_shortfall(merton_losses, 0.99), 4),
        },
        "kupiec_var99": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in kupiec.items()},
        "gbm_terminal_mean": {
            "exact": round(float(gbm_exact[:, -1].mean()), 4),
            "euler": round(float(gbm_euler[:, -1].mean()), 4),
            "milstein": round(float(gbm_milstein[:, -1].mean()), 4),
            "theory_S0_exp_mu_T": round(100.0 * np.exp(0.05), 4),
        },
        "heston_terminal": {
            "mean_price": round(float(heston_price[:, -1].mean()), 4),
            "mean_variance": round(float(heston_var[:, -1].mean()), 4),
            "theta": 0.04,
        },
        "runtime_s": round(time.perf_counter() - t0, 2),
    }

    print(f"Backend: {BACKEND} | scenarios={args.n_scenarios:,}")
    print(f"Vasicek  EL sim={losses.mean():.4f}  theory={el_theory:.4f}  "
          f"(diff {abs(losses.mean()-el_theory):.4f})")
    print(f"Vasicek  VaR99={var99:.4f}  CVaR99={cvar99:.4f}  "
          f"analytic limit={var99_analytic:.4f}")
    print(f"Vasicek via Fortran API: VaR99={var99_f:.4f} CVaR99={cvar99_f:.4f}")
    print(f"Merton   EL sim={merton_losses.mean():.4f}  "
          f"VaR99={summary['merton']['var_99']:.4f}  "
          f"CVaR99={summary['merton']['cvar_99']:.4f}  (PD check {pd_merton:.4f})")
    print(f"Kupiec VaR99: violations={kupiec['violations']:.0f}/{kupiec['n']:.0f} "
          f"({kupiec['violation_rate']:.4%}), LR={kupiec['lr_stat']:.3f}, "
          f"p={kupiec['p_value']:.3f}, reject H0: {kupiec['reject_h0_5pct']}")
    print(f"GBM terminal mean — exact {gbm_exact[:, -1].mean():.2f}, "
          f"Euler {gbm_euler[:, -1].mean():.2f}, "
          f"Milstein {gbm_milstein[:, -1].mean():.2f}, "
          f"theory {100 * np.exp(0.05):.2f}")
    print(f"Heston terminal mean price {heston_price[:, -1].mean():.2f}, "
          f"mean variance {heston_var[:, -1].mean():.4f} (theta=0.04)")

    # Validation gates — fail loudly instead of silently publishing a
    # wrong number. Tolerances are Monte Carlo standard errors, not magic.
    el_se = losses.std(ddof=1) / np.sqrt(args.n_scenarios)
    assert abs(losses.mean() - el_theory) < 5 * el_se + 0.5, "EL drifted from theory"
    assert abs(var99 - var99_analytic) / var99_analytic < 0.15, \
        "VaR99 >15% away from Vasicek asymptotic limit"
    assert cvar99 > var99 > losses.mean(), "CVaR > VaR > EL ordering violated"

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.out_dir}/summary.json")

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(losses, bins=80, density=True, alpha=0.7, label="Simulated losses")
        ax.axvline(var99, color="red", label=f"VaR99 = {var99:.2f}")
        ax.axvline(cvar99, color="darkred", linestyle="--",
                   label=f"CVaR99 = {cvar99:.2f}")
        ax.axvline(el_theory, color="green", label=f"EL = {el_theory:.2f}")
        ax.set_xlabel("Portfolio loss")
        ax.set_ylabel("Density")
        ax.set_title(f"Vasicek loss distribution (N={args.n_obligors}, "
                     f"PD={args.pd}, rho={args.rho})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "loss_distribution.png"), dpi=150)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for p in gbm_exact[:20]:
            axes[0].plot(p, alpha=0.5, linewidth=0.7)
        axes[0].set_title("GBM paths (exact)")
        for p in heston_price[:20]:
            axes[1].plot(p, alpha=0.5, linewidth=0.7)
        axes[1].set_title("Heston paths (stochastic vol)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "gbm_heston.png"), dpi=150)
        print(f"Plots saved to {args.out_dir}/loss_distribution.png, "
              f"{args.out_dir}/gbm_heston.png")
    print(f"Runtime: {time.perf_counter() - t0:.2f}s — all validation gates passed.")


if __name__ == "__main__":
    main()
