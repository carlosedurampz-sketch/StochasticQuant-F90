"""Backtesting VaR with the Kupiec unconditional-coverage test.

Two experiments (both on synthetic data — see README § Data):
1. Well-specified model: daily losses ~ N(0, 1); VaR99 from the true
   normal quantile. Kupiec should NOT reject (p > 0.05).
2. Misspecified model: losses are Student-t with 3 df (heavy tails)
   but VaR is still the normal VaR99. Kupiec SHOULD reject — the test
   has power against exactly this failure, which is why it is used
   in model validation (Basel traffic-light approach).

Outputs: results/backtest.json (printed as well).
Run: python scripts/backtest.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from risk_metrics import kupiec_test, normal_var_cvar  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(123)
    n = 1000
    var99, _ = normal_var_cvar(0.99)
    out = {}
    for name, losses in [
        ("normal_well_specified", rng.standard_normal(n)),
        ("student_t3_misspecified", rng.standard_t(3, size=n) / np.sqrt(3)),
    ]:
        res = kupiec_test(losses, var99, alpha=0.99)
        out[name] = res
        print(f"{name}: violations={res['violations']:.0f}/{n} "
              f"({res['violation_rate']:.1%}, expected 1.0%), "
              f"LR={res['lr_stat']:.2f}, p={res['p_value']:.4f}, "
              f"reject H0 at 5%: {res['reject_h0_5pct']}")
    os.makedirs("results", exist_ok=True)
    with open("results/backtest.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/backtest.json")
    assert not out["normal_well_specified"]["reject_h0_5pct"], "well-specified model rejected?!"
    assert out["student_t3_misspecified"]["reject_h0_5pct"], "test has no power vs heavy tails?!"
    print("Backtest assertions passed: Kupiec accepts the good model, rejects the bad one.")


if __name__ == "__main__":
    main()
