"""StochasticQuant-F90: tail-risk Monte Carlo (Merton / Vasicek / GBM / Heston)."""
from .models import (  # noqa: F401
    merton_pd,
    simulate_gbm,
    simulate_heston,
    simulate_merton_losses,
    simulate_vasicek_losses,
    vasicek_var_analytic,
)
from .risk_metrics import expected_shortfall, kupiec_test, value_at_risk  # noqa: F401

__version__ = "1.0.0"
