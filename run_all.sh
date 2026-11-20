#!/usr/bin/env bash
# StochasticQuant-F90 — run the whole pipeline end to end.
# Usage: ./run_all.sh   (from the repo root; needs Python 3.10+)
# If gfortran is present, the Fortran core is compiled first;
# otherwise the NumPy fallback runs and the script says so.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 install =="
python3 -m pip install -q -r requirements.txt || echo "(pip install failed — continuing with installed packages)"

echo "== 2/5 Fortran core (optional) =="
if command -v gfortran >/dev/null 2>&1; then
  make fortran
else
  echo "gfortran not found — using the NumPy fallback (same API, see README § Fortran)."
fi

echo "== 3/5 tests =="
python3 -m pytest tests/ -q

echo "== 4/5 simulation =="
python3 -m src.run_simulation

echo "== 5/5 benchmark + convergence + backtest =="
python3 -m src.benchmark
python3 scripts/convergence.py
python3 scripts/backtest.py

echo "Done. Results in results/ (summary.json, benchmark.csv/png, loss_distribution.png)."
