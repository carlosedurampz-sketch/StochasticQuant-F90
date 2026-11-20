# StochasticQuant-F90 — Makefile
.PHONY: all fortran test run benchmark convergence backtest clean help

PYTHON ?= python3

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

all: test run benchmark  ## Full pipeline: tests + simulation + benchmark

fortran:  ## Compile the Fortran 90 core with f2py (needs gfortran)
	cd fortran && f2py -c -m montecarlo montecarlo.f90 && cp montecarlo*.so ../src/ 2>/dev/null || cp montecarlo*.so ../src/
	@echo "Compiled. Verify with: $(PYTHON) -c \"import sys; sys.path.insert(0,'src'); from fortran_backend import FORTRAN_AVAILABLE; print(FORTRAN_AVAILABLE)\""

test:  ## Run pytest
	$(PYTHON) -m pytest tests/ -v

run:  ## End-to-end simulation (50k scenarios)
	$(PYTHON) -m src.run_simulation

benchmark:  ## Pure Python vs NumPy vs Fortran timing
	$(PYTHON) -m src.benchmark

convergence:  ## VaR/CVaR convergence + GBM scheme error
	$(PYTHON) scripts/convergence.py

backtest:  ## Kupiec backtest (well-specified vs heavy-tailed)
	$(PYTHON) scripts/backtest.py

clean:  ## Remove build artefacts and results (results/ is git-ignored)
	rm -rf src/__pycache__ src/montecarlo*.so fortran/*.so fortran/*.o fortran/*.mod
	rm -rf .pytest_cache results/*
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
