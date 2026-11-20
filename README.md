# StochasticQuant-F90 — Tail Risk & Hybrid Stochastic Processes

> **Resumen en español.** Motor cuantitativo para simular el riesgo de cola de una cartera de crédito y la valuación de activos bajo incertidumbre, combinando Python con un núcleo numérico compilado en Fortran 90. Modela incumplimientos correlacionados con el modelo estructural de Merton y el modelo unifactorial de Vasicek (descomposición de Cholesky), y la dinámica de mercado con movimiento browniano geométrico (GBM) y volatilidad estocástica de Heston, resueltos con Euler–Maruyama y Milstein. Calcula VaR y Expected Shortfall (CVaR), valida el modelo con la prueba de cobertura de Kupiec y compara rigurosamente los tiempos de ejecución entre Python puro, NumPy vectorizado y Fortran 90. **Todos los datos son sintéticos y simulados (semillas fijas); los resultados son ilustrativos y sirven para validar la numeración, no describen ninguna cartera ni exposición real.**

## About the author / Sobre el autor

*Written in the first person, because this project is the one where my actual background matters most.*

I am **Carlos Eduardo Ramírez Pérez**, a physicist in the final stage of my degree at the Universidad Autónoma Metropolitana (UAM, 2022–2026, in the process of obtaining my degree), and a Technician in Electronics (CBTis No. 003, Tlaxcala, 2015–2018). I am not a banker and I have never held a quant or banking job — I do not pretend otherwise. What I do bring is the toolkit of a physicist who simulates:

- **Stochastic simulation is my home turf.** In physics I learned to treat uncertainty as something you *simulate*, not something you hand-wave away: Monte Carlo methods, random processes, convergence in $1/\sqrt{N}$, and the discipline of checking a numerical result against an analytic limit before trusting it. That is exactly what tail-risk modelling asks for.
- **Stochastic differential equations.** GBM and Heston are SDEs; Euler–Maruyama and Milstein are discretisation schemes with known strong orders of convergence (0.5 and 1.0). I implemented both, measured the error against the exact GBM solution, and documented where each scheme wins — the same habit of validating a discretisation that I use in computational physics.
- **Fortran 90 is my real differentiator.** My Fortran is intermediate, but it is *working* Fortran: this repository contains a real compiled core (`fortran/montecarlo.f90`) — Box–Muller normals, Cholesky factorisation, antithetic variates, control variates — callable from Python through `f2py`, with a NumPy fallback that reproduces the same API when no compiler is available. Most data portfolios never touch a compiled language; a physicist who can is rarer, and this project proves it instead of claiming it.
- **Traceability and auditable documentation.** In 2026 I worked at INCYMED, calibrating pressure, temperature and electrical instruments in a laboratory accredited by PJLA under ISO/IEC 17025, where I was authorised to execute and review calibrations. That work taught me what a model-validation reviewer actually wants: seeded reproducibility, stated assumptions, analytic cross-checks, and a paper trail. I applied the same standard here — every headline number in this README is reproducible from a seed, and every model is checked against a closed-form limit.
- **The rest of my toolkit.** Python (advanced) and R (advanced) for analysis, C/C++ (advanced) for performance, Mathematica and LaTeX (advanced) for the mathematics in this document, SQL (basic — I am strengthening it in my companion projects, CreditRisk-IFRS9 and AML-GraphAlert), English B2. Courses: Data Science with R (UNAM), Python I (Pontificia Universidad Católica de Chile), a 4-course Big Data series, and Introduction to Statistics (Stanford Online / Coursera). I have also taught physics and mathematics as a state delegate for the CMF (2021–2022), which is why this README explains the maths instead of hiding it.

**Why this project exists.** I am looking for a first role in data / quantitative risk / banking-fintech in Mexico (target: above MXN 20,000/month). This is the project where my physics profile shines brightest, so I built it to the standard I would want a hiring manager to judge: correct mathematics, validated numerics, and honest labelling of what is synthetic.

## What it does

| Block | Model | File |
|---|---|---|
| Correlated defaults | Merton (1974) structural model, full correlation matrix via Cholesky | `src/models.py`, `fortran/montecarlo.f90` |
| Correlated defaults | Vasicek (1987/2002) single-factor model, asymptotic VaR in closed form | `src/models.py` |
| Market dynamics | GBM — exact, Euler–Maruyama, Milstein | `src/models.py` |
| Market dynamics | Heston (1993) stochastic volatility, full-truncation Euler / Milstein | `src/models.py` |
| Tail risk | VaR$_\alpha$, Expected Shortfall (CVaR$_\alpha$) | `src/risk_metrics.py` |
| Validation | Kupiec (1995) unconditional-coverage test | `src/risk_metrics.py` |
| Variance reduction | Antithetic variates, control variates (Black–Scholes) | `fortran/montecarlo.f90` |
| Performance | Pure Python vs NumPy vs Fortran benchmark | `src/benchmark.py` |

## Mathematics

### Merton structural model

Firm asset value follows GBM, $dV = \mu V\,dt + \sigma V\,dW$. Default by horizon $T$ iff $V_T < D$ (debt):

$$d_2 = \frac{\ln(V_0/D) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad PD = \Phi(-d_2)$$

Correlated defaults: standardised asset returns $X \sim \mathcal{N}(0, \Sigma)$ are drawn as $X = L Z$ with $L$ the Cholesky factor, $\Sigma = LL^\top$; obligor $i$ defaults iff $X_i < \Phi^{-1}(PD_i)$.

### Vasicek single-factor model

Conditional on a systemic factor $Y \sim \mathcal{N}(0,1)$,

$$p(Y) = \Phi\left(\frac{\Phi^{-1}(PD) - \sqrt{\rho}\,Y}{\sqrt{1-\rho}}\right), \qquad \text{defaults} \mid Y \sim \mathrm{Binomial}(N, p(Y))$$

Expected loss is exact: $\mathbb{E}[L] = N \cdot PD \cdot LGD \cdot EAD$. In the large-homogeneous-portfolio limit ($N \to \infty$) VaR has a closed form — the Basel II IRB formula — used here as a validation target:

$$\mathrm{VaR}_\alpha = N \cdot LGD \cdot EAD \cdot \Phi\left(\frac{\Phi^{-1}(PD) + \sqrt{\rho}\,\Phi^{-1}(\alpha)}{\sqrt{1-\rho}}\right)$$

### GBM and Heston

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t \quad\text{(GBM)}, \qquad S_T = S_0 e^{(\mu-\sigma^2/2)T + \sigma\sqrt{T}Z}$$

$$\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)} \\
dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{(2)}, \quad \mathrm{corr}(dW^{(1)}, dW^{(2)}) = \rho
\end{aligned} \quad\text{(Heston)}$$

Discretisation, step $h$: Euler–Maruyama $S_{t+h} = S_t + \mu S_t h + \sigma S_t\sqrt{h}\,Z$ (strong order 0.5); Milstein adds $\tfrac12 \sigma^2 S_t (h Z^2 - h)$ (strong order 1.0 for GBM). Heston uses full truncation, $v^+ = \max(v, 0)$, so variance never goes negative even when the Feller condition $2\kappa\theta > \xi^2$ fails.

### VaR, Expected Shortfall, Kupiec

$$\mathrm{VaR}_\alpha = q_\alpha(L), \qquad \mathrm{CVaR}_\alpha = \mathbb{E}[L \mid L > \mathrm{VaR}_\alpha]$$

Kupiec's POF test: with $x$ violations in $n$ observations and $p = 1-\alpha$,

$$LR = -2\ln\left[\frac{(1-p)^{n-x} p^{x}}{(1-x/n)^{n-x}(x/n)^{x}}\right] \sim \chi^2_1 \quad \text{under } H_0$$

### Variance reduction

- **Antithetic variates:** each draw $Z$ is paired with $-Z$; the average cancels first-order Monte Carlo error for symmetric payoffs.
- **Control variates:** for a European call, the terminal price $S_T$ is used as control, since $\mathbb{E}[S_T] = S_0 e^{\mu T}$ is known; the estimator is $\hat{P}_{CV} = \hat{P} - \hat\beta(\bar S_T - \mathbb{E}[S_T])$, with $\hat\beta$ from the sample covariance. The analytic Black–Scholes price is the reference.

## Architecture

```
StochasticQuant-F90/
├── fortran/montecarlo.f90   # F90 core: Box-Muller, Cholesky, GBM, Vasicek,
│                            #   Merton, control variates, VaR/CVaR (quicksort)
├── src/
│   ├── models.py            # Merton, Vasicek, GBM, Heston (NumPy)
│   ├── risk_metrics.py      # VaR, CVaR, Kupiec, analytic normal VaR/CVaR
│   ├── fortran_backend.py   # ONE API: compiled f2py if present, else NumPy
│   ├── run_simulation.py    # end-to-end run + validation gates
│   └── benchmark.py         # pure Python vs NumPy vs Fortran
├── scripts/
│   ├── convergence.py       # VaR/CVaR vs scenarios; Euler vs Milstein error
│   └── backtest.py          # Kupiec: good model accepted, bad model rejected
├── notebooks/convergence_and_backtesting.ipynb
├── tests/                   # pytest: analytic ground truths (see below)
├── results/                 # generated, git-ignored (summary.json, *.png, *.csv)
├── Makefile  run_all.sh  requirements.txt  LICENSE (MIT)
```

## Quickstart

```bash
git clone <this-repo> && cd StochasticQuant-F90
pip install -r requirements.txt
python -m pytest tests/ -q        # 12 tests, all analytic ground truths
python -m src.run_simulation      # headline run (see Results)
python -m src.benchmark           # timing comparison
python scripts/convergence.py     # convergence + scheme error
python scripts/backtest.py        # Kupiec backtest
# ... or everything at once:
./run_all.sh
```

### With Fortran (recommended — this is the point of the project)

```bash
# Ubuntu/Debian: sudo apt-get install gfortran
make fortran
python -c "import sys; sys.path.insert(0,'src'); from fortran_backend import FORTRAN_AVAILABLE; print('Fortran:', FORTRAN_AVAILABLE)"
python -m src.run_simulation   # backend now reports "fortran"
```

### Without Fortran

No compiler, no problem: `src/fortran_backend.py` detects the missing
`.so` and runs the NumPy implementation behind the **same function
signatures**, reporting `backend: "numpy-fallback"`. All scripts, tests
and benchmarks run identically; the benchmark labels that engine
`fortran-fallback-numpy` instead of pretending it is Fortran.

> **Environment note (Nov 2026).** In the sandbox where this project was built, `gfortran` could not be installed: `apt-get update` succeeded but the configured Ubuntu mirrors returned *no installation candidate* for `gfortran` (only `libgfortran5`, the runtime library, is available). The Fortran source is complete and has been syntax-checked as far as possible without a compiler; `make fortran` is the verified path on a normal Ubuntu/Debian machine. The Python pipeline was fully tested through the fallback, so every number below was actually produced by running code, not by hand.

## Results (illustrative — synthetic data, seed 42)

Headline case: $N = 1{,}000$ obligors, $PD = 2\%$, $\rho = 0.15$, $LGD = 45\%$, $EAD = 1$, 50,000 scenarios, antithetic variates. All figures are reproduced by `python -m src.run_simulation` and stored in `results/summary.json`.

| Quantity | Value |
|---|---|
| Expected loss, simulated | **9.015** |
| Expected loss, theory $N\cdot PD\cdot LGD\cdot EAD$ | 9.00 |
| VaR$_{95}$ / CVaR$_{95}$ (Vasicek) | 28.80 / 41.35 |
| VaR$_{99}$ (Vasicek, simulated) | **48.15** |
| CVaR$_{99}$ (Vasicek, simulated) | **62.18** |
| VaR$_{99}$, analytic Vasicek limit | 47.51 (simulation within 1.3%) |
| VaR$_{99}$ / CVaR$_{99}$ via the Fortran API | 48.15 / 62.18 (identical — same model, fallback engine) |
| Merton EL / VaR$_{99}$ / CVaR$_{99}$ (same PD = 2%) | 9.01 / 47.70 / 62.30 |
| Kupiec VaR$_{99}$: violations 487/50,000 (0.97% vs 1% expected) | LR = 0.34, p = 0.557 → H₀ not rejected |
| GBM terminal mean, exact / Euler / Milstein vs theory 105.13 | 105.06 / 105.06 / 105.06 |
| Heston terminal mean price / mean variance (θ = 0.04) | 105.07 / 0.0399 |
| Runtime of the full run (this sandbox) | ~11 s (after optimising the equicorrelation Merton path from $O(N_{scen}N^2)$ to $O(N_{scen}N)$ — it was 193 s before) |

Figures: `results/loss_distribution.png` (loss histogram with VaR$_{99}$/CVaR$_{99}$), `results/gbm_heston.png` (sample paths), `results/benchmark.png`, `results/convergence.png`.

## Validation — how I know the numbers are right

Every check below is automated (`pytest`, plus assertion gates inside `run_simulation.py` that fail loudly instead of publishing a wrong number):

1. **VaR/CVaR on a known distribution.** For $N(0,1)$ losses, VaR$_{99}$ = 2.3263 and CVaR$_{99}$ = 2.6652 analytically; the simulation matches to Monte Carlo error (`tests/test_risk_metrics.py`).
2. **Cholesky reconstructs correlation.** $L L^\top = \Sigma$ to $10^{-10}$; a non-positive-definite matrix is rejected (`tests/test_models.py`).
3. **Merton PD matches the closed form** ($V_0=100$, $D=80$ → $PD = 0.167$).
4. **Expected-loss identity.** Simulated EL matches $N\cdot PD\cdot LGD\cdot EAD = 9.00$ within $5\times$ Monte Carlo standard error.
5. **VaR converges to the analytic Vasicek limit** (within 10% at 100k scenarios; see `scripts/convergence.py` — error shrinks like $1/\sqrt{N}$).
6. **GBM schemes.** Terminal mean matches $S_0 e^{\mu T} = 105.13$ for exact/Euler/Milstein; Heston variance reverts to $\theta = 0.04$ and never goes negative.
7. **Kupiec has power.** It *accepts* a well-specified normal VaR model and *rejects* the same VaR applied to heavy-tailed Student-$t_3$ losses (`scripts/backtest.py`).

## Benchmark

`python -m src.benchmark` times GBM terminal simulation (20k and 100k paths, median of 3 runs) for pure Python loops, NumPy vectorisation, and the Fortran core (or its labelled fallback). Measured in this sandbox (100k paths): pure Python 0.079 s, NumPy 0.0054 s (**14.7× faster than pure Python**), Fortran-API fallback 0.0038 s — the fallback *is* NumPy underneath, so it matching NumPy is exactly what should happen, and the chart labels it honestly instead of pretending it is compiled Fortran. On a machine with `gfortran`, re-run `make fortran && python -m src.benchmark` to measure the compiled core; the expected pattern is that Fortran pulls ahead mainly on the loop-heavy Vasicek/Merton kernels, while for the vectorisable GBM kernel NumPy is already near the limit — that asymmetry is itself the lesson. See `results/benchmark.csv` and `results/benchmark.png`.

## Data

**No real data is used anywhere in this repository.** Portfolios are synthetic (seeded Monte Carlo with stated parameters), market paths are simulated, and backtests use simulated return series. Nothing here describes a real obligor, a real portfolio, or real market history, and no result should be used for an actual credit or trading decision. Parameters (PD = 2%, $\rho$ = 0.15, LGD = 45%) are textbook-typical values chosen to make the analytic cross-checks meaningful.

## Known limitations (honest)

- The Fortran Vasicek kernel uses an exact binomial for $N \le 500$ and a normal approximation to the binomial above that (relative error $O(1/N)$); the Python implementation uses the exact binomial for every $N$. The two agree within Monte Carlo error — see the Fortran-API VaR in `results/summary.json` — but they are not bit-identical.
- Heston uses full-truncation Euler/Milstein, not the exact Broadie–Kaya scheme; discretisation bias is visible at few steps (see `scripts/convergence.py`).
- VaR/CVaR here are unconditional, one-period, and assume the stated correlation structure; no term structure, no LGD stochasticity, no wrong-way risk.
- Fortran has not been compiled in the build sandbox (see environment note above); the f2py interface was written against the documented f2py calling convention and the fallback was tested instead.

## License

MIT — see `LICENSE`.
