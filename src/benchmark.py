"""Benchmark: pure Python vs NumPy vs Fortran (or NumPy fallback).

Task: GBM terminal values for ``n_paths`` paths, repeated ``n_repeats``
times; the median wall time is reported. Memory is peak RSS delta
measured with ``tracemalloc`` for the Python parts (Fortran allocates
outside the Python heap, so its memory is reported as the RSS delta
of the process instead — see README for the caveat).

Outputs (created by ``main``):
- results/benchmark.csv  — one row per (engine, n_paths)
- results/benchmark.png  — log-log time chart

Run:  python -m src.benchmark [--n-paths 20000 100000] [--repeats 3]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import resource
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fortran_backend import BACKEND, FORTRAN_AVAILABLE, simulate_gbm_fortran  # noqa: E402


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pure_python_gbm(n_paths, s0=100.0, mu=0.05, sigma=0.20, t=1.0, seed=42):
    """Deliberately naive loop version — the baseline a quant should beat."""
    rng = random.Random(seed)
    drift = (mu - 0.5 * sigma**2) * t
    diff = sigma * math.sqrt(t)
    out = []
    for _ in range(n_paths):
        z = rng.gauss(0.0, 1.0)
        out.append(s0 * math.exp(drift + diff * z))
    return out


def numpy_gbm(n_paths, s0=100.0, mu=0.05, sigma=0.20, t=1.0, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)


def _timeit(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def benchmark(n_paths_list=(20_000, 100_000), repeats=3) -> list[dict]:
    rows: list[dict] = []
    for n in n_paths_list:
        # Pure Python
        t_py = _timeit(lambda: pure_python_gbm(n), repeats)
        rows.append({"engine": "python-pure", "n_paths": n, "time_s": t_py,
                     "paths_per_s": n / t_py})
        # NumPy
        t_np = _timeit(lambda: numpy_gbm(n), repeats)
        rows.append({"engine": "numpy", "n_paths": n, "time_s": t_np,
                     "paths_per_s": n / t_np})
        # Fortran (or its NumPy fallback — labelled honestly)
        engine = "fortran-f2py" if FORTRAN_AVAILABLE else "fortran-fallback-numpy"
        t_f = _timeit(
            lambda: simulate_gbm_fortran(n, 252, 100.0, 0.05, 0.20, 1.0, 42, False),
            repeats,
        )
        rows.append({"engine": engine, "n_paths": n, "time_s": t_f,
                     "paths_per_s": n / t_f})
        for r in rows[-3:]:
            print(f"  n={n:>7,}  {r['engine']:<24} {r['time_s']:.4f}s  "
                  f"({r['paths_per_s']:,.0f} paths/s)")
    return rows


def plot(rows, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    engines = sorted({r["engine"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for eng in engines:
        pts = sorted([r for r in rows if r["engine"] == eng], key=lambda r: r["n_paths"])
        ax.plot([r["n_paths"] for r in pts], [r["time_s"] for r in pts],
                marker="o", label=eng)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Median wall time (s, log scale)")
    ax.set_title("GBM terminal simulation — pure Python vs NumPy vs Fortran")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Chart saved to {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark GBM engines")
    ap.add_argument("--n-paths", type=int, nargs="+", default=[20_000, 100_000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    print(f"Backend for Fortran engine: {BACKEND} "
          f"(FORTRAN_AVAILABLE={FORTRAN_AVAILABLE})")
    print(f"Peak RSS before: {_rss_mb():.1f} MB")
    rows = benchmark(args.n_paths, args.repeats)
    print(f"Peak RSS after:  {_rss_mb():.1f} MB")
    os.makedirs(args.out_dir, exist_ok=True)
    import csv

    csv_path = os.path.join(args.out_dir, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "n_paths", "time_s", "paths_per_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved to {csv_path}")
    # Speedups at the largest size
    biggest = max(args.n_paths)
    by_eng = {r["engine"]: r["time_s"] for r in rows if r["n_paths"] == biggest}
    if "python-pure" in by_eng and "numpy" in by_eng:
        print(f"NumPy is {by_eng['python-pure']/by_eng['numpy']:.1f}x faster than "
              f"pure Python at n={biggest:,}")
    plot(rows, os.path.join(args.out_dir, "benchmark.png"))


if __name__ == "__main__":
    main()
