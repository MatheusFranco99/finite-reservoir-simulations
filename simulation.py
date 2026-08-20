"""
Microscopic simulation of the two particle systems studied in

    M. Franco, T. Franco, P. Goncalves,
    Finite reservoirs lead to Wentzell boundary conditions for independent
    random walks and exclusion processes.

The state is an array eta of size N+1.  Sites 1,...,N are the bulk and site 0
is the finite reservoir: a particle leaves it towards site 1 at rate
alpha*eta(0)/N^theta, while every bulk particle jumps to each neighbour at
rate 1.  Time is always the diffusive one, so a run of macroscopic time t
means microscopic time t*N^2.

For the exclusion dynamics the bulk holds at most one particle per site and a
jump into an occupied bulk site is simply suppressed; the reservoir stays
unbounded.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

try:
    from numba import njit
except ImportError:  # it still runs, just far too slowly to be useful
    def njit(*args, **kwargs):
        return lambda f: f


def gamma(x):
    """The initial profile used for every figure in the paper."""
    return 0.5 + x * (1 - x)


@njit(cache=True)
def resample(clocks, s, n, theta, alpha, eta):
    # after a jump, every clock whose rate involves eta[s] must be redrawn
    if s <= n - 1:
        rate = alpha * eta[s] / n**theta if s == 0 else float(eta[s])
        clocks[s] = np.random.exponential(1.0 / rate) if rate > 0 else np.inf
    if s >= 1:
        rate = float(eta[s])
        clocks[n + s - 1] = np.random.exponential(1.0 / rate) if rate > 0 else np.inf


@njit(cache=True)
def run_once(seed, n, theta, alpha, gamma_values, target, exclusion):
    """One realization up to microscopic time `target`.

    clocks[x] is the clock of the jump x -> x+1 for x = 0,...,n-1, and
    clocks[n+x] the one of x+1 -> x.  Returns the initial and final states.
    """
    np.random.seed(seed)

    eta = np.zeros(n + 1, dtype=np.int64)
    eta[0] = np.random.poisson((n**theta) * gamma_values[0] / alpha)
    for x in range(1, n + 1):
        if exclusion:
            eta[x] = np.random.binomial(1, gamma_values[x])
        else:
            eta[x] = np.random.poisson(gamma_values[x])
    eta_initial = eta.copy()

    clocks = np.full(2 * n, np.inf)
    rate = alpha * eta[0] / n**theta
    if rate > 0:
        clocks[0] = np.random.exponential(1.0 / rate)
    for x in range(1, n):
        rate = float(eta[x])
        if rate > 0:
            clocks[x] = np.random.exponential(1.0 / rate)
    for x in range(1, n + 1):
        rate = float(eta[x])
        if rate > 0:
            clocks[n + x - 1] = np.random.exponential(1.0 / rate)

    elapsed = 0.0
    while elapsed < target:
        idx = int(np.argmin(clocks))
        dt = clocks[idx]
        if not np.isfinite(dt) or elapsed + dt > target:
            break
        clocks -= dt

        if idx < n:
            src, tgt = idx, idx + 1
        else:
            src, tgt = idx - n + 1, idx - n

        if exclusion and tgt >= 1 and eta[tgt] >= 1:
            resample(clocks, src, n, theta, alpha, eta)  # blocked jump
        else:
            eta[src] -= 1
            eta[tgt] += 1
            resample(clocks, src, n, theta, alpha, eta)
            resample(clocks, tgt, n, theta, alpha, eta)

        elapsed += dt

    return eta_initial, eta


def _one_run(seed, n, theta, alpha, gamma_values, target, exclusion):
    # picklable wrapper, so the compiled kernel can go through a process pool
    return run_once(int(seed), n, theta, alpha, gamma_values, target, exclusion)


def rescale_reservoir(mean, n, theta, alpha):
    """Put site 0 on the scale of the bulk.

    eta(0) is of order N^theta, so what we plot (and what converges to
    gamma(0)) is alpha*eta(0)/N^theta.
    """
    out = mean.astype(float)
    out[0] = alpha * mean[0] / n**theta
    return out


def mean_profiles(n, theta, alpha, t, k=400, exclusion=False, seed=None):
    """Average eta over k*n independent realizations at macroscopic time t.

    Returns (mean at time 0, mean at time t), both already rescaled at
    site 0.  The seeds are derived from `seed` by index, so the averages are
    reproducible and do not depend on how the pool schedules the work.
    """
    gamma_values = np.array([gamma(x / n) for x in range(n + 1)])
    target = t * n**2
    runs = k * n

    seeds = np.random.SeedSequence(seed).generate_state(runs)
    total_initial = np.zeros(n + 1)
    total_final = np.zeros(n + 1)

    started = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        chunksize = max(1, runs // ((os.cpu_count() or 1) * 8))
        for eta_initial, eta_final in pool.map(
            _one_run,
            seeds,
            repeat(n),
            repeat(theta),
            repeat(alpha),
            repeat(gamma_values),
            repeat(target),
            repeat(exclusion),
            chunksize=chunksize,
        ):
            total_initial += eta_initial
            total_final += eta_final

    print(f"    t={t:g}: {runs} runs in {time.perf_counter() - started:.0f}s")
    return (
        rescale_reservoir(total_initial / runs, n, theta, alpha),
        rescale_reservoir(total_final / runs, n, theta, alpha),
    )
