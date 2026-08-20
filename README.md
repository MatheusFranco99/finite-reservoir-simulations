# Simulations for "Finite reservoirs lead to Wentzell boundary conditions"

Code that produces the figures of the section on computational simulations of

> M. Franco, T. Franco, P. Gonçalves,
> *Finite reservoirs lead to Wentzell boundary conditions for independent
> random walks and exclusion processes*.

Two particle systems are simulated on `{0, 1, ..., N}`, both in the diffusive
time scale. Sites `1, ..., N` are the bulk, where each particle jumps to a
neighbour at rate one; site `0` is the finite reservoir, which sends a particle
to site `1` at rate `alpha * eta(0) / N^theta`. In the exclusion case the bulk
holds at most one particle per site and a jump into an occupied bulk site is
suppressed, while the reservoir remains unbounded.

## Running it

```
pip install -r requirements.txt
python figures.py
```

This writes six PNG files into `figures/`:

| file | dynamics | theta |
| --- | --- | --- |
| `irw_theta_2.png` | independent random walks | 2 |
| `sep_theta_2.png` | exclusion | 2 |
| `irw_theta_1.png` | independent random walks | 1 |
| `sep_theta_1.png` | exclusion | 1 |
| `irw_theta_0p5.png` | independent random walks | 0.5 |
| `sep_theta_0p5.png` | exclusion | 0.5 |

Every curve is an average over `k * N = 25600` realizations with `N = 64`,
`alpha = 1` and initial profile `gamma(x) = 1/2 + x(1-x)`, at the macroscopic
times `t = 0.1, 0.2, 0.5, 1`. The whole run takes roughly an hour on a laptop
(about ten minutes per figure), and the six figures are independent of each
other, so it is safe to interrupt and start again.

The reservoir is a single site holding a number of particles of order
`N^theta`, so it cannot be drawn on the same axis as the bulk; the figures show
`alpha * eta(0) / N^theta`, which is the quantity that converges to
`gamma(0)`, as the small bar to the left of the box.

## Reproducibility

`simulation.py` derives one seed per realization from a single base seed
(`SEED` in `figures.py`), by index, so an average does not depend on how the
process pool happens to schedule the work. Running the code twice on the same
machine gives the same figures.

The heavy loop is compiled with numba. Without numba installed the code still
runs — the decorator falls back to a no-op — but it is far too slow to be
useful, and, since the compiled kernel draws its random numbers from numba's
generator rather than NumPy's, the numbers coming out of a run without numba
are *not* the ones the figures were made with.

## Files

- `simulation.py` — the dynamics: the clock construction, one realization, and
  the average over realizations.
- `figures.py` — the six figures of the paper.
