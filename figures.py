"""
Produces the six figures of the section on computational simulations.

Run it with

    python figures.py

and it writes irw_theta_*.png and sep_theta_*.png into ./figures.  Each curve
is an average over 25600 realizations, so the whole thing takes about an hour
on a laptop; the six figures are independent, so it is fine to stop and rerun.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from simulation import gamma, mean_profiles

N = 64
K = 400                      # k*N = 25600 realizations behind every curve
ALPHA = 1.0
SEED = 20260726
TIMES = (0.1, 0.2, 0.5, 1.0)
COLORS = ("#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e")

FIGURES = (
    ("irw_theta_2.png", 2.0, False),
    ("sep_theta_2.png", 2.0, True),
    ("irw_theta_1.png", 1.0, False),
    ("sep_theta_1.png", 1.0, True),
    ("irw_theta_0p5.png", 0.5, False),
    ("sep_theta_0p5.png", 0.5, True),
)


def reservoir_height(value, theta, exclusion):
    # at the critical value the exclusion boundary compares with q/(1+q)
    if theta == 1 and exclusion:
        return value / (1.0 + value)
    return value


def ratio(profile, initial, theta):
    """Below the critical value the interesting quantity is the mass left in
    the bulk; at and above it, how much the reservoir still holds."""
    if theta < 1:
        bulk = initial[1:].sum()
        return profile[1:].sum() / bulk * 100 if bulk else float("nan")
    return profile[0] / initial[0] * 100 if initial[0] else float("nan")


def label(t, profile, initial, theta):
    return rf"$t={t:g}$ (${ratio(profile, initial, theta):.1f}\%$)"


def plot(path, theta, exclusion, initial, profiles):
    xs = np.arange(N + 1) / N
    vals = np.array([gamma(x) for x in xs])

    reservoir_left, reservoir_right = -0.11, -0.01
    width = reservoir_right - reservoir_left

    everything = [(0.0, initial), *profiles]
    heights = [reservoir_height(p[0], theta, exclusion) for _, p in everything]

    ys = np.concatenate([vals, *[p[1:] for _, p in everything]])
    if theta >= 1:
        ys = np.concatenate([ys, np.array(heights)])
    y_min = max(0.0, float(ys.min()) - 0.03)
    y_max = float(ys.max()) + 0.03

    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.plot(xs, vals, "-", color="black", linewidth=1, label=r"$\gamma(x)$")
    ax.plot(xs[1:], initial[1:], "o-", color="red", markersize=4, linewidth=1,
            label=label(0.0, initial, initial, theta))
    for color, (t, profile) in zip(COLORS, profiles):
        ax.plot(xs[1:], profile[1:], "o-", color=color, markersize=4, linewidth=1,
                label=label(t, profile, initial, theta))

    # the reservoir is a single site, so we draw it as a little bar chart
    # sitting to the left of the box rather than as one more point
    bar = width / len(everything)
    for i, color in enumerate(("red",) + COLORS[: len(profiles)]):
        ax.add_patch(Rectangle((reservoir_left + i * bar, 0.0), bar, heights[i],
                               facecolor=color, edgecolor="white",
                               linewidth=0.4, alpha=0.35))
    ax.add_patch(Rectangle((reservoir_left, 0.0), width, max(heights),
                           facecolor="none", edgecolor="gray", linewidth=0.7))
    ax.text((reservoir_left + reservoir_right) / 2, y_min + 0.02 * (y_max - y_min),
            "reservoir", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlim(reservoir_left - 0.01, 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$x = i/N$  (reservoir shown at left)", fontsize=11)
    ax.set_ylabel(r"$\mathbb{E}[\eta(x)]$", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_title(
        f"{'SEP' if exclusion else 'IRW'} - mean profiles at multiple times "
        rf"($N={N}$, {K * N} runs, $\theta={theta:g}$, $\alpha={ALPHA:g}$)",
        fontsize=12,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10,
              title="time and bulk mass ratio" if theta < 1 else "time and reservoir ratio",
              title_fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    for name, theta, exclusion in FIGURES:
        print(f"{name}: theta={theta:g}, {'SEP' if exclusion else 'IRW'}")
        initial = None
        profiles = []
        for t in TIMES:
            eta_0, eta_t = mean_profiles(N, theta, ALPHA, t, K, exclusion, SEED)
            if initial is None:
                initial = eta_0
            profiles.append((t, eta_t))
        plot(os.path.join(outdir, name), theta, exclusion, initial, profiles)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figures")
