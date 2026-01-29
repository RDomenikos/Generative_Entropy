#!/usr/bin/env python3
"""
Option A: Apparent H-theorem break from entropy-model misspecification.

System: two metastable basins (mode label z in {0,1}) with slow switching,
fast within-basin mixing (Gaussian emission in x).

Goal: show that Shannon entropy on raw microstate bins H_Sh(X) can be non-monotone
(or even decrease) during a relaxation protocol, while a mode-compatible entropy
H_mode(X)=H(z)+E_z[H(X|z)] behaves as expected.

No 'we' language; meant to be dropped into paper pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def shannon_entropy(p, eps=1e-15):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return -np.sum(p * np.log(p))

def hist_pmf(x, bins, xlim):
    counts, edges = np.histogram(x, bins=bins, range=xlim, density=False)
    p = counts.astype(float)
    if p.sum() == 0:
        p = np.ones_like(p) / len(p)
    else:
        p /= p.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return p, centers, edges

def simulate_two_basin(T=300, N=20000, seed=0,
                       mu0=-1.5, mu1=+1.5,
                       # within-basin spread
                       sigma=0.35,
                       # initial mode probability
                       pi0=0.95,
                       # slow switching rate
                       switch=0.01,
                       # protocol: gradually sharpen sigma and equalize pi
                       sigma_final=0.18,
                       pi_final=0.50):
    """
    At each time t:
    - mode z_t evolves with small switching probability
    - x samples are generated from N(mu_z, sigma_t)

    Protocol:
    - sigma_t decreases linearly (peaks sharpen) -> can reduce H_Sh(X)
    - mode occupancy pi_t moves toward 0.5 (more mixed across basins) -> should increase H(mode)

    The combination can induce non-monotonicity in H_Sh(X) even though basin mixing increases.
    """
    rng = np.random.default_rng(seed)

    # time-varying parameters
    sigmas = np.linspace(sigma, sigma_final, T)
    pis = np.linspace(pi0, pi_final, T)

    # mode state for each particle/sample in a population
    z = (rng.random(N) < pis[0]).astype(int)  # 1 with prob pi, 0 otherwise
    # but interpret "pi" as P(z=1)
    # enforce initial according to pi0
    z = (rng.random(N) < pis[0]).astype(int)

    xs = []
    zs = []
    for t in range(T):
        # slow switching: flip mode with probability 'switch'
        flips = rng.random(N) < switch
        z[flips] = 1 - z[flips]

        # softly steer occupancy toward pis[t] by random relabeling (represents slow drift in macro-field)
        # This is optional but helps create a clear "relaxation" toward equal occupancy.
        target_pi = pis[t]
        current_pi = z.mean()
        if abs(current_pi - target_pi) > 1e-3:
            # relabel a small fraction to move toward target
            frac = min(0.02, abs(current_pi - target_pi))
            k = int(frac * N)
            if k > 0:
                if current_pi < target_pi:
                    idx = np.where(z == 0)[0]
                    pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
                    z[pick] = 1
                else:
                    idx = np.where(z == 1)[0]
                    pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
                    z[pick] = 0

        # sample x conditional on z
        sig = sigmas[t]
        x = rng.normal(loc=np.where(z == 0, mu0, mu1), scale=sig)

        xs.append(x.copy())
        zs.append(z.copy())

    return xs, zs, sigmas, pis

# -----------------------------
# Main experiment
# -----------------------------
def run_experiment():
    # simulation
    T = 280
    N = 25000
    bins = 80
    xlim = (-5, 5)

    xs, zs, sigmas, pis = simulate_two_basin(
        T=T, N=N, seed=2,
        mu0=-1.6, mu1=+1.6,
        sigma=0.45, sigma_final=0.16,
        pi0=0.92, pi_final=0.50,
        switch=0.006
    )

    H_sh = np.zeros(T)        # Shannon on raw x histogram
    H_mode = np.zeros(T)      # mode-compatible entropy: H(z)+E[H(X|z)]
    H_z = np.zeros(T)         # entropy of mode label
    H_x_given = np.zeros(T)   # conditional part

    for t in range(T):
        x = xs[t]
        z = zs[t]

        # raw x entropy
        p_x, _, _ = hist_pmf(x, bins=bins, xlim=xlim)
        H_sh[t] = shannon_entropy(p_x)

        # mode entropy
        pz1 = z.mean()
        pz = np.array([1 - pz1, pz1], dtype=float)
        H_z[t] = shannon_entropy(pz)

        # conditional entropies via histograms within each mode
        H_cond = 0.0
        for m in [0, 1]:
            xm = x[z == m]
            if len(xm) < 50:
                continue
            p_xm, _, _ = hist_pmf(xm, bins=bins, xlim=xlim)
            H_cond += pz[m] * shannon_entropy(p_xm)
        H_x_given[t] = H_cond

        # total compatible entropy
        H_mode[t] = H_z[t] + H_x_given[t]

    # pick a few snapshots for distribution plots
    snap_idx = [0, T//3, 2*T//3, T-1]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 4.2))
    plt.plot(
        H_sh,
        label=r"Shannon on microstates $H_{\mathrm{Sh}}(X)$",
        marker="x",
        markevery=12,
        markersize=7,
        linewidth=1.2
    )

    plt.plot(
        H_mode,
        label=r"Mode-compatible $H_{\mathrm{mode}}(X)=H(Z)+\mathbb{E}[H(X|Z)]$",
        linewidth=2.2
    )

    plt.plot(
        H_z,
        "--",
        label=r"Mode entropy $H(Z)$",
        linewidth=1.8
    )
    plt.xlabel("time step")
    plt.ylabel("entropy (nats)")
    plt.title("Apparent H-theorem break under entropy-model misspecification")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Distribution snapshots
    fig, axes = plt.subplots(1, len(snap_idx), figsize=(14, 3.2), sharey=True)
    for ax, t in zip(axes, snap_idx):
        x = xs[t]
        p_x, centers, _ = hist_pmf(x, bins=bins, xlim=xlim)
        ax.bar(centers, p_x, width=(centers[1]-centers[0]), alpha=0.5)
        ax.set_title(f"t={t}\nH_Sh={H_sh[t]:.2f}, H_mode={H_mode[t]:.2f}\nσ={sigmas[t]:.2f}, pi≈{zs[t].mean():.2f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("pmf over x bins")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
