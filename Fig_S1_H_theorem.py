#!/usr/bin/env python3
"""
H_theorem_mixture.py

Numerical illustration of a generative H-theorem on the energy axis.

We reuse the mixture-of-Gaussians “data system” and the generator-based
energy construction, then:

  - build an empirical energy histogram p_emp(E) from samples,
  - define a concave quadratic generative entropy

        H_micro(p) = sum_k alpha_k sum_{i in band_k} (p_i - p_i^2),

    which depends on the microstructure within each energy band,
  - compute the MaxEnt state p_star that maximizes H_micro under:
        * normalization: sum_i p_i = 1
        * band masses:  sum_{i in band_k} p_i = m_k(emp)
  - start from the empirical distribution p_0 = p_emp
  - evolve via a simple relaxation

        p_{t+1} = (1 - eta) p_t + eta p_star,

    which preserves the constraints and, by concavity of H_micro,
    yields a monotone increase of H_micro(p_t) towards H_micro(p_star).

This script is independent of the band-entropy placeholder used in the
QP examples; it just uses another member of the same concave quadratic
family to visualize the H-theorem numerically.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

np.random.seed(123)

# ----------------------------------------------------------------------
# 1. Mixture-of-Gaussians data system (same as before, but minimal)
# ----------------------------------------------------------------------

def sample_mixture(mu, N=200_000,
                   weights=None,
                   sigmas=None):
    """
    Draw N samples from a 3-component Gaussian mixture:

        w1 N(-mu, sigma1^2) + w2 N(0, sigma2^2) + w3 N(mu, sigma3^2)

    Returns: samples (shape [N,])
    """
    if weights is None:
        weights = np.array([0.3, 0.4, 0.3])
    if sigmas is None:
        sigmas = np.array([0.4, 0.8, 0.4])

    weights = np.asarray(weights)
    sigmas = np.asarray(sigmas)
    means = np.array([-mu, 0.0, mu])

    comps = np.random.choice(3, size=N, p=weights / weights.sum())
    eps = np.random.randn(N) * sigmas[comps]
    samples = means[comps] + eps
    return samples


# ----------------------------------------------------------------------
# 2. Generator energy E_gen(x) (polynomial fit to -log p̂(x))
# ----------------------------------------------------------------------

def fit_generative_energy_from_hist(samples,
                                    num_bins=300,
                                    poly_deg=6):
    """
    Fit E_gen(x) = w0 + w2 x^2 + w4 x^4 + w6 x^6
    (up to the specified even degree: 2,4,6) to -log p̂(x)
    from a histogram of the samples.

    Returns:
        w0, w2, w4, w6,
        x_fit, neglog_p_fit, Egen_fit
    """
    counts, edges = np.histogram(samples, bins=num_bins, density=True)
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0.0

    x_fit = x_centers[mask]
    neglog_p_fit = -np.log(counts[mask])

    cols = [np.ones_like(x_fit), x_fit**2]
    if poly_deg >= 4:
        cols.append(x_fit**4)
    if poly_deg >= 6:
        cols.append(x_fit**6)
    X = np.column_stack(cols)

    w, _, _, _ = np.linalg.lstsq(X, neglog_p_fit, rcond=None)

    w0 = w[0]
    w2 = w[1] if len(w) > 1 else 0.0
    w4 = w[2] if len(w) > 2 else 0.0
    w6 = w[3] if len(w) > 3 else 0.0

    Egen_fit = w0 + w2 * x_fit**2 + w4 * x_fit**4 + w6 * x_fit**6
    return w0, w2, w4, w6, x_fit, neglog_p_fit, Egen_fit


def E_gen(x, w0, w2, w4, w6):
    return w0 + w2 * x**2 + w4 * x**4 + w6 * x**6


# ----------------------------------------------------------------------
# 3. Energy histogram and band constraints
# ----------------------------------------------------------------------

def energy_histogram(samples, w0, w2, w4, w6, num_bins=150):
    """
    Map microstates x -> energies E_gen(x) and build a histogram.
    Returns (E_centers, p_emp, counts, edges).
    """
    E_samples = E_gen(samples, w0, w2, w4, w6)

    # Robust range to avoid extremes dominating bins
    E_min, E_max = np.percentile(E_samples, [0.1, 99.9])
    E_edges = np.linspace(E_min, E_max, num_bins + 1)

    counts, edges = np.histogram(E_samples, bins=E_edges, density=False)
    total = counts.sum()
    if total == 0:
        raise ValueError("No counts in energy histogram.")

    p_emp = counts.astype(float) / total
    E_centers = 0.5 * (edges[:-1] + edges[1:])
    return E_centers, p_emp, counts, edges


def build_core_shape_constraints(E_centers,
                                 p_emp,
                                 q_low=0.25,
                                 q_high=0.75):
    """
    Build linear constraints for band masses in energy-space.

    Bands are defined via empirical CDF quantiles q_low and q_high.
    """
    E = E_centers
    n = len(E)

    # Empirical CDF
    cdf = np.cumsum(p_emp)
    if cdf[-1] <= 0:
        raise ValueError("Empirical probabilities sum to zero.")
    cdf /= cdf[-1]

    def find_cut(q):
        idx = np.searchsorted(cdf, q)
        idx = np.clip(idx, 0, n - 1)
        return E[idx]

    E_low = find_cut(q_low)
    E_high = find_cut(q_high)

    band1 = (E <= E_low)
    band2 = (E > E_low) & (E <= E_high)
    band3 = (E > E_high)

    m1_emp = p_emp[band1].sum()
    m2_emp = p_emp[band2].sum()
    m3_emp = p_emp[band3].sum()

    rows = []
    rhs = []

    # Normalization
    rows.append(np.ones(n))
    rhs.append(1.0)

    # Band masses
    rows.append(band1.astype(float)); rhs.append(m1_emp)
    rows.append(band2.astype(float)); rhs.append(m2_emp)
    rows.append(band3.astype(float)); rhs.append(m3_emp)

    A = np.vstack(rows)
    b = np.array(rhs)
    bands = (band1, band2, band3)

    return A, b, bands


# ----------------------------------------------------------------------
# 4. Micro-resolved quadratic entropy and MaxEnt QP
# ----------------------------------------------------------------------

def build_micro_quadratic_Q_c(n, bands, alphas):
    """
    Build Q, c for

        -H_micro(p) = sum_k alpha_k sum_{i in band_k} (p_i^2 - p_i)

    so that objective(p) = 0.5 p^T Q p + c^T p = -H_micro(p).
    """
    Q = np.zeros((n, n))
    c = np.zeros(n)

    for band_mask, alpha in zip(bands, alphas):
        idx = np.where(band_mask)[0]
        for i in idx:
            Q[i, i] += 2.0 * alpha   # p_i^2 term
            c[i]    -= alpha        # -alpha * p_i
    return Q, c


def H_micro(p, bands, alphas):
    """
    H_micro(p) = sum_k alpha_k sum_{i in band_k} (p_i - p_i^2)
    """
    H = 0.0
    for band_mask, alpha in zip(bands, alphas):
        p_band = p[band_mask]
        H += alpha * (p_band.sum() - np.sum(p_band**2))
    return H


def maxent_qp(E_centers, p_emp,
              q_low=0.25, q_high=0.75,
              alphas=None):
    """
    Maximize H_micro(p) under normalization + band-mass constraints.

    Returns: p_star, H_star, bands, alphas
    """
    n = len(E_centers)
    A, b, bands = build_core_shape_constraints(
        E_centers, p_emp, q_low=q_low, q_high=q_high
    )

    if alphas is None:
        alphas = np.array([1.0, 0.5, 0.2])
    alphas = np.asarray(alphas)

    Q, c = build_micro_quadratic_Q_c(n, bands, alphas)

    def objective(p):
        return 0.5 * p @ (Q @ p) + c @ p  # = -H_micro(p)

    def grad(p):
        return Q @ p + c

    # Linear equality constraints A p = b
    cons = []
    for k in range(A.shape[0]):
        a_row = A[k, :]
        b_k = b[k]
        cons.append({
            'type': 'eq',
            'fun': lambda p, a=a_row, b0=b_k: np.dot(a, p) - b0,
            'jac': lambda p, a=a_row: a
        })

    bounds = [(0.0, 1.0) for _ in range(n)]

    # Start from empirical distribution (feasible)
    p0 = p_emp.copy()
    eps = 1e-12
    p0 = np.clip(p0, eps, None)
    p0 /= p0.sum()

    res = minimize(
        objective, p0,
        method='SLSQP',
        jac=grad,
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-12, 'maxiter': 20000, 'disp': False}
    )

    if not res.success:
        print("WARNING: MaxEnt QP did not converge:", res.message)

    p_star = res.x
    p_star = np.clip(p_star, 0.0, None)
    p_star /= p_star.sum()

    H_star = H_micro(p_star, bands, alphas)
    return p_star, H_star, bands, alphas


# ----------------------------------------------------------------------
# 5. H-theorem style relaxation and plots
# ----------------------------------------------------------------------

def run_H_theorem_demo():
    out_dir = "plots_H_theorem"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Plots will be saved in '{out_dir}/'")

    # Use the most separated mixture (mu = 3) so the energy structure is rich
    mu = 3.0
    N_samples = 200_000

    print(f"\n=== Building mixture with mu = {mu:.2f} ===")
    samples = sample_mixture(mu, N=N_samples)

    # Fit generator and energy histogram
    w0, w2, w4, w6, x_fit, neglog_p_fit, Egen_fit = \
        fit_generative_energy_from_hist(samples, num_bins=300, poly_deg=6)

    print(f"Fitted generator coefficients: "
          f"w0={w0:.3f}, w2={w2:.3f}, w4={w4:.3f}, w6={w6:.3f}")

    E_centers, p_emp, counts_E, edges_E = energy_histogram(
        samples, w0, w2, w4, w6, num_bins=150
    )

    # Compute MaxEnt state for H_micro
    p_star, H_star, bands, alphas = maxent_qp(E_centers, p_emp)
    H_emp = H_micro(p_emp, bands, alphas)

    print(f"Empirical H_micro(p_emp)   = {H_emp:.6f}")
    print(f"MaxEnt   H_micro(p_star)  = {H_star:.6f}")
    print(f"H_star - H_emp            = {H_star - H_emp:.6e}")

    # Simple relaxation: p_{t+1} = (1 - eta) p_t + eta p_star
    print("\nRunning relaxation flow (H-theorem demo)...")
    eta = 0.05
    n_iter = 60

    p_t = p_emp.copy()

    H_hist = [H_micro(p_t, bands, alphas)]
    dist_hist = [np.linalg.norm(p_t - p_star)]
    iters = [0]

    for t in range(1, n_iter + 1):
        p_t = (1.0 - eta) * p_t + eta * p_star
        H_hist.append(H_micro(p_t, bands, alphas))
        dist_hist.append(np.linalg.norm(p_t - p_star))
        iters.append(t)

    iters = np.array(iters)
    H_hist = np.array(H_hist)
    dist_hist = np.array(dist_hist)

    # Plot 1: distance to MaxEnt state
    plt.figure(figsize=(6, 4))
    plt.semilogy(iters, dist_hist, 'o-')
    plt.xlabel('iteration')
    plt.ylabel(r'$||p_t - p^\star||_2$')
    plt.title('Convergence to generative MaxEnt state')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "H_theorem_distance_vs_iter.png"))
    plt.close()

    # Plot 2: generative entropy along the flow
    plt.figure(figsize=(6, 4))
    plt.plot(iters, H_hist, 'o-', label=r'$H_{\mathrm{micro}}(p_t)$')
    plt.axhline(H_star, color='k', linestyle='--',
                label=r'$H_{\mathrm{micro}}(p^\star)$ (MaxEnt)')
    plt.xlabel('iteration')
    plt.ylabel('generative entropy $H_{\mathrm{micro}}$')
    plt.title('H-theorem: generative entropy along relaxation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "H_theorem_entropy_vs_iter.png"))
    plt.close()

    print(f"Done. H-theorem plots saved in '{out_dir}/'.")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    run_H_theorem_demo()
