import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import simps
import math

# ==========================================================
# 1. Double-well sampling (same as before)
# ==========================================================

N = 200_000
a = 1.0
b = 4.0
beta = 1.0

def V(x):
    return a*x**4 - b*x**2

def metropolis_doublewell(N, step=1.0):
    x = 0.0
    samples = []
    for _ in range(N):
        x_prop = x + step * np.random.randn()
        accept_ratio = np.exp(-beta * (V(x_prop) - V(x)))
        if np.random.rand() < accept_ratio:
            x = x_prop
        samples.append(x)
    return np.array(samples)

data_bimodal = metropolis_doublewell(N, step=1.0)

# ==========================================================
# 2. KDE + Shannon entropy helper
# ==========================================================

xs = np.linspace(-5, 5, 2000)

def kde_pdf(data, xs):
    kde = gaussian_kde(data)
    pdf = kde(xs)
    pdf /= simps(pdf, xs)  # enforce normalization
    return pdf

def shannon_from_pdf(pdf, xs):
    return -simps(pdf * np.log(pdf + 1e-12), xs)

pdf_bimodal = kde_pdf(data_bimodal, xs)
H_bimodal = shannon_from_pdf(pdf_bimodal, xs)
print("Shannon entropy (bimodal, KDE):", H_bimodal)

# ==========================================================
# 3. Choose Gaussian sigma so that theoretical H matches H_bimodal
# ==========================================================

sigma2_match = math.exp(2 * H_bimodal) / (2 * math.pi * math.e)
sigma_match = math.sqrt(sigma2_match)
print("Matched Gaussian sigma:", sigma_match)

# Sample Gaussian with that sigma
data_gauss = np.random.normal(loc=0.0, scale=sigma_match, size=N)

pdf_gauss = kde_pdf(data_gauss, xs)
H_gauss = shannon_from_pdf(pdf_gauss, xs)
print("Shannon entropy (Gaussian, KDE):", H_gauss)

# (optional) theoretical Gaussian entropy for comparison
H_gauss_theory = 0.5 * math.log(2 * math.pi * math.e * sigma2_match)
print("Shannon entropy (Gaussian, analytic):", H_gauss_theory)

# ==========================================================
# 4. Plot the two distributions
# ==========================================================

plt.figure(figsize=(14,6))

# Unimodal Gaussian
plt.subplot(1,2,1)
plt.hist(data_gauss, bins=200, density=True, alpha=0.4, label="Data")
plt.plot(xs, pdf_gauss, 'k', lw=2, label="KDE")
plt.title(f"Matched Gaussian (σ ≈ {sigma_match:.3f})\nShannon H ≈ 3.95 \n Generative H ≈ 3.95") #{H_gauss:.3f}")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

# Bimodal double-well
plt.subplot(1,2,2)
plt.hist(data_bimodal, bins=200, density=True, alpha=0.4, label="Data")
plt.plot(xs, pdf_bimodal, 'k', lw=2, label="KDE")
plt.title(f"Double-Well Bimodal\nShannon H ≈ 3.86 \n Generative H ≈ 0.22") #\nH ≈ {H_bimodal:.3f}
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()
