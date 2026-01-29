# benchmark_water_with_Cv_stochastic_v3.py

# ─── Imports, constants & simulation parameters ─────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from openmm import unit, openmm as mm
from openmm.app import Simulation, Modeller, ForceField, PME, HBonds, Topology

# -----------------------------------------------------------------------------
# 0. Physical constants & conversion
# -----------------------------------------------------------------------------
# Gas constant R in kJ/(mol·K)
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA   # J/(mol·K)
R  = 0.001 * kB.value_in_unit(unit.joule / (unit.mole * unit.kelvin))
# R ≈ 0.008314462618 kJ/(mol·K)

# -----------------------------------------------------------------------------
# 1. Simulation parameters (same at each T)
# -----------------------------------------------------------------------------
padding_nm  = 1.5        # solvent padding [nm]
dt_fs       = 2.0        # timestep [fs]
equil_steps = 500        # equilibration steps (~10 ps)
prod_steps  = 50_000     # production steps (~100 ps)
nsnap       = 500        # number of snapshots per T
interval    = prod_steps // nsnap

# ─── NPT runner returning KE_trans, KE_rot, PE, VOL ─────────────────────────
def run_npt_collect_all(T, P_bar=1.0, padding=padding_nm, seed=None):
    """
    Run TIP3P‐water NPT @ T [K], P_bar [bar].
    Returns four length‐nsnap arrays (units in kJ/mol or L):
      - KE_trans_vals: translational kinetic energy per mole   [kJ/mol]
      - KE_rot_vals:   rotational kinetic energy per mole      [kJ/mol]
      - PE_vals:       potential energy per mole               [kJ/mol]
      - VOL_vals:      instantaneous volume                    [L]
    """
    # --- Build & solvate ---------------------------------------------------
    modeller = Modeller(Topology(), [])
    ff       = ForceField('tip3p.xml')
    modeller.addSolvent(ff, padding=padding * unit.nanometer)

    # --- System + barostat + integrator ------------------------------------
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds
    )
    system.addForce(mm.MonteCarloBarostat(P_bar * unit.bar, T * unit.kelvin, 25))

    integrator = mm.LangevinIntegrator(
        T * unit.kelvin,
        1.0 / unit.picosecond,
        dt_fs * unit.femtosecond
    )
    # Seed the stochastic integrator if requested
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))

    sim = Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()

    # Set velocities (no randomSeed kw arg in your OpenMM version)
    sim.context.setVelocitiesToTemperature(T * unit.kelvin)

    # --- Equilibrate ---
    sim.step(equil_steps)

    # --- Precompute molecule grouping & masses ---
    atoms_by_res = {}
    for atom in modeller.topology.atoms():
        atoms_by_res.setdefault(atom.residue.index, []).append(atom.index)
    n_mol = len(atoms_by_res)

    masses = np.array([
        atom.element.mass.value_in_unit(unit.dalton)
        for atom in modeller.topology.atoms()
    ])  # dalton→kg later
    dalton_to_kg = 1.66053906660e-27

    # --- Storage arrays ---
    KE_trans_vals = np.zeros(nsnap)
    KE_rot_vals   = np.zeros(nsnap)
    PE_vals       = np.zeros(nsnap)
    VOL_vals      = np.zeros(nsnap)

    # --- Production & data collection ---
    for i in range(nsnap):
        sim.step(interval)
        state = sim.context.getState(
            getEnergy=True, getVelocities=True, getPositions=True
        )

        # potential + total KE (kJ/mol)
        PE_vals[i] = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        ke_tot     = state.getKineticEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

        # instantaneous volume (nm^3 → L)
        box      = state.getPeriodicBoxVectors(asNumpy=True)._value
        vol_nm3  = np.linalg.det(box)
        VOL_vals[i] = vol_nm3 * 1e-27 * 1000.0

        # velocities → m/s
        vel_nm_ps = state.getVelocities(asNumpy=True)._value
        vel_m_s   = vel_nm_ps * 1e-3 / 1e-12

        # per‐molecule translational & rotational KE
        ke_t_sum = 0.0
        ke_r_sum = 0.0
        for res_idx, atom_idxs in atoms_by_res.items():
            m_i = masses[atom_idxs] * dalton_to_kg
            v_i = vel_m_s[atom_idxs]

            M     = m_i.sum()
            v_cm  = (m_i[:, None] * v_i).sum(axis=0) / M
            ke_t  = 0.5 * M * (v_cm @ v_cm)

            ke_atoms = 0.5 * (m_i * np.sum(v_i**2, axis=1)).sum()
            ke_r    = ke_atoms - ke_t

            ke_t_sum += ke_t
            ke_r_sum += ke_r

        # convert box‐total J → kJ/mol per molecule
        conv = (
            1e-3
            * unit.AVOGADRO_CONSTANT_NA.value_in_unit(unit.mole**-1)
            / n_mol
        )
        KE_trans_vals[i] = ke_t_sum * conv
        KE_rot_vals[i]   = ke_r_sum * conv

    return KE_trans_vals, KE_rot_vals, PE_vals, VOL_vals

# ─── Collect data at each temperature (with replicas) ──────────────────────
n_rep = 15   # number of replicas per temperature
temperature_list = np.arange(280, 371, 2)   # 280 … 370 K (1 K spacing)
data_all = {}  # data_all[T] → list of replicas (each is dict of arrays)

for T in temperature_list:
    print(f"Running {n_rep} replicas @ {T} K …")
    reps_T = []

    for rep in range(n_rep):
        print(f"   Replica {rep+1}/{n_rep}")
        seed = np.random.randint(0, 2**31 - 1)
        KE_tr, KE_rot, PE, VOL = run_npt_collect_all(T, P_bar=1.0, seed=seed)
        reps_T.append({
            'KE_trans': KE_tr,
            'KE_rot'  : KE_rot,
            'PE'      : PE,
            'VOL'     : VOL
        })

    data_all[T] = reps_T

print("✓ Done collecting KE_trans, KE_rot, PE, VOL for all T and replicas.")

# ─── Global (E,V) grid for Shannon & NIST reference function ───────────────
# Build global PE, VOL arrays from ALL temperatures and ALL replicas
all_PE = np.hstack([
    rep['PE']
    for T in temperature_list
    for rep in data_all[T]
])

all_VOL = np.hstack([
    rep['VOL']
    for T in temperature_list
    for rep in data_all[T]
])

# Standard 2D grid
n_bins_E, n_bins_V = 100, 100
E_edges = np.linspace(all_PE.min(),  all_PE.max(),  n_bins_E + 1)
V_edges = np.linspace(all_VOL.min(), all_VOL.max(), n_bins_V + 1)

dE = E_edges[1] - E_edges[0]
dV = V_edges[1] - V_edges[0]
bin_area = dE * dV

# NIST water-vapor entropy correlation
def real_entropy_kJ_per_molK(T):
    A, B, C, D = -203.6060, 1523.290, -3196.413, 2474.455
    Ecoef, G = 3.855326, -488.7163
    t = T / 1000.0
    S_J = A*np.log(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - Ecoef/(2*t**2) + G
    return S_J / 1000.0

# ─── CV at 360 K: 2-D Gaussian baseline ────────────────────────────────────
from sklearn.model_selection import KFold
from numpy.linalg import slogdet, inv

T_CV      = 360.0     # K
n_folds   = 5
rho_grid  = [1e-5, 1e-4, 1e-3, 1e-2]

replicas_ev = []      # list of (PE, VOL) arrays per replica

for rep in range(2):
    print(f"⇢  Running held-out replica {rep+1}/2 @ {T_CV:g} K …")
    KE_tr, KE_rot, PE, VOL = run_npt_collect_all(
        T_CV, P_bar=1.0, padding=padding_nm
    )
    replicas_ev.append(np.vstack([PE, VOL]).T)   # shape (n_win, 2)

cv_scores = np.zeros((len(rho_grid), 2, n_folds))  # ρ × replica × fold

for r, X in enumerate(replicas_ev):
    kf = KFold(n_splits=n_folds, shuffle=False)
    for fold, (idx_tr, idx_val) in enumerate(kf.split(X)):
        X_tr, X_val = X[idx_tr], X[idx_val]

        μ = X_tr.mean(axis=0)
        Σ = np.cov(X_tr, rowvar=False, ddof=0)

        for iρ, ρ in enumerate(rho_grid):
            Σ_reg = Σ + ρ * np.eye(2)            # ridge-regularised
            sign, logdet = slogdet(Σ_reg)
            if sign <= 0:
                raise RuntimeError("Σ not PD even after ridge")
            Σ_inv = inv(Σ_reg)

            diff   = X_val - μ
            ll_val = -0.5 * np.sum(diff @ Σ_inv * diff, axis=1) \
                     - 0.5 * logdet - np.log(2*np.pi)
            score  = -ll_val.mean()               # lower is better
            cv_scores[iρ, r, fold] = score

mean_sc = cv_scores.mean(axis=(1, 2))
sem_sc  = cv_scores.std(axis=(1, 2), ddof=1) / np.sqrt(cv_scores.size // len(rho_grid))

plt.figure(figsize=(6, 4))
plt.errorbar(
    np.log10(rho_grid), mean_sc, yerr=sem_sc,
    marker='o', capsize=4, lw=1.5
)
plt.xlabel('log$_{10}$ ridge $\\rho$')
plt.ylabel('5-fold CV proper score (−log p)')
plt.title('360K Water (E,V) 2-D Gaussian — held-out 2 rep × 5-fold CV')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Fig_S5_Water_CV.png', dpi=300)
plt.show()

# ─── Entropies with replica averaging & SEM ────────────────────────────────
import math
from scipy.special import digamma, gammaln

def gamma_entropy_nats(mu, var, eps=1e-16):
    mu  = max(mu, eps)
    var = max(var, eps)
    k     = mu**2 / var
    theta = var / mu
    return k + math.log(theta) + gammaln(k) - (k - 1)*digamma(k)

S_EV_ME, S_EV_SH = [], []
S_KET_ME, S_KET_SH = [], []
S_KER_ME, S_KER_SH = [], []

S_EV_ME_sem, S_EV_SH_sem = [], []
S_KET_ME_sem, S_KET_SH_sem = [], []
S_KER_ME_sem, S_KER_SH_sem = [], []

def avg_sem(arr):
    arr = np.array(arr)
    return arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr))

for T in temperature_list:

    rep_ev_me  = []; rep_ev_sh  = []
    rep_ket_me = []; rep_ket_sh = []
    rep_ker_me = []; rep_ker_sh = []

    for rep in data_all[T]:
        PE = rep['PE']
        VOL = rep['VOL']
        KE_tr = rep['KE_trans']
        KE_rot = rep['KE_rot']

        # EV MaxEnt
        X = np.vstack([PE, VOL]).T
        mu = X.mean(axis=0)
        Sig = np.cov(X, rowvar=False, ddof=0)
        detSig = np.linalg.det(Sig)
        rep_ev_me.append(0.5 * math.log((2*np.pi*math.e)**2 * detSig) * R)

        # EV Shannon
        counts, _, _ = np.histogram2d(PE, VOL, bins=[E_edges, V_edges])
        p = counts / counts.sum()
        mask = p > 0
        rep_ev_sh.append(
            (-np.sum(p[mask]*np.log(p[mask])) + math.log(bin_area)) * R
        )

        # KE_trans MaxEnt
        mu_tr, var_tr = KE_tr.mean(), KE_tr.var(ddof=0)
        rep_ket_me.append(gamma_entropy_nats(mu_tr, var_tr) * R)

        # KE_trans Shannon
        n = len(KE_tr)
        q75, q25 = np.percentile(KE_tr, [75, 25])
        h = max(2*(q75 - q25)/np.cbrt(n), 1e-12)
        ct, edges = np.histogram(
            KE_tr, bins=np.arange(KE_tr.min(), KE_tr.max()+h, h)
        )
        pt = ct/ct.sum(); mask = pt > 0
        rep_ket_sh.append(
            (-np.sum(pt[mask]*np.log(pt[mask])) + math.log(h)) * R
        )

        # KE_rot MaxEnt
        mu_rt, var_rt = KE_rot.mean(), KE_rot.var(ddof=0)
        rep_ker_me.append(gamma_entropy_nats(mu_rt, var_rt) * R)

        # KE_rot Shannon
        n = len(KE_rot)
        q75, q25 = np.percentile(KE_rot, [75, 25])
        h = max(2*(q75 - q25)/np.cbrt(n), 1e-12)
        ct, edges = np.histogram(
            KE_rot, bins=np.arange(KE_rot.min(), KE_rot.max()+h, h)
        )
        pt = ct/ct.sum(); mask = pt > 0
        rep_ker_sh.append(
            (-np.sum(pt[mask]*np.log(pt[mask])) + math.log(h)) * R
        )

    # Average + SEM across replicas
    for arr, store, semstore in [
        (rep_ev_me,  S_EV_ME,  S_EV_ME_sem),
        (rep_ev_sh,  S_EV_SH,  S_EV_SH_sem),
        (rep_ket_me, S_KET_ME, S_KET_ME_sem),
        (rep_ket_sh, S_KET_SH, S_KET_SH_sem),
        (rep_ker_me, S_KER_ME, S_KER_ME_sem),
        (rep_ker_sh, S_KER_SH, S_KER_SH_sem),
    ]:
        a, b = avg_sem(arr)
        store.append(a)
        semstore.append(b)

# ─── Total entropies, anchoring, and S(T) plots ────────────────────────────
temps = temperature_list

S_EV_ME   = np.array(S_EV_ME)
S_EV_SH   = np.array(S_EV_SH)
S_KET_ME  = np.array(S_KET_ME)
S_KET_SH  = np.array(S_KET_SH)
S_KER_ME  = np.array(S_KER_ME)
S_KER_SH  = np.array(S_KER_SH)

# Raw (non-anchored) totals
S_tot_ME_raw = S_EV_ME + S_KET_ME + S_KER_ME
S_tot_SH_raw = S_EV_SH + S_KET_SH + S_KER_SH

# Total SEM (independent components → add in quadrature)
S_tot_ME_sem = np.sqrt(
    np.array(S_EV_ME_sem)**2 +
    np.array(S_KET_ME_sem)**2 +
    np.array(S_KER_ME_sem)**2
)
S_tot_SH_sem = np.sqrt(
    np.array(S_EV_SH_sem)**2 +
    np.array(S_KET_SH_sem)**2 +
    np.array(S_KER_SH_sem)**2
)

# NIST curve (absolute)
S_NIST = np.array([real_entropy_kJ_per_molK(T) for T in temps])

# Reference T0 for anchoring
T0 = float(temps[0])
S0 = real_entropy_kJ_per_molK(T0)

# Anchored totals: shift so S(T0) = S_NIST(T0)
S_tot_ME_anch = S_tot_ME_raw + (S0 - S_tot_ME_raw[0])
S_tot_SH_anch = S_tot_SH_raw + (S0 - S_tot_SH_raw[0])

# ── Plot 1: NON-anchored S(T) (absolute raw MD entropies) ──────────────────
plt.figure(figsize=(8, 5))
plt.errorbar(temps, S_tot_ME_raw, yerr=S_tot_ME_sem,
             fmt='o-', label='Generative total (raw)')
plt.errorbar(temps, S_tot_SH_raw, yerr=S_tot_SH_sem,
             fmt='x-', label='Shannon total (raw)')
plt.plot(temps, S_NIST, '--k', label='NIST (absolute)')
plt.xlabel('Temperature (K)')
plt.ylabel('Entropy (kJ/mol/K)')
plt.title('Entropy vs T (raw, non-anchored)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Fig_Water_S_vs_T_raw.png', dpi=300)
plt.show()

# ── Plot 2: ANCHORED S(T) (as in the paper figure) ─────────────────────────
plt.figure(figsize=(8, 5))
plt.errorbar(temps, S_tot_ME_anch, yerr=S_tot_ME_sem,
             fmt='o-', label='Generative total (anchored)')
plt.errorbar(temps, S_tot_SH_anch, yerr=S_tot_SH_sem,
             fmt='x-', label='Shannon total (anchored)')
plt.plot(temps, S_NIST, '--k', label='NIST')
plt.xlabel('Temperature (K)')
plt.ylabel('Entropy (kJ/mol/K)')
plt.title(f'Entropy vs T (anchored at {T0:.0f} K)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Fig_Water_S_vs_T_anchored.png', dpi=300)
plt.show()

# ─── Helper: wider symmetric derivative using up to ±span neighbours ───────
def symmetric_derivative(y, x, span=2):
    """
    Compute dy/dx using a symmetric finite-difference stencil.

    For each point i, uses the largest symmetric stencil up to ±span
    that fits inside the array:
        dy_i ≈ [y_{i+s} - y_{i-s}] / [x_{i+s} - x_{i-s}]
    At the extreme endpoints, falls back to one-sided differences.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    dy = np.zeros(n, float)

    for i in range(n):
        s = min(span, i, n - 1 - i)
        if s == 0:
            if i == 0:
                dy[i] = (y[1] - y[0]) / (x[1] - x[0])
            elif i == n - 1:
                dy[i] = (y[-1] - y[-2]) / (x[-1] - x[-2])
            else:
                dy[i] = 0.0
        else:
            dy[i] = (y[i + s] - y[i - s]) / (x[i + s] - x[i - s])
    return dy

# ─── Compute C(T) from S(T):  C ≈ T dS/dT (with wider stencil) ─────────────
temps_arr = np.array(temps, float)
DERIV_SPAN = 2   # or 3, as you prefer

dS_ME_dT   = symmetric_derivative(S_tot_ME_raw, temps_arr, span=DERIV_SPAN)
dS_SH_dT   = symmetric_derivative(S_tot_SH_raw, temps_arr, span=DERIV_SPAN)
dS_NIST_dT = symmetric_derivative(S_NIST,        temps_arr, span=DERIV_SPAN)

C_ME   = temps_arr * dS_ME_dT
C_SH   = temps_arr * dS_SH_dT
C_NIST = temps_arr * dS_NIST_dT

# SEM propagation: apply same stencil to S_tot_ME_sem
dS_ME_sem_dT = symmetric_derivative(S_tot_ME_sem, temps_arr, span=DERIV_SPAN)
C_ME_sem     = np.abs(temps_arr * dS_ME_sem_dT)

# ─── Save S(T) and C(T) to Excel ───────────────────────────────────────────
results_df = pd.DataFrame({
    "T_K": temps_arr,
    "S_gen_raw_kJmolK": S_tot_ME_raw,
    "S_gen_anch_kJmolK": S_tot_ME_anch,
    "S_gen_sem_kJmolK": S_tot_ME_sem,
    "S_sh_raw_kJmolK": S_tot_SH_raw,
    "S_sh_anch_kJmolK": S_tot_SH_anch,
    "S_sh_sem_kJmolK": S_tot_SH_sem,
    "S_NIST_kJmolK": S_NIST,
    "C_gen_kJmolK": C_ME,
    "C_gen_sem_kJmolK": C_ME_sem,
    "C_sh_kJmolK": C_SH,
    "C_NIST_kJmolK": C_NIST,
})
results_df.to_excel("water_entropy_Cv_results.xlsx", index=False)

results_df.to_excel("water_entropy_Cv_results.xlsx", index=False)

# ─── Plot C(T) curves ──────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.errorbar(temps_arr, C_ME, yerr=C_ME_sem,
             fmt='o-', label='Generative $C(T)$')
plt.plot(temps_arr, C_SH, 'x-', label='Shannon $C(T)$')
plt.plot(temps_arr, C_NIST, '--k', label='NIST $C_P$')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat capacity (kJ/mol/K)')
plt.title('Heat capacity from averaged S(T) (wider stencil)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ─── GMM CV (Gaussian vs GMM) ──────────────────────────────────────────────
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

rho_grid = [1e-5, 1e-4, 1e-3, 1e-2]
n_repl   = len(replicas_ev)
n_folds  = 2  # as in your last version

cv_gauss = np.zeros((len(rho_grid), n_repl, n_folds))
cv_gmm   = np.zeros_like(cv_gauss)

for r, X in enumerate(replicas_ev):
    kf = KFold(n_splits=n_folds, shuffle=False)
    for fold, (i_tr, i_va) in enumerate(kf.split(X)):
        X_tr, X_va = X[i_tr], X[i_va]

        μ  = X_tr.mean(axis=0)
        Σ  = np.cov(X_tr, rowvar=False, ddof=0)
        for iρ, ρ in enumerate(rho_grid):
            Σr = Σ + ρ * np.eye(2)
            sign, ld = np.linalg.slogdet(Σr)
            Σi = np.linalg.inv(Σr)
            d  = X_va - μ
            ll = -0.5 * np.sum(d @ Σi * d, axis=1) - 0.5 * ld - np.log(2 * np.pi)
            cv_gauss[iρ, r, fold] = -ll.mean()

        for iρ in range(len(rho_grid)):
            gmm = GaussianMixture(
                n_components=2,
                covariance_type='full',
                random_state=0
            )
            gmm.fit(X_tr)
            logp = gmm.score_samples(X_va)
            cv_gmm[iρ, r, fold] = -np.mean(logp)

mean_gauss = cv_gauss.mean(axis=(1, 2))
sem_gauss  = cv_gauss.std(axis=(1, 2), ddof=1) / np.sqrt(cv_gauss.size//len(rho_grid))
mean_gmm   = cv_gmm.mean(axis=(1, 2))
sem_gmm    = cv_gmm.std(axis=(1, 2), ddof=1) / np.sqrt(cv_gmm.size//len(rho_grid))

plt.figure(figsize=(6, 4))
xs = np.log10(rho_grid)
plt.errorbar(xs, mean_gauss, marker='o', capsize=4, yerr=sem_gauss, label='2-D Gaussian')
plt.errorbar(xs, mean_gmm,   marker='^', capsize=4, yerr=sem_gmm,   label='GMM (2 components)')
plt.xlabel('log$_{10}$ ridge $\\rho$ (for Gaussian)')
plt.ylabel('5-fold CV proper score $(-\\overline{\\ln p})$')
plt.title('360 K Water (E,V): Gaussian vs GMM baseline')
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Fig_S5_Water_CV_GMM.png', dpi=300)
plt.show()

# ─── 4-D GMM-based entropy vs T ────────────────────────────────────────────
gmm_ent_4d = []
gmm_sem_4d = []

for T in temperature_list:
    PE_all  = np.hstack([rep['PE']        for rep in data_all[T]])
    VOL_all = np.hstack([rep['VOL']       for rep in data_all[T]])
    KET_all = np.hstack([rep['KE_trans']  for rep in data_all[T]])
    KER_all = np.hstack([rep['KE_rot']    for rep in data_all[T]])

    X4 = np.vstack([PE_all, VOL_all, KET_all, KER_all]).T

    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=0
    )
    gmm.fit(X4)
    logp = gmm.score_samples(X4)

    h_nat = -np.mean(logp)
    h_kJ  = h_nat * R
    gmm_ent_4d.append(h_kJ)

    gmm_sem_4d.append(logp.std(ddof=1) / np.sqrt(len(logp)) * R)

h0_4d = gmm_ent_4d[temperature_list.tolist().index(T0)]
offset_4d = S0 - h0_4d
gmm_ent_4d = [h + offset_4d for h in gmm_ent_4d]

# ─── 2-D (E,V) GMM-based entropy vs T ───────────────────────────────────
from sklearn.mixture import GaussianMixture

gmm_ent_raw = []   # non-anchored GMM entropy
gmm_sem      = []

for T in temperature_list:
    # pool all replicas at this T
    PE_all  = np.hstack([rep['PE']  for rep in data_all[T]])
    VOL_all = np.hstack([rep['VOL'] for rep in data_all[T]])
    X = np.vstack([PE_all, VOL_all]).T  # shape (n_samples, 2)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=0
    )
    gmm.fit(X)
    logp = gmm.score_samples(X)   # log p(x) per sample

    h_nat = -logp.mean()          # differential entropy in nats
    h_kJ  = h_nat * R             # kJ·mol⁻¹·K⁻¹
    gmm_ent_raw.append(h_kJ)

    sem_nat = logp.std(ddof=1) / np.sqrt(len(logp))
    gmm_sem.append(sem_nat * R)

gmm_ent_raw = np.array(gmm_ent_raw)
gmm_sem     = np.array(gmm_sem)

# Anchored GMM: shift so S_GMM(T0) = S_NIST(T0)
off_gmm        = S0 - gmm_ent_raw[0]
gmm_ent_anch   = gmm_ent_raw + off_gmm

# ── Plot A: NON-anchored S(T) + GMM (raw) ─────────────────────────────
plt.figure(figsize=(8, 5))
plt.errorbar(temps, S_tot_ME_raw, yerr=S_tot_ME_sem,
             fmt='o-', label='Generative total (raw)')
plt.errorbar(temps, S_tot_SH_raw, yerr=S_tot_SH_sem,
             fmt='x-', label='Shannon total (raw)')
plt.plot(temps, S_NIST, '--k', label='NIST (absolute)')
plt.errorbar(temps, gmm_ent_raw, yerr=gmm_sem,
             marker='^', linestyle='--', capsize=3,
             label='GMM (2 components, raw)')

plt.xlabel('Temperature (K)')
plt.ylabel('Entropy $S$ (kJ·mol$^{-1}$·K$^{-1}$)')
plt.title('Water entropy vs T (raw, non-anchored, with GMM)')
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Fig_Water_S_vs_T_GMM_raw.png', dpi=300)
plt.show()

# ── Plot B: ANCHORED S(T) + GMM (for main text) ───────────────────────
plt.figure(figsize=(8, 5))
plt.errorbar(temps, S_tot_ME_anch, yerr=S_tot_ME_sem,
             fmt='o-', label='Generative total (anchored)')
plt.errorbar(temps, S_tot_SH_anch, yerr=S_tot_SH_sem,
             fmt='x-', label='Shannon total (anchored)')
plt.plot(temps, S_NIST, '--k', label='NIST')
plt.errorbar(temps, gmm_ent_anch, yerr=gmm_sem,
             marker='^', linestyle='--', capsize=3,
             label='GMM (2 components, anchored)')

plt.xlabel('Temperature (K)')
plt.ylabel('Entropy $S$ (kJ·mol$^{-1}$·K$^{-1}$)')
plt.title(f'Water entropy vs T with GMM (anchored at {T0:.0f} K)')
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Fig_Water_S_vs_T_GMM_anchored.png', dpi=300)
plt.show()
