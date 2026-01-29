#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# QM_Solving_SP500_v15.py
#
# Improvements over v14:
#  (1) QP computed for ALL years (train coverage -> ~100%)
#  (2) Fixed histogram bin edges for QP across time (comparability/stability)
#  (3) Stronger tail constraints: 1σ, 2σ, 3σ
#  (4) Stronger QP-derived score: abs(lam_1s)+abs(lam_2s)+2*abs(lam_3s)
#  (5) Train-only z-score and sign selection still respected (no leakage)
#
# Expectation: Learned-QP tail score AUC should improve vs v14’s 0.683.
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# ------------------ CONFIG --------------------------------------------------
START_YR     = 2000
WINDOW_DAYS  = 21

BINS_ROLL    = 50
Q_TSALLIS    = 3
GAMMA_TAIL   = 1.0
GAMMA_VAR    = 1.0
GAMMA_SKEW   = 1.0
EPS          = 1e-16

TRAIN_END_YR = 2022
TEST_YR      = 2024

# QP now computed for all years (coverage fix)
QP_YR_START  = 2000

# QP histogram settings (fixed bin edges)
QP_BINS      = 41
QP_RANGE_SIGMA_FIXED = 8.0   # range = +/- (this)*sigma_train (fixed across time)
QP_PSEUDOCOUNT = 1e-2        # smoothing
QP_DEG         = 3
QP_U_GRID_N    = 200
QP_RIDGE       = 1e-5
QP_ETA         = 0.0

QP_SOLVER_PRIMARY  = "OSQP"
QP_SOLVER_FALLBACK = "SCS"
# ----------------------------------------------------------------------------

def safe_series(x):
    return x.squeeze() if isinstance(x, pd.DataFrame) else x

def z_from_index(series: pd.Series, idx_for_stats: pd.Index, eps: float = 1e-12) -> pd.Series:
    s = series.loc[idx_for_stats].dropna()
    if len(s) < 20:
        return series * np.nan
    mu = float(s.mean())
    sd = float(s.std())
    if sd < eps:
        return (series - mu) * 0.0
    return (series - mu) / sd

def choose_sign_on_index(score: pd.Series, y: pd.Series, idx_for_sign: pd.Index) -> float:
    s = score.loc[idx_for_sign].dropna()
    yy = y.loc[s.index]
    if yy.nunique() < 2 or len(s) < 50:
        return +1.0
    auc_pos = roc_auc_score(yy, s)
    auc_neg = roc_auc_score(yy, -s)
    return +1.0 if auc_pos >= auc_neg else -1.0

def solve_stage1_qp_shannon_core(p_hat, x_centers, sigma_ref, deg=3, u_grid_n=200,
                                ridge=1e-5, eta=0.0, tail_sigmas=(1.0,2.0,3.0),
                                solver_primary="OSQP", solver_fallback="SCS"):
    import cvxpy as cp

    p_hat = np.asarray(p_hat, float)
    x_centers = np.asarray(x_centers, float)

    nz = p_hat > 0
    p = p_hat[nz]
    x = x_centers[nz]

    feats = [np.ones_like(x), x, x**2, np.abs(x)**4]
    for k in tail_sigmas:
        feats.append((np.abs(x) > k * sigma_ref).astype(float))
    F = np.column_stack(feats)  # [1,x,x^2,|x|^4,tail1,tail2,tail3]
    a = cp.Variable(deg)
    lam = cp.Variable(F.shape[1])

    U = np.column_stack([p**m for m in range(1, deg + 1)])
    g_p = 1.0 + cp.log(p) + U @ a

    w = np.sqrt(np.maximum(p, 1e-12))
    r = g_p - F @ lam
    obj = cp.sum_squares(cp.multiply(w, r)) + ridge * cp.sum_squares(a)

    u_min = max(float(p.min()), 1e-10)
    u_grid = np.exp(np.linspace(np.log(u_min), np.log(1.0), u_grid_n))
    dU = np.column_stack([m * (u_grid ** (m - 1)) for m in range(1, deg + 1)])
    gprime = 1.0 / u_grid + dU @ a

    prob = cp.Problem(cp.Minimize(obj), [gprime >= eta])

    solved = False
    try:
        if solver_primary.upper() == "OSQP":
            prob.solve(
                solver=cp.OSQP,
                verbose=False,
                max_iter=200000,
                eps_abs=1e-8,
                eps_rel=1e-8,
                polish=True
            )
        else:
            prob.solve(solver=getattr(cp, solver_primary.upper()), verbose=False)
        solved = True
    except Exception:
        solved = False

    if (not solved) or (prob.status not in ("optimal","optimal_inaccurate")):
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-5)
        except Exception:
            pass

    if prob.status not in ("optimal","optimal_inaccurate"):
        raise RuntimeError(f"QP failed: {prob.status}")

    return np.array(lam.value).ravel()

# ------------------ Download data ------------------------------------------
start = dt.datetime(START_YR, 1, 1)
end   = dt.datetime.today()

df = yf.download("^GSPC", start=start, end=end, progress=False).dropna()
prices = safe_series(df["Close"])
returns = safe_series(prices.pct_change().dropna())
rv_21   = safe_series(returns.rolling(WINDOW_DAYS).var().dropna())

rvals, idxs = returns.values, returns.index
sigma_full = float(np.std(rvals, ddof=1))

# Fixed sigma_train and fixed bin edges for QP
train_pool_mask = returns.index.year <= TRAIN_END_YR
r_train = returns.loc[train_pool_mask].values
sigma_train = float(np.std(r_train, ddof=1))
L = QP_RANGE_SIGMA_FIXED * sigma_train
qp_edges = np.linspace(-L, L, QP_BINS + 1)
qp_centers = qp_edges[:-1] + np.diff(qp_edges)/2

# Rolling outputs
H_vals, TM_vals, VM_vals, SM_vals, Tq_vals = [], [], [], [], []
QP_TAILSCORE_vals = []
dates = []

for i in range(WINDOW_DAYS, len(rvals) + 1):
    window = rvals[i-WINDOW_DAYS:i]
    t = idxs[i-1]

    # Baselines
    counts, edges = np.histogram(window, bins=BINS_ROLL)
    tot = counts.sum()
    if tot <= 0:
        continue
    p_all = counts / tot
    centers = edges[:-1] + np.diff(edges)/2
    nz = p_all > 0

    H = -np.sum(p_all[nz] * np.log(p_all[nz] + EPS))
    mask_tail = nz & (np.abs(centers) > sigma_full)
    S_tail = H - GAMMA_TAIL * p_all[mask_tail].sum()
    mu2 = np.sum(p_all[nz] * centers[nz]**2)
    S_var = H - GAMMA_VAR * mu2
    mu3_abs = abs(np.sum(p_all[nz] * centers[nz]**3))
    S_skew = H - GAMMA_SKEW * mu3_abs
    q = Q_TSALLIS
    Tq = (1 - np.sum(p_all[nz]**q)) / (q - 1)

    # QP on fixed edges (more stable)
    qp_score = np.nan
    if t.year >= QP_YR_START:
        try:
            c_w, _ = np.histogram(window, bins=qp_edges)
            c_w = c_w.astype(float) + QP_PSEUDOCOUNT
            p_w = c_w / c_w.sum()

            lam = solve_stage1_qp_shannon_core(
                p_w, qp_centers, sigma_ref=sigma_train,
                deg=QP_DEG,
                u_grid_n=QP_U_GRID_N,
                ridge=QP_RIDGE,
                eta=QP_ETA,
                tail_sigmas=(1.0,2.0,3.0),
                solver_primary=QP_SOLVER_PRIMARY,
                solver_fallback=QP_SOLVER_FALLBACK
            )

            # lam order: [1, x, x^2, |x|^4, tail1, tail2, tail3]
            lam1, lam2, lam3 = float(lam[4]), float(lam[5]), float(lam[6])

            # stronger tail pressure score (robust to sign flips)
            qp_score = abs(lam1) + abs(lam2) + 2.0*abs(lam3)

        except Exception:
            qp_score = np.nan

    H_vals.append(H); TM_vals.append(S_tail); VM_vals.append(S_var); SM_vals.append(S_skew); Tq_vals.append(Tq)
    QP_TAILSCORE_vals.append(qp_score)
    dates.append(t)

# Series
H_ser  = pd.Series(H_vals, index=dates, name="Shannon")
TM_ser = pd.Series(TM_vals, index=dates, name="TailMass")
VM_ser = pd.Series(VM_vals, index=dates, name="VarMass")
SM_ser = pd.Series(SM_vals, index=dates, name="SkewMass")
Tq_ser = pd.Series(Tq_vals, index=dates, name=f"Tsallis_q{Q_TSALLIS}")
QP_ser = pd.Series(QP_TAILSCORE_vals, index=dates, name="QP_TailPressure")

rv_ser = rv_21.loc[pd.Index(dates)]
train_mask = rv_ser.index.year <= TRAIN_END_YR
test_mask  = rv_ser.index.year == TEST_YR
train_index = rv_ser.index[train_mask]

# Labels
future_max5 = rv_ser.rolling(5, min_periods=1).max().shift(-1)
for pct in range(90, 49, -5):
    thr = np.percentile(rv_ser.loc[train_mask], pct)
    labs = (future_max5 > thr).astype(int)
    y_tmp = safe_series(labs.loc[test_mask].dropna())
    counts = y_tmp.value_counts()
    pos, neg = int(counts.get(1,0)), int(counts.get(0,0))
    if pos>0 and neg>0:
        labels = safe_series(labs)
        print(f"Using {pct}th percentile (thr={thr:.4f}) → pos={pos}, neg={neg}")
        break
else:
    raise ValueError("Single-class held-out even at 50th percentile.")

y_test = safe_series(labels.loc[test_mask].dropna())

# Z-scores (train-only)
H_z  = z_from_index(H_ser,  train_index)
TM_z = z_from_index(TM_ser, train_index)
VM_z = z_from_index(VM_ser, train_index)
SM_z = z_from_index(SM_ser, train_index)
Tq_z = z_from_index(Tq_ser, train_index)

qp_train_idx = QP_ser.loc[train_index].dropna().index
QP_z = z_from_index(QP_ser, qp_train_idx)

# Choose sign on train-only for QP score (though abs() makes it usually monotone)
sign_QP = choose_sign_on_index(QP_z, labels, qp_train_idx) if len(qp_train_idx) else +1.0

scores = {
    "Shannon+TailMass": -TM_z,
    "Shannon+VarMass":  -VM_z,
    "Shannon+SkewMass": -SM_z,
    f"Tsallis_q{Q_TSALLIS}": -Tq_z,
    "Shannon":             -H_z,
    "Generative": sign_QP * QP_z
}

# Metrics
metrics = {}
for name, sc in scores.items():
    sc_test = sc.loc[y_test.index].dropna()
    y_aligned = safe_series(y_test.loc[sc_test.index])
    if int(y_aligned.nunique()) < 2:
        continue
    roc = roc_auc_score(y_aligned, sc_test)
    pr  = average_precision_score(y_aligned, sc_test)
    prec, rec, _ = precision_recall_curve(y_aligned, sc_test)
    f1_max = np.nanmax(2*prec*rec/np.maximum(prec+rec, 1e-20))
    metrics[name] = (roc, pr, f1_max)

print("\nHeld-out 2024 metrics (ROC-AUC, PR-AUC, max F₁):")
for name, (r,p,f) in sorted(metrics.items(), key=lambda kv: kv[1][0], reverse=True):
    print(f"  {name:26s}: {r:.3f}, {p:.3f}, {f:.3f}")

# ROC plot
plt.figure(figsize=(7,6))
for name, (r,p,f) in sorted(metrics.items(), key=lambda kv: kv[1][0], reverse=True):
    sc = scores[name].loc[y_test.index].dropna()
    y_aligned = safe_series(y_test.loc[sc.index])
    fpr, tpr, _ = roc_curve(y_aligned, sc)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={r:.3f})")
plt.plot([0,1],[0,1],'k--',alpha=0.6)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Held-out ROC — Next-5-day RV Spike (2024)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# PR plot
plt.figure(figsize=(7,6))
baseline = float(np.mean(y_test.to_numpy()))
for name, (r,p,f) in sorted(metrics.items(), key=lambda kv: kv[1][1], reverse=True):
    sc = scores[name].loc[y_test.index].dropna()
    y_aligned = safe_series(y_test.loc[sc.index])
    prec, rec, _ = precision_recall_curve(y_aligned, sc)
    plt.plot(rec, prec, lw=2, label=f"{name} (AP={p:.3f})")
plt.plot([0,1],[baseline, baseline], 'k--', alpha=0.6, label=f"Chance (AP={baseline:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curves (held-out 2024)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
