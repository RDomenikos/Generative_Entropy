#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# QM_Solving_SP500_ROC_2012_2024_GenerativeExcessEnergy_parallelQP_STANDARDS.py
#
# IMPORTANT:
#   - Generative/QM (QP + ΔE excess-energy) is UNCHANGED.
#   - Only the comparison set is standards:
#       Generative, Shannon, Tsallis q2/q3, Renyi, Sharma-Mittal
#
# NEW OUTPUTS (requested):
#   - daily_scores_YYYY.csv  (per plot-year, per method: date, y, method, score)
#   - ROC_pooled_2012_2024.png
#   - PR_pooled_2012_2024.png
#   - AUC_by_year_with_CI_blockbootstrap.png
#   - AUC_summary_by_year_with_CI.csv
#   - AUC_pooled_summary.csv
# ---------------------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress common CVXPY/SCS warnings (optional)
SUPPRESS_CVX_WARNINGS = True
SUPPRESS_SCS_WARNINGS = True
if SUPPRESS_CVX_WARNINGS:
    warnings.filterwarnings("ignore", message="Solution may be inaccurate.*", category=UserWarning)
if SUPPRESS_SCS_WARNINGS:
    warnings.filterwarnings("ignore", message="Converting .* to a CSC.*", category=UserWarning)

import numpy as np
import pandas as pd
import datetime as dt

import matplotlib
matplotlib.use("Agg")  # multiprocessing-safe on macOS
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# ------------------ CONFIG --------------------------------------------------
TICKER       = "^GSPC"
START_YR     = 2000

# Baseline rolling window (standards on rolling hist)
WINDOW_DAYS  = 21
BINS_ROLL    = 50
EPS          = 1e-16

# Label horizon
FUTURE_HORIZON_DAYS = 5

# Years to plot
PLOT_YEARS   = list(range(2012, 2025))  # 2012..2024 inclusive

# Split definition:
# Train: [Y-12 .. Y-2], Calib: Y-1, Plot: Y
TRAIN_YEARS_BACK = 12
TRAIN_END_LAG    = 2

# ---------- STANDARD ENTROPIES (requested) ----------
TSALLIS_QS = (2, 3)
RENYI_ALPHA = 2.0

# Sharma–Mittal parameters (single setting; keep fixed)
# H_SM(r,q) = ( (sum p^q)^((1-r)/(1-q)) - 1 ) / (1 - r)
SM_R = 1.7
SM_Q = 3.0

# ---------- QP / Generative settings (UNCHANGED) ----------
QP_YR_START  = 2000

# Longer window for QP stability
QP_WINDOW_DAYS       = 63

# Fixed histogram edges in standardized coordinates
QP_BINS      = 41
QP_RANGE_SIGMA_FIXED = 8.0
QP_PSEUDOCOUNT = 1e-2

# Stage-I core degree in p
QP_DEG         = 3
QP_U_GRID_N    = 250
QP_ETA         = 1e-6

# Regularization (stability)
QP_RIDGE_A     = 1e-4
QP_RIDGE_LAM   = 1e-3

# Residual weighting in QP fit
QP_WEIGHTING   = "inv_sqrt_p"  # recommended

# Tail features
TAIL_SIGMAS = (1.0, 2.0, 3.0)
USE_SMOOTH_TAIL_FEATURES = True
TAIL_SMOOTH_WIDTH = 0.25  # in standardized units (sigma_ref=1)

# Generative energy weights (fixed a priori)
# Order: [1, x, x^2, |x|^4, tail1, tail2, tail3]
# weight on x set to 0 (drift nuisance)
APPLY_SYMMETRY_W = True

# Optional smoothing of ΔE series before z-scoring
APPLY_DEGEN_EMA  = True
DEGEN_EMA_SPAN   = 5

QP_SOLVER_PRIMARY  = "OSQP"
QP_SOLVER_FALLBACK = "SCS"

# Output
OUTDIR = "Yearl_with_Pooled_rolling_outputs_sp500_2012_2024_gen_excess_energy_STANDARDS"
SAVE_PR = True

# Threshold selection percentiles (train-only thresholds)
PCT_GRID = list(range(95, 24, -5))  # 95,90,...,25
MIN_POS_CALIB = 1
MIN_POS_PLOT  = 1

# Parallelism
YEAR_WORKERS = max(1, (os.cpu_count() or 2) - 1)
QP_WORKERS   = min(max(1, (os.cpu_count() or 2) - 2), 24)

# ---- Consistent plot ordering + colors (fixed across years) ----
METHOD_ORDER = [
    "Generative (ΔE_gen)",
    "Shannon",
    "Tsallis_q2",
    "Tsallis_q3",
    f"Renyi_a{RENYI_ALPHA:g}",
    f"SharmaMittal_r{SM_R:g}_q{SM_Q:g}",
]
METHOD_COLOR = {
    "Generative (ΔE_gen)": "C0",
    "Shannon":             "C5",
    "Tsallis_q2":          "C4",
    "Tsallis_q3":          "C3",
    f"Renyi_a{RENYI_ALPHA:g}": "C2",
    f"SharmaMittal_r{SM_R:g}_q{SM_Q:g}": "C1",
}

# ---------- NEW (requested): pooled ROC + per-year AUC CI ----------
MAKE_POOLED_PLOTS = True
BOOTSTRAP_N = 500          # 500–2000 is typical; keep moderate for speed
BOOTSTRAP_BLOCK = 5        # block length in trading days
BOOTSTRAP_SEED = 123
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


# ------------------ STANDARD ENTROPIES (requested) --------------------------
def shannon_entropy(p: np.ndarray, eps: float = 1e-16) -> float:
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-np.sum(p * np.log(p + eps)))

def tsallis_entropy(p: np.ndarray, q: float) -> float:
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    if abs(q - 1.0) < 1e-12:
        return shannon_entropy(p, eps=EPS)
    return float((1.0 - np.sum(p**q)) / (q - 1.0))

def renyi_entropy(p: np.ndarray, alpha: float, eps: float = 1e-16) -> float:
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    if abs(alpha - 1.0) < 1e-12:
        return shannon_entropy(p, eps=eps)
    return float(np.log(np.sum(p**alpha) + eps) / (1.0 - alpha))

def sharma_mittal_entropy(p: np.ndarray, r: float, q: float, eps: float = 1e-16) -> float:
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    if abs(r - 1.0) < 1e-10:
        return tsallis_entropy(p, q=q)
    if abs(q - 1.0) < 1e-10:
        return shannon_entropy(p, eps=eps)
    S = float(np.sum(p**q))
    expo = (1.0 - r) / (1.0 - q)
    val = (S + eps)**expo
    return float((val - 1.0) / (1.0 - r))


# ------------------ QP / Generative helpers (UNCHANGED) ---------------------
def tail_feat_sigmoid(abs_x: np.ndarray, k_sigma: float, width_sigma: float) -> np.ndarray:
    w = max(width_sigma, 1e-12)
    z = (abs_x - k_sigma) / w
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def build_tail_features_on_centers(
    x_centers_std: np.ndarray,
    tail_sigmas=(1.0, 2.0, 3.0),
    use_smooth=True,
    smooth_width=0.25
) -> np.ndarray:
    absx = np.abs(x_centers_std)
    cols = []
    if use_smooth:
        for k in tail_sigmas:
            cols.append(tail_feat_sigmoid(absx, k_sigma=k, width_sigma=smooth_width))
    else:
        for k in tail_sigmas:
            cols.append((absx > k).astype(float))
    return np.column_stack(cols).astype(float)


def solve_stage1_qp_shannon_core(
    p_hat, x_centers_std,
    deg=3, u_grid_n=250,
    ridge_a=1e-4, ridge_lam=1e-3,
    eta=1e-6, tail_sigmas=(1.0, 2.0, 3.0),
    weighting="inv_sqrt_p",
    use_smooth_tail=True,
    tail_smooth_width=0.25,
    solver_primary="OSQP", solver_fallback="SCS",
    tail_weight_alpha=5.0
):
    import cvxpy as cp
    import numpy as np

    p_hat = np.asarray(p_hat, float)
    x_centers_std = np.asarray(x_centers_std, float)

    nz = p_hat > 0
    p = p_hat[nz]
    x = x_centers_std[nz]
    absx = np.abs(x)

    feats = [np.ones_like(x), x, x**2, absx**4]
    if use_smooth_tail:
        for k in tail_sigmas:
            feats.append(tail_feat_sigmoid(absx, k_sigma=k, width_sigma=tail_smooth_width))
    else:
        for k in tail_sigmas:
            feats.append((absx > k).astype(float))
    F = np.column_stack(feats)

    a = cp.Variable(deg)
    lam = cp.Variable(F.shape[1])

    U = np.column_stack([p**m for m in range(1, deg + 1)])
    g_p = 1.0 + cp.log(p) + U @ a

    if weighting == "sqrt_p":
        w = np.sqrt(np.maximum(p, 1e-12))
    elif weighting == "ones":
        w = np.ones_like(p)
    elif weighting == "inv_sqrt_p":
        w = 1.0 / np.sqrt(np.maximum(p, 1e-12))
    else:
        raise ValueError("Unknown weighting")

    if tail_weight_alpha and tail_weight_alpha > 0:
        if use_smooth_tail:
            tail2 = tail_feat_sigmoid(absx, k_sigma=2.0, width_sigma=tail_smooth_width)
            tail3 = tail_feat_sigmoid(absx, k_sigma=3.0, width_sigma=tail_smooth_width)
        else:
            tail2 = (absx > 2.0).astype(float)
            tail3 = (absx > 3.0).astype(float)
        tail_boost = 0.5 * tail2 + 1.0 * tail3
        w = w * (1.0 + float(tail_weight_alpha) * tail_boost)

    r = g_p - F @ lam
    obj = (
        cp.sum_squares(cp.multiply(w, r))
        + ridge_a * cp.sum_squares(a)
        + ridge_lam * cp.sum_squares(lam)
    )

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

    if (not solved) or (prob.status not in ("optimal", "optimal_inaccurate")):
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=80000, eps=1e-5)
        except Exception:
            pass

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP failed: {prob.status}")

    return np.array(lam.value).ravel()


def _chunk_list(lst, n_chunks):
    n_chunks = max(1, int(n_chunks))
    chunks = [lst[k::n_chunks] for k in range(n_chunks)]
    return [c for c in chunks if len(c) > 0]


def _qp_worker_lam_mu(args):
    (i_chunk, rvals, qp_edges_std, qp_centers_std,
     sigma_ref, qp_window_days, qp_pseudocount,
     qp_deg, qp_u_grid_n, qp_ridge_a, qp_ridge_lam, qp_eta,
     tail_sigmas, use_smooth_tail, tail_smooth_width,
     weighting, solver_primary, solver_fallback) = args

    Phi_tail = build_tail_features_on_centers(
        qp_centers_std,
        tail_sigmas=tail_sigmas,
        use_smooth=use_smooth_tail,
        smooth_width=tail_smooth_width
    )

    x = qp_centers_std.astype(float)
    absx = np.abs(x)

    out = []
    lam_dim = 4 + len(tail_sigmas)

    for i in i_chunk:
        try:
            window = rvals[i - qp_window_days:i]
            window_std = window / max(sigma_ref, 1e-12)

            c_w, _ = np.histogram(window_std, bins=qp_edges_std)
            c_w = c_w.astype(float) + qp_pseudocount
            p_w = c_w / c_w.sum()

            lam = solve_stage1_qp_shannon_core(
                p_w, qp_centers_std,
                deg=qp_deg,
                u_grid_n=qp_u_grid_n,
                ridge_a=qp_ridge_a,
                ridge_lam=qp_ridge_lam,
                eta=qp_eta,
                tail_sigmas=tail_sigmas,
                weighting=weighting,
                use_smooth_tail=use_smooth_tail,
                tail_smooth_width=tail_smooth_width,
                solver_primary=solver_primary,
                solver_fallback=solver_fallback
            )

            mu0 = 1.0
            mux = float(np.sum(p_w * x))
            mux2 = float(np.sum(p_w * (x**2)))
            muabs4 = float(np.sum(p_w * (absx**4)))
            mutail = (p_w.reshape(-1, 1) * Phi_tail).sum(axis=0).astype(float)

            mu = np.concatenate([[mu0, mux, mux2, muabs4], mutail], axis=0)

            out.append((i, *lam.tolist(), *mu.tolist()))
        except Exception:
            out.append((i, *([np.nan] * lam_dim), *([np.nan] * lam_dim)))

    return out


def compute_all_features_parallel_qp(returns: pd.Series):
    rvals = returns.values
    idxs = returns.index

    max_train_end = max(PLOT_YEARS) - TRAIN_END_LAG
    ref_mask = (returns.index.year >= START_YR) & (returns.index.year <= max_train_end)
    r_ref = returns.loc[ref_mask].values
    sigma_ref = float(np.std(r_ref, ddof=1)) if len(r_ref) > 50 else float(np.std(rvals, ddof=1))

    L = QP_RANGE_SIGMA_FIXED
    qp_edges_std = np.linspace(-L, L, QP_BINS + 1)
    qp_centers_std = qp_edges_std[:-1] + np.diff(qp_edges_std) / 2

    lam_dim = 4 + len(TAIL_SIGMAS)

    H_vals = []
    T2_vals, T3_vals = [], []
    R_vals, SM_vals = [], []
    dates, i_list = [], []

    for i in range(WINDOW_DAYS, len(rvals) + 1):
        window = rvals[i - WINDOW_DAYS:i]
        t = idxs[i - 1]

        counts, _edges = np.histogram(window, bins=BINS_ROLL)
        tot = counts.sum()
        if tot <= 0:
            continue

        p_all = (counts / tot).astype(float)

        H  = shannon_entropy(p_all, eps=EPS)
        T2 = tsallis_entropy(p_all, q=2.0)
        T3 = tsallis_entropy(p_all, q=3.0)
        R  = renyi_entropy(p_all, alpha=RENYI_ALPHA, eps=EPS)
        SM = sharma_mittal_entropy(p_all, r=SM_R, q=SM_Q, eps=EPS)

        H_vals.append(H)
        T2_vals.append(T2)
        T3_vals.append(T3)
        R_vals.append(R)
        SM_vals.append(SM)

        dates.append(t)
        i_list.append(i)

    qp_i_list = [
        i for i in i_list
        if (i >= QP_WINDOW_DAYS) and (idxs[i - 1].year >= QP_YR_START)
    ]

    chunks = _chunk_list(qp_i_list, QP_WORKERS * 2)
    print(f"[i] QP parallel: {len(qp_i_list)} endpoints → {len(chunks)} chunks using {QP_WORKERS} workers")

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")

    lam_map = {}
    mu_map = {}

    worker_args = []
    for ch in chunks:
        worker_args.append((
            ch,
            rvals,
            qp_edges_std,
            qp_centers_std,
            sigma_ref,
            QP_WINDOW_DAYS,
            QP_PSEUDOCOUNT,
            QP_DEG,
            QP_U_GRID_N,
            QP_RIDGE_A,
            QP_RIDGE_LAM,
            QP_ETA,
            TAIL_SIGMAS,
            USE_SMOOTH_TAIL_FEATURES,
            TAIL_SMOOTH_WIDTH,
            QP_WEIGHTING,
            QP_SOLVER_PRIMARY,
            QP_SOLVER_FALLBACK
        ))

    with ProcessPoolExecutor(max_workers=QP_WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(_qp_worker_lam_mu, a) for a in worker_args]
        for fut in as_completed(futs):
            for rec in fut.result():
                i = rec[0]
                lam = np.array(rec[1:1 + lam_dim], dtype=float)
                mu  = np.array(rec[1 + lam_dim:1 + 2 * lam_dim], dtype=float)
                lam_map[i] = lam
                mu_map[i] = mu

    lam_mat = np.vstack([lam_map.get(i, np.full(lam_dim, np.nan)) for i in i_list])
    mu_mat  = np.vstack([mu_map.get(i,  np.full(lam_dim, np.nan)) for i in i_list])

    lam_cols = ["lam_1", "lam_x", "lam_x2", "lam_absx4"] + [f"lam_tail_{k}σ" for k in TAIL_SIGMAS]
    mu_cols  = ["mu_1", "mu_x", "mu_x2", "mu_absx4"] + [f"mu_tail_{k}σ" for k in TAIL_SIGMAS]

    lam_df = pd.DataFrame(lam_mat, index=dates, columns=lam_cols)
    mu_df  = pd.DataFrame(mu_mat,  index=dates, columns=mu_cols)

    return {
        "H_ser":  pd.Series(H_vals,  index=dates, name="Shannon"),
        "T2_ser": pd.Series(T2_vals, index=dates, name="Tsallis_q2"),
        "T3_ser": pd.Series(T3_vals, index=dates, name="Tsallis_q3"),
        "R_ser":  pd.Series(R_vals,  index=dates, name=f"Renyi_a{RENYI_ALPHA:g}"),
        "SMstd_ser": pd.Series(SM_vals, index=dates, name=f"SharmaMittal_r{SM_R:g}_q{SM_Q:g}"),
        "lam_df": lam_df,
        "mu_df":  mu_df,
        "sigma_ref": sigma_ref
    }


def pick_percentile_threshold_trainonly_require_eval(
    rv_ser: pd.Series,
    future_max: pd.Series,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    plot_mask: pd.Series,
    percentiles,
    min_pos_calib=1,
    min_pos_plot=1
):
    train_rv = rv_ser.loc[train_mask].dropna()
    if len(train_rv) < 50:
        thr = np.nan
        labs = (future_max > thr).astype(int)
        return None, thr, safe_series(labs), {"status": "insufficient_train"}

    for pct in percentiles:
        thr = float(np.percentile(train_rv, pct))
        labs = (future_max > thr).astype(int)

        y_cal  = safe_series(labs.loc[calib_mask].dropna())
        y_plot = safe_series(labs.loc[plot_mask].dropna())

        if y_cal.nunique() < 2 or y_plot.nunique() < 2:
            continue

        pos_cal = int((y_cal == 1).sum()); neg_cal = int((y_cal == 0).sum())
        pos_pl  = int((y_plot == 1).sum()); neg_pl  = int((y_plot == 0).sum())

        if pos_cal >= min_pos_calib and neg_cal >= 1 and pos_pl >= min_pos_plot and neg_pl >= 1:
            return pct, thr, safe_series(labs), {
                "status": "ok",
                "pos_cal": pos_cal, "neg_cal": neg_cal,
                "pos_plot": pos_pl, "neg_plot": neg_pl
            }

    pct = min(percentiles)
    thr = float(np.percentile(train_rv, pct))
    labs = (future_max > thr).astype(int)
    y_cal  = safe_series(labs.loc[calib_mask].dropna())
    y_plot = safe_series(labs.loc[plot_mask].dropna())
    return pct, thr, safe_series(labs), {
        "status": "fallback",
        "pos_cal": int((y_cal == 1).sum()), "neg_cal": int((y_cal == 0).sum()),
        "pos_plot": int((y_plot == 1).sum()), "neg_plot": int((y_plot == 0).sum())
    }


def save_placeholder_roc(plot_year, train_start, train_end, calib_year, pct, thr, pos, neg, outdir, reason):
    roc_path = os.path.join(outdir, f"ROC_plot_{plot_year}.png")
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.title(
        f"ROC — Plot {plot_year} | Train {train_start}-{train_end} | Calib {calib_year}\n"
        f"UNDEFINED ({reason}): pos={pos}, neg={neg}\n"
        f"thr={pct}th pct of train RV (thr={thr})"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()


def save_placeholder_pr(plot_year, train_start, train_end, calib_year, pct, thr, pos, neg, baseline, outdir, reason):
    pr_path = os.path.join(outdir, f"PR_plot_{plot_year}.png")
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [baseline, baseline], "k--", alpha=0.6)
    plt.title(
        f"PR — Plot {plot_year} | Train {train_start}-{train_end} | Calib {calib_year}\n"
        f"UNDEFINED ({reason}): pos={pos}, neg={neg}\n"
        f"thr={pct}th pct of train RV (thr={thr})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()


# ------------------ NEW: bootstrap helpers (requested) ----------------------
def _block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, int(block))
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = []
    for s in starts:
        idx.extend([(s + k) % n for k in range(block)])
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def block_bootstrap_auc(y: np.ndarray, s: np.ndarray, nboot: int, block: int, seed: int):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]; s = s[m]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    aucs = []
    for b in range(int(nboot)):
        idx = _block_bootstrap_indices(len(y), block, rng)
        yy = y[idx]; ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yy, ss))
        except Exception:
            continue

    if len(aucs) < 30:
        base = roc_auc_score(y, s)
        return base, np.nan, np.nan

    aucs = np.array(aucs, dtype=float)
    base = roc_auc_score(y, s)
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return float(base), lo, hi


def run_one_plot_year(plot_year: int, feats: dict, rv_ser: pd.Series, outdir: str, gen_w: np.ndarray):
    train_start = plot_year - TRAIN_YEARS_BACK
    train_end   = plot_year - TRAIN_END_LAG
    calib_year  = plot_year - 1

    idx = rv_ser.index
    train_mask = (idx.year >= train_start) & (idx.year <= train_end)
    calib_mask = (idx.year == calib_year)
    plot_mask  = (idx.year == plot_year)

    train_index = idx[train_mask]
    if len(train_index) < 50:
        save_placeholder_roc(plot_year, train_start, train_end, calib_year, "NA", "NA", 0, 0, outdir, "insufficient_train")
        if SAVE_PR:
            save_placeholder_pr(plot_year, train_start, train_end, calib_year, "NA", "NA", 0, 0, 0.0, outdir, "insufficient_train")
        return {"plot_year": plot_year, "note": "insufficient_train"}

    future_max = rv_ser.rolling(FUTURE_HORIZON_DAYS, min_periods=1).max().shift(-1)
    pct, thr, labels, diag = pick_percentile_threshold_trainonly_require_eval(
        rv_ser, future_max,
        train_mask=train_mask,
        calib_mask=calib_mask,
        plot_mask=plot_mask,
        percentiles=PCT_GRID,
        min_pos_calib=MIN_POS_CALIB,
        min_pos_plot=MIN_POS_PLOT
    )

    y_plot = safe_series(labels.loc[plot_mask].dropna())
    pos = int((y_plot == 1).sum())
    neg = int((y_plot == 0).sum())

    if y_plot.nunique() < 2:
        save_placeholder_roc(plot_year, train_start, train_end, calib_year, pct, thr, pos, neg, outdir, "single_class_plot_year")
        if SAVE_PR:
            baseline = float(np.mean(y_plot.to_numpy())) if len(y_plot) else 0.0
            save_placeholder_pr(plot_year, train_start, train_end, calib_year, pct, thr, pos, neg, baseline, outdir, "single_class_plot_year")
        return {
            "plot_year": plot_year,
            "train_start": train_start,
            "train_end": train_end,
            "calib_year": calib_year,
            "pct": pct, "thr": thr,
            "pos": pos, "neg": neg,
            "note": "single_class_plot_year"
        }

    # ----------- Standard entropies z-scores (train-only) -----------
    H_z  = z_from_index(feats["H_ser"], train_index)
    T2_z = z_from_index(feats["T2_ser"], train_index)
    T3_z = z_from_index(feats["T3_ser"], train_index)
    R_z  = z_from_index(feats["R_ser"],  train_index)
    SM_z = z_from_index(feats["SMstd_ser"], train_index)

    # ----------- Generative (UNCHANGED) -----------
    lam_df = feats["lam_df"]
    mu_df  = feats["mu_df"]

    mu_ref = mu_df.loc[train_index].mean(axis=0)
    dmu = mu_df.sub(mu_ref, axis=1)

    cols = lam_df.columns.tolist()
    lam_mat = lam_df[cols]

    dmu_mat = dmu[[c.replace("lam_", "mu_") for c in cols]]
    dmu_mat.columns = cols

    dE = (lam_mat.mul(dmu_mat).mul(gen_w, axis=1)).sum(axis=1)
    dE.name = "DeltaE_gen"

    if APPLY_DEGEN_EMA:
        dE = dE.ewm(span=DEGEN_EMA_SPAN, adjust=False, min_periods=1).mean()
        dE.name = f"DeltaE_gen_EMA{DEGEN_EMA_SPAN}"

    dE_train_idx = dE.loc[train_index].dropna().index
    dE_z = z_from_index(dE, dE_train_idx)

    sign_E = choose_sign_on_index(dE_z, labels, dE_train_idx) if len(dE_train_idx) else +1.0

    sign_H  = choose_sign_on_index(-H_z,  labels, train_index)
    sign_T2 = choose_sign_on_index(-T2_z, labels, train_index)
    sign_T3 = choose_sign_on_index(-T3_z, labels, train_index)
    sign_R  = choose_sign_on_index(-R_z,  labels, train_index)
    sign_SM = choose_sign_on_index(-SM_z, labels, train_index)

    scores = {
        "Shannon": sign_H * (-H_z),
        "Tsallis_q2": sign_T2 * (-T2_z),
        "Tsallis_q3": sign_T3 * (-T3_z),
        f"Renyi_a{RENYI_ALPHA:g}": sign_R * (-R_z),
        f"SharmaMittal_r{SM_R:g}_q{SM_Q:g}": sign_SM * (-SM_z),
        "Generative (ΔE_gen)": sign_E * dE_z
    }

    metrics = {}
    for name, sc in scores.items():
        sc_y = sc.loc[y_plot.index].dropna()
        y_aligned = safe_series(y_plot.loc[sc_y.index])
        if int(y_aligned.nunique()) < 2:
            continue
        roc = roc_auc_score(y_aligned, sc_y)
        pr  = average_precision_score(y_aligned, sc_y)
        prec, rec, _ = precision_recall_curve(y_aligned, sc_y)
        f1_max = np.nanmax(2 * prec * rec / np.maximum(prec + rec, 1e-20))
        metrics[name] = (roc, pr, f1_max)

    # ---------- NEW: save per-day scores for pooled analyses ----------
    daily_rows = []
    for name in METHOD_ORDER:
        if name not in scores:
            continue
        sc = scores[name].loc[y_plot.index].dropna()
        y_aligned = safe_series(y_plot.loc[sc.index])
        if y_aligned.nunique() < 2:
            continue
        for d, yy, ss in zip(sc.index, y_aligned.values, sc.values):
            daily_rows.append((d, int(yy), name, float(ss)))

    if len(daily_rows) > 0:
        daily_df = pd.DataFrame(daily_rows, columns=["date", "y", "method", "score"])
        daily_df_path = os.path.join(outdir, f"daily_scores_{plot_year}.csv")
        daily_df.to_csv(daily_df_path, index=False)

    # ROC plot (per-year)
    roc_path = os.path.join(outdir, f"ROC_plot_{plot_year}.png")
    plt.figure(figsize=(7, 6))
    for name in METHOD_ORDER:
        if name not in metrics:
            continue
        r, p, f1 = metrics[name]
        sc = scores[name].loc[y_plot.index].dropna()
        y_aligned = safe_series(y_plot.loc[sc.index])
        fpr, tpr, _ = roc_curve(y_aligned, sc)
        plt.plot(fpr, tpr, lw=2, color=METHOD_COLOR.get(name, None),
                 label=f"{name} (AUC={r:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC — Plot {plot_year} | Train {train_start}-{train_end} | Calib {calib_year}\n"
        f"thr={pct}th pct of train RV (thr={thr:.6g}) | "
        f"cal(pos={diag.get('pos_cal')},neg={diag.get('neg_cal')}), "
        f"plot(pos={diag.get('pos_plot')},neg={diag.get('neg_plot')}) "
        f"[{diag.get('status')}]"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # PR plot (per-year)
    if SAVE_PR:
        pr_path = os.path.join(outdir, f"PR_plot_{plot_year}.png")
        plt.figure(figsize=(7, 6))
        baseline = float(np.mean(y_plot.to_numpy()))
        for name in METHOD_ORDER:
            if name not in metrics:
                continue
            r, p, f1 = metrics[name]
            sc = scores[name].loc[y_plot.index].dropna()
            y_aligned = safe_series(y_plot.loc[sc.index])
            prec, rec, _ = precision_recall_curve(y_aligned, sc)
            plt.plot(rec, prec, lw=2, color=METHOD_COLOR.get(name, None),
                     label=f"{name} (AP={p:.3f})")

        plt.plot([0, 1], [baseline, baseline], "k--", alpha=0.6,
                 label=f"Chance (AP={baseline:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"PR — Plot {plot_year} | Train {train_start}-{train_end} | Calib {calib_year}\n"
            f"thr={pct}th pct of train RV (thr={thr:.6g}) | "
            f"cal(pos={diag.get('pos_cal')},neg={diag.get('neg_cal')}), "
            f"plot(pos={diag.get('pos_plot')},neg={diag.get('neg_plot')}) "
            f"[{diag.get('status')}]"
        )
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_path, dpi=200)
        plt.close()

    row = {
        "plot_year": plot_year,
        "train_start": train_start,
        "train_end": train_end,
        "calib_year": calib_year,
        "pct": pct,
        "thr": thr,
        "pos": pos,
        "neg": neg,
        "threshold_status": diag.get("status", ""),
        "pos_cal": diag.get("pos_cal", np.nan),
        "neg_cal": diag.get("neg_cal", np.nan),
        "pos_plot": diag.get("pos_plot", np.nan),
        "neg_plot": diag.get("neg_plot", np.nan),
        "gen_sign": float(sign_E),
    }
    for name, (roc, pr, f1) in metrics.items():
        key = (
            name.replace(" ", "_")
                .replace("+", "_plus_")
                .replace("-", "_")
                .replace("(", "")
                .replace(")", "")
        )
        row[f"{key}__roc"] = roc
        row[f"{key}__pr"] = pr
        row[f"{key}__f1max"] = f1
    return row


# ------------------ NEW: pooled + per-year AUC(CI) plots --------------------
def build_pooled_outputs(outdir: str, years: list[int]):
    all_paths = [os.path.join(outdir, f"daily_scores_{y}.csv") for y in years]
    all_paths = [p for p in all_paths if os.path.exists(p)]
    if len(all_paths) == 0:
        print("[warn] No daily_scores_YYYY.csv files found; skipping pooled outputs.")
        return

    df = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in all_paths], axis=0, ignore_index=True)
    df = df.dropna(subset=["y", "score", "method"])
    df["y"] = df["y"].astype(int)

    # --- Pooled ROC/PR per method ---
    pooled_summary = []
    roc_fig = os.path.join(outdir, f"ROC_pooled_{years[0]}_{years[-1]}.png")
    plt.figure(figsize=(7, 6))
    for m in METHOD_ORDER:
        d = df[df["method"] == m]
        if len(d) < 50 or d["y"].nunique() < 2:
            continue
        y = d["y"].values
        s = d["score"].values
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, lw=2, color=METHOD_COLOR.get(m, None), label=f"{m} (AUC={auc:.3f})")
        pooled_summary.append({"method": m, "pooled_auc": float(auc), "n": int(len(d)), "pos": int((y == 1).sum()), "neg": int((y == 0).sum())})
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled ROC (out-of-sample) — {years[0]}–{years[-1]}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_fig, dpi=250)
    plt.close()

    if SAVE_PR:
        pr_fig = os.path.join(outdir, f"PR_pooled_{years[0]}_{years[-1]}.png")
        plt.figure(figsize=(7, 6))
        # pooled baseline = pooled positive rate (method-independent)
        y_all = df.drop_duplicates(subset=["date", "y"])[["date", "y"]]["y"].values
        baseline = float(np.mean(y_all)) if len(y_all) else 0.0
        for m in METHOD_ORDER:
            d = df[df["method"] == m]
            if len(d) < 50 or d["y"].nunique() < 2:
                continue
            y = d["y"].values
            s = d["score"].values
            ap = average_precision_score(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            plt.plot(rec, prec, lw=2, color=METHOD_COLOR.get(m, None), label=f"{m} (AP={ap:.3f})")
        plt.plot([0, 1], [baseline, baseline], "k--", alpha=0.6, label=f"Chance (AP={baseline:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Pooled PR (out-of-sample) — {years[0]}–{years[-1]}")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_fig, dpi=250)
        plt.close()

    pooled_df = pd.DataFrame(pooled_summary)
    pooled_csv = os.path.join(outdir, "AUC_pooled_summary.csv")
    pooled_df.to_csv(pooled_csv, index=False)

    # --- Per-year AUC with block bootstrap CI ---
    rows = []
    for y in years:
        dY = df[df["date"].dt.year == y]
        if len(dY) == 0:
            continue
        for m in METHOD_ORDER:
            dm = dY[dY["method"] == m]
            if len(dm) < 30 or dm["y"].nunique() < 2:
                rows.append({"plot_year": y, "method": m, "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": int(len(dm))})
                continue
            yy = dm["y"].values
            ss = dm["score"].values
            auc, lo, hi = block_bootstrap_auc(yy, ss, nboot=BOOTSTRAP_N, block=BOOTSTRAP_BLOCK, seed=BOOTSTRAP_SEED + 97*y + hash(m) % 1000)
            rows.append({"plot_year": y, "method": m, "auc": auc, "ci_lo": lo, "ci_hi": hi, "n": int(len(dm))})

    auc_df = pd.DataFrame(rows)
    auc_csv = os.path.join(outdir, "AUC_summary_by_year_with_CI.csv")
    auc_df.to_csv(auc_csv, index=False)

    # plot (error bars)
    fig_path = os.path.join(outdir, "AUC_by_year_with_CI_blockbootstrap.png")
    plt.figure(figsize=(10, 6))
    for m in METHOD_ORDER:
        sub = auc_df[auc_df["method"] == m].sort_values("plot_year")
        xs = sub["plot_year"].values
        ys = sub["auc"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values

        # error bars only where CI exists
        yerr = None
        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            lower = ys - lo
            upper = hi - ys
            yerr = np.vstack([lower, upper])
        plt.errorbar(xs, ys, yerr=yerr, fmt="-o", capsize=3, lw=2, color=METHOD_COLOR.get(m, None), label=m)

    plt.axhline(0.5, ls="--", lw=1.5, alpha=0.7)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Plot year")
    plt.ylabel("AUC")
    plt.title(f"Per-year out-of-sample AUC with 95% CI (block bootstrap, block={BOOTSTRAP_BLOCK}, n={BOOTSTRAP_N})")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=250)
    plt.close()

    print(f"[done] Saved pooled ROC:\n  {roc_fig}")
    if SAVE_PR:
        print(f"[done] Saved pooled PR:\n  {os.path.join(outdir, f'PR_pooled_{years[0]}_{years[-1]}.png')}")
    print(f"[done] Saved pooled AUC summary:\n  {pooled_csv}")
    print(f"[done] Saved per-year AUC(CI) summary:\n  {auc_csv}")
    print(f"[done] Saved per-year AUC(CI) figure:\n  {fig_path}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[i] Output directory: {OUTDIR}")
    print(f"[i] Year plotting workers: {YEAR_WORKERS}")
    print(f"[i] QP workers: {QP_WORKERS}")
    print(f"[i] Plot years: {PLOT_YEARS[0]}..{PLOT_YEARS[-1]}")
    print(f"[i] Baseline window: {WINDOW_DAYS}d | QP window: {QP_WINDOW_DAYS}d")
    print(f"[i] Threshold percentiles tried (train-only): {PCT_GRID}")
    print(f"[i] Standards: Tsallis q={TSALLIS_QS}, Renyi a={RENYI_ALPHA}, Sharma–Mittal r={SM_R}, q={SM_Q}")
    print(f"[i] NEW: pooled ROC/PR + per-year AUC(CI): block={BOOTSTRAP_BLOCK}, nboot={BOOTSTRAP_N}")

    start = dt.datetime(START_YR, 1, 1)
    end   = dt.datetime.today()
    df = yf.download(
        TICKER,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    ).dropna()

    prices = safe_series(df["Close"])
    returns = safe_series(prices.pct_change().dropna())
    rv_21 = safe_series(returns.rolling(WINDOW_DAYS).var().dropna())

    print(f"[i] Downloaded {len(returns)} daily returns from {returns.index[0].date()} to {returns.index[-1].date()}")

    print("[i] Computing standards + QP lambda/mu series (QP parallel; standardized x) ...")
    feats = compute_all_features_parallel_qp(returns)
    print(f"[i] sigma_ref for standardization: {feats['sigma_ref']:.6g}")

    feat_dates = feats["H_ser"].index
    rv_ser = rv_21.loc[pd.Index(feat_dates)]

    cols = feats["lam_df"].columns.tolist()
    gen_w = np.ones(len(cols), dtype=float)
    if APPLY_SYMMETRY_W and "lam_x" in cols:
        gen_w[cols.index("lam_x")] = 0.0

    print("[i] Generative weights:")
    for c, w in zip(cols, gen_w.tolist()):
        print(f"    {c:12s}: {w:g}")

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")

    results = []
    with ProcessPoolExecutor(max_workers=YEAR_WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(run_one_plot_year, Y, feats, rv_ser, OUTDIR, gen_w) for Y in PLOT_YEARS]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"plot_year": None, "note": f"FAILED: {repr(e)}"})

    res_df = pd.DataFrame(results).sort_values("plot_year", na_position="last")
    csv_path = os.path.join(OUTDIR, f"rolling_summary_metrics_{PLOT_YEARS[0]}_{PLOT_YEARS[-1]}.csv")
    res_df.to_csv(csv_path, index=False)

    diag_path = os.path.join(OUTDIR, "diagnostics_lambda_mu.csv")
    pd.concat([feats["lam_df"], feats["mu_df"]], axis=1).to_csv(diag_path)

    print(f"[done] Saved summary CSV:\n  {csv_path}")
    print(f"[done] Saved diagnostics CSV:\n  {diag_path}")
    print(f"[done] Saved ROC/PR plots for every year {PLOT_YEARS[0]}..{PLOT_YEARS[-1]}.")

    # NEW: pooled ROC/PR + per-year AUC(CI)
    if MAKE_POOLED_PLOTS:
        print("[i] Building pooled ROC/PR and per-year AUC(CI) from daily_scores_YYYY.csv ...")
        build_pooled_outputs(OUTDIR, PLOT_YEARS)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
