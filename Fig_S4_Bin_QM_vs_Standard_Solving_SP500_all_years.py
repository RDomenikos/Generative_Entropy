#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# QM_SingleSplit_2016_2017_2018_BinSweep_ALIGNED_QP_GenerativeOnly.py
#
# PURPOSE (paper figure):
#   - Demonstrate discretization (bin) robustness for the GENERATIVE/QP method.
#   - We vary the number of histogram bins B used by the QP histogram grid.
#   - For each B, we re-solve the Stage-I QP on that grid and compute the
#     out-of-sample AUC on the test year (2018), with Train=2016, Calib=2017.
#
# KEY POINTS:
#   - QP formulation, constraints, solver settings, tail features, ΔE construction,
#     train-only z-scoring, train-only sign selection: unchanged.
#   - The ONLY change across the sweep is the discretization grid:
#       edges_std = linspace(-L, L, B+1), centers_std accordingly.
#   - Standard entropy baselines are NOT computed/ploted (to keep figure uncluttered).
#
# OUTPUTS:
#   - AUC_vs_bins_2016_2017_2018_alignedQP_GenerativeOnly.csv
#   - AUC_vs_bins_2016_2017_2018_alignedQP_GenerativeOnly.png
#   Optional (toggle): per-B ROC/PR plots and daily score CSVs.
# ---------------------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# =============================== CONFIG =====================================
TICKER   = "^GSPC"
START_YR = 2000

# Split (fixed)
TRAIN_YEAR = 2016
CALIB_YEAR = 2017
TEST_YEAR  = 2018

# RV rolling window
WINDOW_DAYS = 21
FUTURE_HORIZON_DAYS = 5

# Fixed standardized support for QP histograms in the sweep
L_STD = 8.0  # standardized coordinate support [-L_STD, L_STD]

# Bin sweep (include 41 to match your current QP_BINS if desired)
BINS_SWEEP = [10, 15, 20, 30, 41, 50, 75, 100, 150, 200]

# Threshold selection (train-only)
PCT_GRID = list(range(95, 24, -5))
MIN_POS_CALIB = 1
MIN_POS_TEST  = 1

# --- QP / Generative settings (UNCHANGED except discretization grid) ---
QP_YR_START       = 2000
QP_WINDOW_DAYS    = 63
QP_PSEUDOCOUNT    = 1e-2
QP_DEG            = 3
QP_U_GRID_N       = 250
QP_ETA            = 1e-6
QP_RIDGE_A        = 1e-4
QP_RIDGE_LAM      = 1e-3
QP_WEIGHTING      = "inv_sqrt_p"
TAIL_SIGMAS       = (1.0, 2.0, 3.0)
USE_SMOOTH_TAIL_FEATURES = True
TAIL_SMOOTH_WIDTH = 0.25
QP_SOLVER_PRIMARY  = "OSQP"
QP_SOLVER_FALLBACK = "SCS"

# ΔE smoothing
APPLY_DEGEN_EMA  = True
DEGEN_EMA_SPAN   = 5

# Symmetry weight: set lam_x weight to 0
APPLY_SYMMETRY_W = True

# Parallelism
QP_WORKERS = min(max(1, (os.cpu_count() or 2) ), 24)

# Outputs
OUTDIR = "SingleSplit_2016_2017_2018_ALIGNED_QP_binsweep_GenerativeOnly"

# Optional per-B artifacts (off by default to keep things light)
SAVE_PER_B_ROC_PR = False   # ROC/PR for each B (test year only)
SAVE_PER_B_DAILY  = False   # daily scores CSV for test year (each B)
# ===========================================================================

METHOD_NAME = "Generative (ΔE_gen)"
METHOD_COLOR = "C0"


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
    (i_chunk, rvals, edges_std, centers_std,
     sigma_ref, qp_window_days, qp_pseudocount,
     qp_deg, qp_u_grid_n, qp_ridge_a, qp_ridge_lam, qp_eta,
     tail_sigmas, use_smooth_tail, tail_smooth_width,
     weighting, solver_primary, solver_fallback) = args

    Phi_tail = build_tail_features_on_centers(
        centers_std,
        tail_sigmas=tail_sigmas,
        use_smooth=use_smooth_tail,
        smooth_width=tail_smooth_width
    )

    x = centers_std.astype(float)
    absx = np.abs(x)

    out = []
    lam_dim = 4 + len(tail_sigmas)

    for i in i_chunk:
        try:
            window = rvals[i - qp_window_days:i]
            window_std = window / max(sigma_ref, 1e-12)

            c_w, _ = np.histogram(window_std, bins=edges_std)
            c_w = c_w.astype(float) + qp_pseudocount
            p_w = c_w / c_w.sum()

            lam = solve_stage1_qp_shannon_core(
                p_w, centers_std,
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


def compute_sigma_ref_trainonly(returns: pd.Series) -> float:
    """
    Leakage-safe sigma_ref:
      compute std on returns from START_YR up to TRAIN_YEAR inclusive.
    """
    mask = (returns.index.year >= START_YR) & (returns.index.year <= TRAIN_YEAR)
    r_ref = returns.loc[mask].dropna().values
    if len(r_ref) < 50:
        r_ref = returns.dropna().values
    return float(np.std(r_ref, ddof=1))


def build_edges_centers(B: int, L: float):
    edges = np.linspace(-L, L, int(B) + 1)
    centers = edges[:-1] + np.diff(edges) / 2.0
    return edges.astype(float), centers.astype(float)


def compute_qp_lam_mu_parallel_aligned(
    returns: pd.Series,
    edges_std: np.ndarray,
    centers_std: np.ndarray,
    sigma_ref: float
):
    """
    Compute QP lam/mu series on the fixed standardized edges for this B.
    """
    rvals = returns.values
    idxs = returns.index

    # rolling endpoints list: i = WINDOW_DAYS..N, endpoint date = idxs[i-1]
    i_list = list(range(WINDOW_DAYS, len(rvals) + 1))
    dates = [idxs[i - 1] for i in i_list]

    qp_i_list = [
        i for i in i_list
        if (i >= QP_WINDOW_DAYS) and (idxs[i - 1].year >= QP_YR_START)
    ]

    chunks = _chunk_list(qp_i_list, QP_WORKERS * 2)
    print(f"    [QP] {len(qp_i_list)} endpoints → {len(chunks)} chunks using {QP_WORKERS} workers")

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
            edges_std,
            centers_std,
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

    lam_dim = 4 + len(TAIL_SIGMAS)

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
    return lam_df, mu_df


def pick_percentile_threshold_trainonly_require_eval(
    rv_ser: pd.Series,
    future_max: pd.Series,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    test_mask: pd.Series,
    percentiles,
    min_pos_calib=1,
    min_pos_test=1
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
        y_test = safe_series(labs.loc[test_mask].dropna())

        if y_cal.nunique() < 2 or y_test.nunique() < 2:
            continue

        pos_cal = int((y_cal == 1).sum()); neg_cal = int((y_cal == 0).sum())
        pos_te  = int((y_test == 1).sum()); neg_te  = int((y_test == 0).sum())

        if pos_cal >= min_pos_calib and neg_cal >= 1 and pos_te >= min_pos_test and neg_te >= 1:
            return pct, thr, safe_series(labs), {
                "status": "ok",
                "pos_cal": pos_cal, "neg_cal": neg_cal,
                "pos_test": pos_te, "neg_test": neg_te
            }

    pct = min(percentiles)
    thr = float(np.percentile(train_rv, pct))
    labs = (future_max > thr).astype(int)
    y_cal  = safe_series(labs.loc[calib_mask].dropna())
    y_test = safe_series(labs.loc[test_mask].dropna())
    return pct, thr, safe_series(labs), {
        "status": "fallback",
        "pos_cal": int((y_cal == 1).sum()), "neg_cal": int((y_cal == 0).sum()),
        "pos_test": int((y_test == 1).sum()), "neg_test": int((y_test == 0).sum())
    }


def save_roc_pr_for_B(B: int, y_test: pd.Series, score_gen: pd.Series, outdir: str):
    """
    Optional: per-B ROC/PR on test year.
    """
    sc = score_gen.loc[y_test.index].dropna()
    yy = safe_series(y_test.loc[sc.index])
    if len(sc) < 20 or yy.nunique() < 2:
        return

    # ROC
    roc_path = os.path.join(outdir, f"ROC_B{B}_{TEST_YEAR}.png")
    plt.figure(figsize=(7, 6))
    auc = roc_auc_score(yy, sc)
    fpr, tpr, _ = roc_curve(yy, sc)
    plt.plot(fpr, tpr, lw=2, color=METHOD_COLOR, label=f"{METHOD_NAME} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (aligned QP bins) — B={B} | Test {TEST_YEAR}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=250)
    plt.close()

    # PR
    pr_path = os.path.join(outdir, f"PR_B{B}_{TEST_YEAR}.png")
    plt.figure(figsize=(7, 6))
    baseline = float(np.mean(yy))
    ap = average_precision_score(yy, sc)
    prec, rec, _ = precision_recall_curve(yy, sc)
    plt.plot(rec, prec, lw=2, color=METHOD_COLOR, label=f"{METHOD_NAME} (AP={ap:.3f})")
    plt.plot([0, 1], [baseline, baseline], "k--", alpha=0.6, label=f"Chance (AP={baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (aligned QP bins) — B={B} | Test {TEST_YEAR}")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=250)
    plt.close()


def save_daily_scores_for_B(B: int, y_test: pd.Series, score_gen: pd.Series, outdir: str):
    sc = score_gen.loc[y_test.index].dropna()
    yy = safe_series(y_test.loc[sc.index])
    if len(sc) < 20 or yy.nunique() < 2:
        return
    df = pd.DataFrame({
        "date": sc.index,
        "y": yy.astype(int).values,
        "method": [METHOD_NAME] * len(sc),
        "score": sc.values.astype(float),
    })
    path = os.path.join(outdir, f"daily_scores_B{B}_{TEST_YEAR}.csv")
    df.to_csv(path, index=False)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[i] OUTDIR: {OUTDIR}")
    print(f"[i] Split: Train={TRAIN_YEAR}, Calib={CALIB_YEAR}, Test={TEST_YEAR}")
    print(f"[i] Sweep bins: {BINS_SWEEP}")
    print(f"[i] Fixed standardized support: [-{L_STD}, {L_STD}]")
    print(f"[i] QP workers: {QP_WORKERS}")

    start = dt.datetime(START_YR, 1, 1)
    end   = dt.datetime.today()

    df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False).dropna()
    prices = safe_series(df["Close"])
    returns = safe_series(prices.pct_change().dropna())

    print(f"[i] Downloaded returns: {len(returns)} days ({returns.index[0].date()} → {returns.index[-1].date()})")

    # leakage-safe sigma_ref computed from <= TRAIN_YEAR
    sigma_ref = compute_sigma_ref_trainonly(returns)
    print(f"[i] sigma_ref(train-only<= {TRAIN_YEAR}): {sigma_ref:.6g}")

    # rolling endpoints (matching i=WINDOW_DAYS..N)
    base_index = returns.index[WINDOW_DAYS - 1:]

    # realized variance on endpoints
    rv = safe_series(returns.rolling(WINDOW_DAYS).var()).loc[base_index].dropna()

    # labels (same logic style as your main)
    future_max = rv.rolling(FUTURE_HORIZON_DAYS, min_periods=1).max().shift(-1)

    idx = rv.index
    train_mask = (idx.year == TRAIN_YEAR)
    calib_mask = (idx.year == CALIB_YEAR)
    test_mask  = (idx.year == TEST_YEAR)

    pct, thr, labels, diag = pick_percentile_threshold_trainonly_require_eval(
        rv_ser=rv,
        future_max=future_max,
        train_mask=train_mask,
        calib_mask=calib_mask,
        test_mask=test_mask,
        percentiles=PCT_GRID,
        min_pos_calib=MIN_POS_CALIB,
        min_pos_test=MIN_POS_TEST
    )

    print(f"[i] Threshold: pct={pct}, thr={thr:.6g} [{diag.get('status')}] "
          f"cal(pos={diag.get('pos_cal')},neg={diag.get('neg_cal')}), "
          f"test(pos={diag.get('pos_test')},neg={diag.get('neg_test')})")

    # sanity: test must have both classes
    y_test0 = safe_series(labels.loc[test_mask].dropna())
    if y_test0.nunique() < 2:
        raise RuntimeError("Test year has single class under this threshold selection. Adjust PCT_GRID or split.")

    rows = []

    for B in BINS_SWEEP:
        print(f"[i] ===== B={B} =====")
        edges_std, centers_std = build_edges_centers(B, L_STD)

        # QP on aligned discretization (this is what we are sweeping)
        lam_df, mu_df = compute_qp_lam_mu_parallel_aligned(returns, edges_std, centers_std, sigma_ref)

        # align to label index (rolling endpoints)
        common = pd.Index(base_index).intersection(lam_df.index).intersection(labels.index)
        lam_df = lam_df.loc[common]
        mu_df  = mu_df.loc[common]
        labels_al = labels.loc[common]

        train_index = common[common.year == TRAIN_YEAR]
        test_index  = common[common.year == TEST_YEAR]

        # --- Generative ΔE (same math as your main code) ---
        cols = lam_df.columns.tolist()
        gen_w = np.ones(len(cols), dtype=float)
        if APPLY_SYMMETRY_W and "lam_x" in cols:
            gen_w[cols.index("lam_x")] = 0.0

        mu_ref = mu_df.loc[train_index].mean(axis=0)
        dmu = mu_df.sub(mu_ref, axis=1)

        dmu_mat = dmu[[c.replace("lam_", "mu_") for c in cols]]
        dmu_mat.columns = cols

        dE = (lam_df[cols].mul(dmu_mat).mul(gen_w, axis=1)).sum(axis=1)

        if APPLY_DEGEN_EMA:
            dE = dE.ewm(span=DEGEN_EMA_SPAN, adjust=False, min_periods=1).mean()

        dE_train_idx = dE.loc[train_index].dropna().index
        dE_z = z_from_index(dE, dE_train_idx)

        sign_E = choose_sign_on_index(dE_z, labels_al, dE_train_idx) if len(dE_train_idx) else +1.0
        score_gen = sign_E * dE_z

        # --- AUC on test year ---
        y_test = safe_series(labels_al.loc[test_index].dropna())
        sc = score_gen.loc[y_test.index].dropna()
        yy = safe_series(y_test.loc[sc.index])

        auc = np.nan
        if len(sc) >= 20 and yy.nunique() >= 2:
            auc = roc_auc_score(yy, sc)

        rows.append({"bins": int(B), "auc": float(auc)})

        if SAVE_PER_B_ROC_PR:
            save_roc_pr_for_B(B, y_test, score_gen, OUTDIR)
        if SAVE_PER_B_DAILY:
            save_daily_scores_for_B(B, y_test, score_gen, OUTDIR)

        print(f"    AUC(test {TEST_YEAR}) = {auc:.6f}" if np.isfinite(auc) else "    AUC = NaN (insufficient / single class)")

    auc_df = pd.DataFrame(rows).sort_values("bins").reset_index(drop=True)

    auc_csv = os.path.join(
        OUTDIR,
        f"AUC_vs_bins_{TRAIN_YEAR}_{CALIB_YEAR}_{TEST_YEAR}_alignedQP_GenerativeOnly.csv"
    )
    auc_df.to_csv(auc_csv, index=False)

    fig_path = os.path.join(
        OUTDIR,
        f"AUC_vs_bins_{TRAIN_YEAR}_{CALIB_YEAR}_{TEST_YEAR}_alignedQP_GenerativeOnly.png"
    )

    plt.figure(figsize=(9, 5))
    plt.plot(auc_df["bins"].values, auc_df["auc"].values, "-o", lw=2, color=METHOD_COLOR, label=METHOD_NAME)
    plt.axhline(0.5, ls="--", lw=1.5, alpha=0.7)
    plt.ylim(0.0, 1.0)
    plt.xlabel("QP histogram bins B (fixed standardized support)")
    plt.ylabel(f"AUC on test year {TEST_YEAR}")
    plt.title(
        f"AUC vs Number of Bins — Train {TRAIN_YEAR}, Calib {CALIB_YEAR}, Test {TEST_YEAR}"
    )
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[done] Saved CSV:    {auc_csv}")
    print(f"[done] Saved figure: {fig_path}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
