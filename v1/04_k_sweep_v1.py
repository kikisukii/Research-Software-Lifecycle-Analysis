#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_k_sweep_v1.py  (coarse-to-fine ready, zero-args friendly)
------------------------------------------------------------
Performs two main tasks (Step 04 stops here, NO labeling yet):
  1) Generate Fold Assignment (GroupKFold=5, grouped by repo):
     -> v1_data/04_cluster/04_a_fold_assign_v1_<STAMP>.csv
  2) Run (K, Alpha) Grid Search on "Train Folds Only" using Silhouette/CH scores:
     -> v1_data/04_cluster/04_b_alpha_k_grid_v1_<STAMP>.csv

Features & Transformation (v1):
  - Filter: dead_flag==0 AND age_weeks>=24
  - log1p(c8_total) -> RobustScaler([log_c8, M8_24]) -> Alpha Weighting (Column Scaling)
  - Weights: w_activity = 2 - alpha, w_momentum = alpha
  - Standardization is fitted once per fold; Alpha weighting is applied as constant column multiplication.

Performance Optimization:
  - Silhouette sampling (--sil_sample, default 20000; 0 for full sample)
  - Grid search settings: n_init=10, max_iter=200
  - Algorithm preference: 'elkan' (if supported) for speed.

Usage (Zero-Args):
  - Automatically finds the latest '03_features_weekly_v1_*_filtered.csv' in 'v1_data/03_features/'
  - Infers <STAMP> and outputs directly to 'v1_data/04_cluster/' (No extra subfolder)

Parameter Tuning Guide:
  - To customize the search range, use command line arguments.
  - Example (Standard):
      python 04_k_sweep_v1.py --alpha_range 0.6 1.6 0.1
  - Example (Follow-up Experiment / Fine-tuning):
      # Use --out_dir to separate tuning results from the main folder
      python 04_k_sweep_v1.py --alpha_range 1.5 1.9 0.05 --out_dir ../v1_data/04_cluster/tuning_1.5_1.9

Arguments:
  --features_csv PATH
  --out_dir PATH
  --stamp STAMP
  --n_splits 5 --kmin 2 --kmax 5
  --alpha_range 0.6 1.6 0.1
  --sil_sample 20000
"""

import os
import re
import glob
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# ---------- utilities ----------
STAMP_RE = re.compile(r"(\d{8}_\d{6})")

# [CONFIG] Path Logic
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../v1
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # .../project_root
V1_DATA_ROOT = os.path.join(PROJECT_ROOT, "v1_data")  # .../v1_data
INPUT_DIR_DEFAULT = os.path.join(V1_DATA_ROOT, "03_features")
OUTPUT_DIR_ROOT = os.path.join(V1_DATA_ROOT, "04_cluster")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def infer_stamp_from_name(path: str) -> str:
    m = STAMP_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot infer <STAMP> from filename: {path}")
    return m.group(1)


def find_latest_filtered_csv(base_dir: str) -> str:
    # Look into the specific 03_features directory
    pattern = os.path.join(base_dir, "*_filtered.csv")
    cands = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(f"No matching files found: {pattern}\nPlease check if Step 03 has been run.")

    def stamp_key(p: str):
        m = STAMP_RE.search(os.path.basename(p))
        return m.group(1) if m else "00000000_000000"

    cands.sort(key=stamp_key)
    return cands[-1]


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "week_date_utc" not in df.columns and "week_date" in df.columns:
        df = df.rename(columns={"week_date": "week_date_utc"})
    required = {"repo", "week_unix", "week_date_utc", "c8_total", "M8_24", "dead_flag", "age_weeks"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# ---------- fold assignment ----------
def make_fold_assign(df_in: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """
    Creates a fold assignment ensuring all weeks of a single repo belong to the same fold.
    """
    df = df_in.copy()
    # Ensure consistency with the 'filtered' logic
    df = df[(df["dead_flag"] == 0) & (df["age_weeks"] >= 24)].sort_values(["repo", "week_unix"])
    gkf = GroupKFold(n_splits=n_splits)
    repos = df["repo"].values
    X_dummy = np.zeros((len(df), 1), dtype=np.int8)
    fold_of_repo = {}

    # Split based on repo groups
    for fid, (_, test_idx) in enumerate(gkf.split(X_dummy, groups=repos)):
        for r in pd.Series(repos[test_idx]).unique():
            fold_of_repo[r] = fid

    fa = pd.DataFrame({"repo": list(fold_of_repo.keys()), "fold_id": list(fold_of_repo.values())})
    fa = fa.sort_values("repo").reset_index(drop=True)
    assert fa["repo"].is_unique
    return fa


# ---------- weighting ----------
def alpha_weights(alpha: float) -> Tuple[float, float]:
    """Returns weights (w_activity, w_momentum) based on alpha."""
    return 2.0 - alpha, alpha


# ---------- scoring ----------
def kmeans_score(X: np.ndarray, k: int, n_init: int, max_iter: int,
                 random_state: int, sil_sample: int) -> Tuple[float, float]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] <= k:
        return np.nan, np.nan

    # Adapt for sklearn versions regarding 'algorithm' parameter
    try:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                    random_state=random_state, algorithm="elkan")
    except TypeError:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                    random_state=random_state)

    y = km.fit_predict(X)

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    try:
        if sil_sample and X.shape[0] > sil_sample:
            sil = silhouette_score(X, y, sample_size=sil_sample, random_state=random_state)
        else:
            sil = silhouette_score(X, y)
    except Exception:
        sil = np.nan

    try:
        ch = calinski_harabasz_score(X, y)
    except Exception:
        ch = np.nan

    return sil, ch


def run_grid(df_in: pd.DataFrame, fold_assign: pd.DataFrame,
             k_values: List[int], alphas: List[float],
             n_splits: int, n_init: int, max_iter: int, random_state: int,
             sil_sample: int) -> pd.DataFrame:
    # Only non-dead projects; prepare base features
    df = df_in[df_in["dead_flag"] == 0].copy()
    df["log_c8"] = np.log1p(df["c8_total"]).astype(np.float32)
    df["M"] = df["M8_24"].astype(np.float32)

    # Merge fold_ids
    fmap = dict(zip(fold_assign["repo"], fold_assign["fold_id"]))
    df["fold_id"] = df["repo"].map(fmap)

    if df["fold_id"].isna().any():
        miss = df[df["fold_id"].isna()]["repo"].unique()
        raise RuntimeError(
            f"Repositories missing fold_id: {miss[:5]}...; Check if fold assignment covers all modeled repos.")

    # Pre-calculate Scaled Features per Fold (Optimization)
    fold_scaled = {}
    for f in range(n_splits):
        # Scale based on Training data (NOT fold f), apply to all?
        # Logic here: we fit scaler on Training set (folds != f), and transform validation set (fold f)?
        # The code iterates `for f in range(n_splits)` and then inside grid loop it uses `fold_scaled[f]`.
        # Wait, the original code (v1) calculated `Xs` using `df.loc[df["fold_id"] != f]`.
        # This implies `fold_scaled[f]` contains the TRANSFORMED TRAINING DATA, not the validation data.
        # Then `kmeans_score` is called on `Xw` (which comes from `Xs`).
        # So we are clustering the TRAINING set to evaluate stability/score?
        # YES: The logic is evaluating the quality of clustering on the Training set itself
        # (or potentially finding K on training set). This is consistent with "Training Folds Only".

        Xtr_base = df.loc[df["fold_id"] != f, ["log_c8", "M"]].to_numpy(dtype=np.float32)
        scaler = RobustScaler()
        Xs = scaler.fit_transform(Xtr_base).astype(np.float32)
        fold_scaled[f] = Xs

    results = []
    total = len(k_values) * len(alphas)

    with tqdm(total=total, ncols=100, desc="[GRID] (K, Alpha) Scoring") as bar:
        for k in k_values:
            for alpha in alphas:
                w_c, w_m = alpha_weights(alpha)
                sils, chs = [], []

                for f in range(n_splits):
                    Xs = fold_scaled[f]
                    Xw = Xs.copy()
                    # Apply Alpha weights
                    Xw[:, 0] *= w_c
                    Xw[:, 1] *= w_m

                    sil, ch = kmeans_score(Xw, k, n_init, max_iter, random_state, sil_sample)
                    if not (np.isnan(sil) and np.isnan(ch)):
                        sils.append(sil)
                        chs.append(ch)

                row = {
                    "K": k, "alpha": alpha,
                    "sil_mean": float(np.nanmean(sils)) if sils else np.nan,
                    "sil_std": float(np.nanstd(sils)) if sils else np.nan,
                    "ch_mean": float(np.nanmean(chs)) if chs else np.nan,
                    "ch_std": float(np.nanstd(chs)) if chs else np.nan,
                    "used_folds": len(sils),
                    "n_splits": n_splits,
                    "n_init": n_init, "max_iter": max_iter, "random_state": random_state,
                    "log1p": True, "scaler": "robust",
                    "w_c_formula": "2 - alpha", "w_m_formula": "alpha",
                    "sil_sample": sil_sample
                }
                results.append(row)
                bar.update(1)

    out = pd.DataFrame(results).sort_values(["sil_mean", "ch_mean"], ascending=[False, False]).reset_index(drop=True)
    return out


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="04_k_sweep_v1: Fold Assignment + (K, Alpha) Grid Search")
    parser.add_argument("--features_csv", type=str, default=None, help="Path to 03_features_weekly_v1_*_filtered.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: v1_data/04_cluster/)")
    parser.add_argument("--stamp", type=str, default=None, help="<STAMP> (default: inferred from filename)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=5)
    parser.add_argument("--alpha_range", nargs=3, type=float, default=[0.6, 1.6, 0.1],
                        help="Alpha Start End Step (e.g., 0.6 1.6 0.1)")
    parser.add_argument("--sil_sample", type=int, default=20000, help="Silhouette sample size; 0 for full (slow)")
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    # Use INPUT_DIR_DEFAULT logic
    features_csv = args.features_csv or find_latest_filtered_csv(INPUT_DIR_DEFAULT)
    stamp = args.stamp or infer_stamp_from_name(features_csv)

    # [CHANGED] Simplified Output Logic
    # If out_dir is not provided, use OUTPUT_DIR_ROOT directly (no subdirectory)
    # If out_dir is provided (e.g. for tuning), use it as is.
    out_dir = args.out_dir or OUTPUT_DIR_ROOT
    ensure_dir(out_dir)

    print("=" * 80)
    print(f"[INFO] 04_k_sweep_v1 | STAMP={stamp}")
    print(f"[INFO] features_csv : {features_csv}")
    print(f"[INFO] out_dir      : {out_dir}")
    print(f"[INFO] k in [{args.kmin}, {args.kmax}], alpha={args.alpha_range}, "
          f"n_splits={args.n_splits}, n_init={args.n_init}, max_iter={args.max_iter}, "
          f"random_state={args.random_state}, sil_sample={args.sil_sample}")
    print("=" * 80)

    df = load_features(features_csv)
    n_full = len(df)
    n_repo = df["repo"].nunique()
    tmin = pd.to_datetime(df["week_unix"], unit="s", utc=True).min().date()
    tmax = pd.to_datetime(df["week_unix"], unit="s", utc=True).max().date()
    print(f"[INFO] 03_filtered Rows: {n_full:,}; Repos: {n_repo:,}; Range: {tmin} -> {tmax} (UTC)")

    # 04_a: Fold Assignment
    print("\n[STEP] 04_a Generating Fold Assignment (GroupKFold by repo) ...")
    fold_assign = make_fold_assign(df, n_splits=args.n_splits)
    fold_csv = os.path.join(out_dir, f"04_a_fold_assign_v1_{stamp}.csv")
    fold_assign.to_csv(fold_csv, index=False)
    print(f"[OK] Saved: {fold_csv} | Rows: {len(fold_assign):,} (One fold_id per repo)")

    # 04_b: Grid Search
    print("\n[STEP] 04_b (K, Alpha) Grid Scoring (Train Folds Only) ...")
    k_values = list(range(args.kmin, args.kmax + 1))
    a0, a1, astep = args.alpha_range
    # Using numpy to handle float steps accurately
    alphas = [round(x, 10) for x in np.arange(a0, a1 + 1e-9, astep)]

    print(f"[INFO] K Candidates: {k_values}")
    print(f"[INFO] Alpha Candidates: {alphas}")

    grid_df = run_grid(df, fold_assign, k_values, alphas,
                       args.n_splits, args.n_init, args.max_iter, args.random_state,
                       args.sil_sample)

    grid_csv = os.path.join(out_dir, f"04_b_alpha_k_grid_v1_{stamp}.csv")
    grid_df.to_csv(grid_csv, index=False)
    print(f"[OK] Saved: {grid_csv} | Rows: {len(grid_df):,}")

    print("\n[DONE] Step 04 complete: Fold assignment + Grid scoring generated.")
    print("       Next Step: Select optimal (K*, Alpha*) and proceed to 05_a_kmeans/b.")


if __name__ == "__main__":
    main()