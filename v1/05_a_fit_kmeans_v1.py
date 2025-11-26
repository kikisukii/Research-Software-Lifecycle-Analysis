#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_a_fit_kmeans_v1.py
Objective: Train a KMeans model (Unsupervised) on the full compliant dataset using (K*, Alpha*),
           freeze the model, and export inspection CSVs (Assignments & Profiles).

- Compliant Samples: dead_flag==0 AND age_weeks>=24 (v1 rule)
- Feature Space: x=log1p(c8_total), y=M8_24
- Preprocessing: RobustScaler -> Alpha Weighting (w_c=2-alpha, w_m=alpha) -> KMeans
- Output Directory: v1_data/05_a_kmeans/
    - 05_global_models_v1_<STAMP>.json
    - 05_labeling_template_v1_<STAMP>_K{K}.json
    - 05_assignment_v1_<STAMP>_K{K}.csv  <-- NEW: See which repo is in which cluster
    - 05_profile_v1_<STAMP>_K{K}.csv     <-- NEW: Cluster statistics

Usage:
  python 05_a_fit_kmeans_v1.py --features_csv ... --k 4 --alpha 1.90
"""

import os
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# [CONFIG] Path Logic
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V1_DATA_ROOT = PROJECT_ROOT / "v1_data"
INPUT_DIR_DEFAULT = V1_DATA_ROOT / "03_features"
OUTPUT_DIR_ROOT = V1_DATA_ROOT / "05_a_kmeans"

STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def infer_stamp(p: str) -> str:
    m = STAMP_RE.search(Path(p).name)
    if not m:
        raise ValueError(f"Cannot infer <STAMP> from filename: {p}")
    return m.group(1)


def find_latest_features(base_dir: Path) -> Path:
    files = list(base_dir.glob("*_filtered.csv"))
    if not files:
        raise FileNotFoundError(f"No filtered features found in {base_dir}")
    return sorted(files, key=lambda p: infer_stamp(p.name))[-1]


def alpha_weights(alpha: float):
    return float(2.0 - alpha), float(alpha)


def pick_from_grid(grid_csv: str):
    df = pd.read_csv(grid_csv)
    df = df.sort_values(["sil_mean", "ch_mean"], ascending=[False, False]).reset_index(drop=True)
    row = df.iloc[0]
    print(f"[AUTO-PICK] Selected K={int(row['K'])}, Alpha={float(row['alpha'])}")
    return int(row["K"]), float(row["alpha"])


def load_features(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = df[(df["dead_flag"] == 0) & (df["age_weeks"] >= 24)].copy()
    df["log_c8"] = np.log1p(df["c8_total"]).astype(np.float32)
    df["M"] = df["M8_24"].astype(np.float32)
    return df


def fit_kmeans(df: pd.DataFrame, K: int, alpha: float, random_state: int = 42):
    # 1) Robust Scaling
    X_base = df[["log_c8", "M"]].to_numpy(dtype=np.float32)
    scaler = RobustScaler()
    X_std = scaler.fit_transform(X_base).astype(np.float32)

    # 2) Alpha Weighting
    w_c, w_m = alpha_weights(alpha)
    X_w = X_std.copy()
    X_w[:, 0] *= w_c
    X_w[:, 1] *= w_m

    # 3) KMeans Training
    km = KMeans(n_clusters=K, n_init=50, max_iter=1000, random_state=random_state)
    km.fit(X_w)

    # Save unweighted centers for portable model
    centers_w = km.cluster_centers_.astype(np.float32)
    centers_un = centers_w / np.array([w_c, w_m], dtype=np.float32)

    return {
        "scaler_center": scaler.center_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "centers_std": centers_un.astype(float).tolist(),
        "w_c": float(w_c), "w_m": float(w_m),
        "K": int(K), "alpha": float(alpha),
        "n_init": int(km.n_init), "max_iter": int(km.max_iter),
        "fit_samples": int(len(df)), "fit_repos": int(df["repo"].nunique()),
    }


def assign_clusters_weighted(df: pd.DataFrame, model: dict):
    """
    Re-calculates cluster assignments using the EXACT same weighted logic as training.
    Essential for exporting accurate CSVs.
    """
    X = df[["log_c8", "M"]].to_numpy(dtype=np.float32)

    # 1. Standardize
    center = np.array(model["scaler_center"], dtype=np.float32)
    scale = np.array(model["scaler_scale"], dtype=np.float32)
    Xz = (X - center) / np.maximum(scale, 1e-12)

    # 2. Apply Weights
    w_c, w_m = model["w_c"], model["w_m"]
    Xw = Xz.copy()
    Xw[:, 0] *= w_c
    Xw[:, 1] *= w_m

    # 3. Calculate Distance to Weighted Centers
    # Reconstruct Weighted Centers from the saved Unweighted ones
    centers_un = np.array(model["centers_std"], dtype=np.float32)
    Cw = centers_un.copy()
    Cw[:, 0] *= w_c
    Cw[:, 1] *= w_m

    # Distance
    d2 = ((Xw[:, None, :] - Cw[None, :, :]) ** 2).sum(axis=2)
    cid = np.argmin(d2, axis=1).astype(int)

    return cid


def build_outputs(df: pd.DataFrame, model: dict):
    # 1. Get assignments
    cid = assign_clusters_weighted(df, model)
    df_assigned = df.assign(cluster_id=cid)

    # 2. Build Profile (Stats per cluster)
    profile_df = (df_assigned
                  .groupby("cluster_id")
                  .agg(n=("repo", "size"),
                       n_repo=("repo", "nunique"),
                       log_c8_med=("log_c8", "median"),
                       log_c8_q25=("log_c8", lambda x: float(np.quantile(x, 0.25))),
                       log_c8_q75=("log_c8", lambda x: float(np.quantile(x, 0.75))),
                       M_med=("M", "median"),
                       M_q25=("M", lambda x: float(np.quantile(x, 0.25))),
                       M_q75=("M", lambda x: float(np.quantile(x, 0.75))),
                       )).reset_index().sort_values("cluster_id")

    # 3. Build Template JSON structure
    clusters_list = []
    for _, r in profile_df.iterrows():
        clusters_list.append({
            "cluster_id": int(r.cluster_id),
            "n": int(r.n),
            "n_repo": int(r.n_repo),
            "log_c8": {"median": float(r.log_c8_med), "q25": float(r.log_c8_q25), "q75": float(r.log_c8_q75)},
            "M8_24": {"median": float(r.M_med), "q25": float(r.M_q25), "q75": float(r.M_q75)},
            "suggested_stage": "",
            "stage": ""  # User to fill this
        })

    tmpl_json = {
        "K": model["K"],
        "features": ["log_c8", "M8_24"],
        "clusters": clusters_list
    }

    return df_assigned, profile_df, tmpl_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--grid_csv", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    # Locate Input
    if args.features_csv:
        features_path = Path(args.features_csv)
    else:
        features_path = find_latest_features(INPUT_DIR_DEFAULT)
    stamp = infer_stamp(features_path.name)

    # Locate Output
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = OUTPUT_DIR_ROOT
    outdir.mkdir(parents=True, exist_ok=True)

    # Params
    if (args.k is None or args.alpha is None):
        if not args.grid_csv:
            raise ValueError("Must provide --k and --alpha, OR --grid_csv")
        K, alpha = pick_from_grid(args.grid_csv)
    else:
        K, alpha = args.k, args.alpha

    print(f"[INFO] Training KMeans K={K}, Alpha={alpha} on {features_path.name}")

    # Load & Train
    df = load_features(str(features_path))
    model = fit_kmeans(df, K, alpha)

    # Generate Assignments, Profiles, and Template
    df_assigned, df_profile, tmpl_json = build_outputs(df, model)

    # Construct Model JSON
    model_json = {
        "features": ["log_c8", "M8_24"],
        "centers_space": "std_unweighted",
        "scaler": "robust",
        "alpha": model["alpha"], "K": model["K"],
        "scaler_center": model["scaler_center"],
        "scaler_scale": model["scaler_scale"],
        "centers_std": model["centers_std"],
        "w_c": model["w_c"], "w_m": model["w_m"],
        "n_init": model["n_init"], "max_iter": model["max_iter"],
        "fit_samples": model["fit_samples"], "fit_repos": model["fit_repos"]
    }

    # Save Paths
    fp_model = outdir / f"05_global_models_v1_{stamp}.json"
    fp_tmpl = outdir / f"05_labeling_template_v1_{stamp}_K{K}.json"
    fp_assign = outdir / f"05_assignment_v1_{stamp}_K{K}.csv"
    fp_prof = outdir / f"05_profile_v1_{stamp}_K{K}.csv"

    # Write Files
    with open(fp_model, "w", encoding="utf-8") as f:
        json.dump({f"K{K}": model_json}, f, ensure_ascii=False, indent=2)
    with open(fp_tmpl, "w", encoding="utf-8") as f:
        json.dump(tmpl_json, f, ensure_ascii=False, indent=2)

    # Save CSVs
    cols_assign = ["repo", "week_unix", "c8_total", "log_c8", "M8_24", "cluster_id"]
    df_assigned[cols_assign].to_csv(fp_assign, index=False)
    df_profile.to_csv(fp_prof, index=False)

    print(f"[OK] Model JSON   -> {fp_model}")
    print(f"[OK] Label Tmpl   -> {fp_tmpl}")
    print(f"[OK] Assignments  -> {fp_assign}")
    print(f"[OK] Profiles     -> {fp_prof}")
    print("-" * 60)
    print("NEXT: Check CSVs to confirm clusters, edit Label Tmpl JSON, then run 05b.")


if __name__ == "__main__":
    main()