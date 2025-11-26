# -*- coding: utf-8 -*-
"""
v2/05a_cluster_and_profiles_v2.py — Final clustering (K=6) + profiles.
Writes to v2_data/05_a_kmeans/.

Change in this version:
  - Append per-cluster PCA centroids to the profiles CSV as columns:
    pc1_centroid, pc2_centroid, pc3_centroid, ... (up to n_components).

Pipeline:
  - Read latest 03 features
  - Filter dead_flag==0 (keep zero/low windows)
  - Preprocess: log1p -> StandardScaler -> PCA (>=90% cumulative EVR; fail-safe whiten)
  - Fit KMeans in PCA space (K=6)
  - Save:
    1) v2_data/05_a_kmeans/05a_cluster_assignments_v2_<STAMP>_K6.csv
    2) v2_data/05_a_kmeans/05a_cluster_profiles_v2_<STAMP>_K6.csv
    3) v2_data/05_a_kmeans/05a_pca_model_meta_v2_<STAMP>_K6.json
    4) v2_data/05_a_kmeans/05a_kmeans_meta_v2_<STAMP>_K6.json
"""

from __future__ import annotations
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------- paths & stamp ----------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
D03   = ROOT / "v2_data" / "03_features"
D05_A = ROOT / "v2_data" / "05_a_kmeans"   # <= keep as requested
D05_A.mkdir(parents=True, exist_ok=True)

def latest_stamp_from_03(d03: Path) -> str:
    cands = sorted(d03.glob("03_features_weekly_v2_*.csv"))
    if not cands:
        raise FileNotFoundError("No 03_features_weekly_v2_*.csv under v2_data/03_features/")
    m = re.search(r"_(\d{8}_\d{6})\.csv$", cands[-1].name)
    if not m:
        raise RuntimeError(f"Cannot parse STAMP from filename: {cands[-1].name}")
    return m.group(1)

STAMP = latest_stamp_from_03(D03)
F_FEAT = D03 / f"03_features_weekly_v2_{STAMP}.csv"

# ---------------- config ----------------
RANDOM_STATE = 123
K = 6  # decided this batch
EXPLAINED_VAR_THRESHOLD = 0.90   # PCA cumulative EVR threshold
PC1_DOMINANCE_THRESHOLD = 0.70   # if PC1 EVR > 0.70 ...
PC1_ISSUES_LOADING_MIN  = 0.70   # ... AND |loading on issues_closed| >= 0.70 => whiten=True

FEATURES = [
    "commits_8w_sum",
    "contributors_8w_unique",
    "issues_closed_8w_count",
    "releases_8w_count",
]
ISSUES_COL = "issues_closed_8w_count"

# ---------------- helpers ----------------
def log1p_df(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log1p transform."""
    return np.log1p(df)

def _n_comp(pca_obj) -> int:
    return int(getattr(pca_obj, "n_components_", getattr(pca_obj, "n_components", 0)))

def choose_pca(Xz: np.ndarray, feature_names: List[str]) -> Tuple[PCA, int, bool, np.ndarray, np.ndarray]:
    """
    Fit PCA on standardized FULL DATA (dead_flag==0); pick #components; decide whitening.
    Returns: (pca, n_components_used, whiten_flag, loadings, evr)
    """
    pca_raw = PCA(n_components=min(Xz.shape[1], len(feature_names)),
                  whiten=False, random_state=RANDOM_STATE).fit(Xz)
    evr = pca_raw.explained_variance_ratio_
    cum = np.cumsum(evr)
    m = int(np.searchsorted(cum, EXPLAINED_VAR_THRESHOLD) + 1)
    m = max(1, min(m, Xz.shape[1]))

    loadings = pca_raw.components_
    whiten_flag = False
    if len(evr) > 0 and evr[0] > PC1_DOMINANCE_THRESHOLD:
        try:
            idx_issues = feature_names.index(ISSUES_COL)
            if abs(loadings[0, idx_issues]) >= PC1_ISSUES_LOADING_MIN:
                whiten_flag = True
        except ValueError:
            pass

    pca = PCA(n_components=m, whiten=whiten_flag, random_state=RANDOM_STATE).fit(Xz)
    return pca, _n_comp(pca), bool(whiten_flag), pca.components_, pca.explained_variance_ratio_

# ---------------- load ----------------
# (drop infer_datetime_format to avoid FutureWarning)
df = pd.read_csv(F_FEAT, parse_dates=["week_dt"])
required = {"repo","week_unix","week_dt","dead_flag", *FEATURES}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns in {F_FEAT.name}: {sorted(missing)}")

# has_release_8w (create if absent)
if "has_release_8w" not in df.columns:
    df["has_release_8w"] = (df["releases_8w_count"] > 0).astype(int)

# zero_8w flag (all four features are zero)
df["zero_8w"] = (
    (df["commits_8w_sum"]==0) &
    (df["contributors_8w_unique"]==0) &
    (df["issues_closed_8w_count"]==0) &
    (df["releases_8w_count"]==0)
).astype(int)

# filter: dead_flag==0
df = df[df["dead_flag"] == 0].copy().reset_index(drop=True)

# ---------------- preprocess & PCA ----------------
X_raw = df[FEATURES].copy()
X_log = log1p_df(X_raw)
scaler = StandardScaler(with_mean=True, with_std=True).fit(X_log)
X_z = scaler.transform(X_log)

pca, m, whiten_flag, loadings, evr = choose_pca(X_z, FEATURES)
Z = pca.transform(X_z)  # PCA coordinates

# ---------------- clustering in PCA space ----------------
km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
labels = km.fit_predict(Z)

# Build assignments with PCA coordinates
assign_df = df[["repo","week_unix","week_dt"] + FEATURES + ["has_release_8w","zero_8w"]].copy()
assign_df["cluster"] = labels

# Rank clusters by median commits (descending = more active); keep both raw and rank label
med = assign_df.groupby("cluster")["commits_8w_sum"].median().sort_values(ascending=False)
rank_map = {cl: i for i, cl in enumerate(med.index)}  # 0 is highest activity
assign_df["cluster_rank_by_commits"] = assign_df["cluster"].map(rank_map)

# expose per-window PCA coordinates (unchanged)
for i in range(_n_comp(pca)):
    assign_df[f"PC{i+1}"] = Z[:, i]

# ---------------- profiles on ORIGINAL features ----------------
def q25(s: pd.Series) -> float: return float(s.quantile(0.25))
def q75(s: pd.Series) -> float: return float(s.quantile(0.75))

agg_dict = {
    "n_windows":          ("repo", "size"),
    "has_release_8w_rate":("has_release_8w", "mean"),
    "zero_8w_rate":       ("zero_8w", "mean"),
    "cluster_rank_by_commits": ("cluster_rank_by_commits", "first"),
}
for col in FEATURES:
    agg_dict[f"{col}_p50"] = (col, "median")
    agg_dict[f"{col}_p25"] = (col, q25)
    agg_dict[f"{col}_p75"] = (col, q75)

profiles = assign_df.groupby("cluster", as_index=True).agg(**agg_dict).reset_index()

# === NEW: attach PCA centroids per cluster into profiles (ONLY this addition) ===
centroid_cols = [f"pc{i+1}_centroid" for i in range(_n_comp(pca))]
centroid_df = pd.DataFrame(km.cluster_centers_, columns=centroid_cols)
centroid_df["cluster"] = np.arange(K, dtype=int)
profiles = profiles.merge(centroid_df, on="cluster", how="left")
# === END NEW ===

profiles = profiles.sort_values("cluster_rank_by_commits").reset_index(drop=True)

# ---------------- save ----------------
assign_csv   = D05_A / f"05a_cluster_assignments_v2_{STAMP}_K{K}.csv"
profiles_csv = D05_A / f"05a_cluster_profiles_v2_{STAMP}_K{K}.csv"
assign_df.to_csv(assign_csv, index=False)
profiles.to_csv(profiles_csv, index=False)

# PCA & KMeans meta for traceability
pca_meta = {
    "stamp": STAMP,
    "features": FEATURES,
    "n_components": _n_comp(pca),
    "whiten": bool(whiten_flag),
    "explained_variance_ratio": [float(x) for x in evr],
    "loadings": {f"PC{i+1}": {FEATURES[j]: float(loadings[i, j]) for j in range(len(FEATURES))}
                 for i in range(loadings.shape[0])},
    "scaler_mean_log1p": [float(x) for x in scaler.mean_],
    "scaler_scale_log1p": [float(x) for x in scaler.scale_],
}
with open(D05_A / f"05a_pca_model_meta_v2_{STAMP}_K{K}.json", "w", encoding="utf-8") as f:
    json.dump(pca_meta, f, ensure_ascii=False, indent=2)

km_meta = {
    "stamp": STAMP,
    "k": int(K),
    "random_state": RANDOM_STATE,
    "n_init": 10,
    "centroids_PCA": km.cluster_centers_.tolist(),
    "label_order_by_commits_median": med.index.tolist(),
}
with open(D05_A / f"05a_kmeans_meta_v2_{STAMP}_K{K}.json", "w", encoding="utf-8") as f:
    json.dump(km_meta, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved assignments -> {assign_csv}")
print(f"[OK] Saved profiles   -> {profiles_csv}")
print(f"[OK] Saved PCA meta   -> {D05_A / f'05a_pca_model_meta_v2_{STAMP}_K{K}.json'}")
print(f"[OK] Saved KMeans meta-> {D05_A / f'05a_kmeans_meta_v2_{STAMP}_K{K}.json'}")
print(f"[INFO] STAMP={STAMP} | K={K} | PCA n_components={_n_comp(pca)} | whiten={whiten_flag} | windows={len(df)}")
