# -*- coding: utf-8 -*-
"""
v2/07_export_bundle_v2.py
Objective:
  Pack the trained PCA, KMeans models, and scaling logic into a single binary file (.pkl).
  This allows the web app (Streamlit) to load the "brain" instantly without retraining.

Outputs:
  - v2_data/07_bundle/model_bundle_v2.pkl
"""

import re
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np

# [CONFIG] Paths
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
D03 = ROOT / "v2_data" / "03_features"
D05_A = ROOT / "v2_data" / "05_a_kmeans"
D05_B = ROOT / "v2_data" / "05_b_apply"

# [MODIFIED] Output to a specific 07 folder to keep things organized
OUTPUT_DIR = ROOT / "v2_data" / "07_bundle"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# [CONFIG] Columns used for training
FEATURES = [
    "commits_8w_sum",
    "contributors_8w_unique",
    "issues_closed_8w_count",
    "releases_8w_count",
]


# Helper: Find the latest stamp from 05a
def get_latest_stamp_k():
    cands = sorted(D05_A.glob("05a_pca_model_meta_v2_*.json"))
    if not cands:
        raise FileNotFoundError("No PCA meta found in 05_a_kmeans. Did you run 05a?")
    # Filename format: 05a_pca_model_meta_v2_<STAMP>_K<K>.json
    m = re.search(r"_v2_(\d{8}_\d{6})_K(\d+)\.json$", cands[-1].name)
    if not m:
        raise ValueError(f"Cannot parse STAMP from {cands[-1].name}")
    return m.group(1), int(m.group(2))


def main():
    print("[INFO] Starting Model Bundle Export (Step 07)...")

    # 1. Identify the latest model version
    stamp, k = get_latest_stamp_k()
    print(f"[INFO] Target Version: STAMP={stamp}, K={k}")

    # 2. Define file paths for Meta and Assignments
    f_pca_meta = D05_A / f"05a_pca_model_meta_v2_{stamp}_K{k}.json"
    f_kmeans_meta = D05_A / f"05a_kmeans_meta_v2_{stamp}_K{k}.json"
    f_cluster_map = D05_B / f"05b_cluster_label_map_v2_{stamp}_K{k}.csv"

    if not f_cluster_map.exists():
        raise FileNotFoundError(f"Cluster Map not found: {f_cluster_map}\nPlease run 05b first.")

    # 3. Load Metadata
    with open(f_pca_meta, "r", encoding="utf-8") as f:
        pca_data = json.load(f)

    with open(f_kmeans_meta, "r", encoding="utf-8") as f:
        kmeans_data = json.load(f)

    # Load Cluster Label Map (ID -> Name)
    df_map = pd.read_csv(f_cluster_map)
    # Create a dictionary: {0: "Dormant", 1: "Internal Development", ...}
    label_map = dict(zip(df_map["cluster"], df_map["stage_name"]))

    # 4. Reconstruct the Scaler (StandardScaler)
    scaler_params = {
        "mean": np.array(pca_data["scaler_mean_log1p"]),
        "scale": np.array(pca_data["scaler_scale_log1p"])
    }

    # 5. Reconstruct PCA
    n_components = pca_data["n_components"]
    pca_components = np.zeros((n_components, len(FEATURES)))

    # Fill the matrix from JSON
    loadings_dict = pca_data["loadings"]
    for i in range(n_components):
        pc_key = f"PC{i + 1}"
        for j, feat in enumerate(FEATURES):
            pca_components[i, j] = loadings_dict[pc_key][feat]

    pca_params = {
        "components": pca_components,
        "whiten": pca_data["whiten"],
        "explained_variance": np.array(pca_data["explained_variance_ratio"])
    }

    # 6. Reconstruct KMeans Centroids
    kmeans_centroids = np.array(kmeans_data["centroids_PCA"])

    # 7. Pack everything into a bundle dictionary
    bundle = {
        "version_stamp": stamp,
        "K": k,
        "features": FEATURES,
        "scaler": scaler_params,
        "pca": pca_params,
        "kmeans_centroids": kmeans_centroids,
        "label_map": label_map,
        "dead_threshold_weeks": 24
    }

    # 8. Save as pickle
    out_file = OUTPUT_DIR / "model_bundle_v2.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[OK] Bundle saved to: {out_file}")
    print(f"[INFO] Ready for deployment! This file contains the trained logic.")


if __name__ == "__main__":
    main()