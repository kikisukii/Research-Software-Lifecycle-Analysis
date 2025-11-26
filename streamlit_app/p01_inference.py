# -*- coding: utf-8 -*-
"""
inference.py
Logic for the Streamlit App:
1. Git clone a target repo to a temp folder.
2. Extract weekly commits (and releases) similar to v2 logic.
3. Compute 8-week rolling features.
4. Apply the pre-trained PCA + KMeans model (from .pkl).
5. Return a DataFrame ready for plotting.
"""

import os
import shutil
import pickle
import tempfile
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


# Helper: Align to Sunday 00:00 UTC
def week_start_sunday_unix(dt_utc):
    offset = (dt_utc.weekday() + 1) % 7
    wk = dt_utc - timedelta(days=offset)
    wk = wk.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(wk.timestamp())


def run_git_analysis(repo_url, model_bundle_path):
    """
    Main entry point:
    - Clones the repo
    - Parses git log
    - Computes features
    - Predicts stages
    """

    # 1. Load Model Bundle
    with open(model_bundle_path, "rb") as f:
        bundle = pickle.load(f)

    # Unpack model parts
    scaler_mean = bundle["scaler"]["mean"]
    scaler_scale = bundle["scaler"]["scale"]
    pca_comps = bundle["pca"]["components"]
    pca_whiten = bundle["pca"]["whiten"]
    pca_explained_var = bundle["pca"]["explained_variance"]
    kmeans_centroids = bundle["kmeans_centroids"]
    label_map = bundle["label_map"]
    features_list = bundle["features"]  # ["commits_8w_sum", ...]

    # 2. Clone Repo to Temp Dir
    temp_dir = tempfile.mkdtemp()
    print(f"[INFO] Cloning {repo_url} into {temp_dir}...")

    try:
        # Clone bare to save space/time
        subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none", "--quiet", repo_url, temp_dir],
            check=True
        )

        # Get all commits timestamps
        # Format: UnixTimestamp (newline separated)
        cmd = ["git", "--git-dir", temp_dir, "log", "--all", "--date-order", "--pretty=%ct"]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        timestamps = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]

        # Get tags/releases (simple approximation via git tags)
        # Format: UnixTimestamp
        cmd_tag = ["git", "--git-dir", temp_dir, "log", "--tags", "--simplify-by-decoration", "--pretty=%at"]
        result_tag = subprocess.run(cmd_tag, capture_output=True, text=True, errors="replace")
        release_timestamps = [int(line.strip()) for line in result_tag.stdout.splitlines() if line.strip()]

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone/log failed: {e}")
    finally:
        # Clean up immediately
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not timestamps:
        raise ValueError("Repository seems empty (no commits found).")

    # 3. Aggregate Weekly
    # Convert timestamps to Sunday-Weeks
    week_counts = {}
    for ts in timestamps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        wk = week_start_sunday_unix(dt)
        week_counts[wk] = week_counts.get(wk, 0) + 1

    rel_counts = {}
    for ts in release_timestamps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        wk = week_start_sunday_unix(dt)
        rel_counts[wk] = rel_counts.get(wk, 0) + 1

    # Create Timeline Grid
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    # Start from the first Sunday
    start_dt = datetime.fromtimestamp(week_start_sunday_unix(datetime.fromtimestamp(min_ts, tz=timezone.utc)),
                                      tz=timezone.utc)
    end_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)

    grid = pd.date_range(start_dt, end_dt, freq="W-SUN", tz="UTC")

    # Build DataFrame
    rows = []
    consec_zero = 0

    # Prepare sliding windows helpers
    # We will compute rolling 8w manually or via pandas
    df_weekly = pd.DataFrame(index=grid)
    df_weekly["week_unix"] = df_weekly.index.astype(np.int64) // 10 ** 9
    df_weekly["commits"] = [week_counts.get(t, 0) for t in df_weekly["week_unix"]]
    df_weekly["releases"] = [rel_counts.get(t, 0) for t in df_weekly["week_unix"]]

    # 03 Logic: Issues & Contributors
    # NOTE: For this lightweight inference, we might NOT fetch Issues/Contributors from API
    # to avoid Token/RateLimit issues on the web app.
    # WE WILL USE PROXIES based on Commits if real data is missing,
    # OR we fill 0. Ideally, for perfect reproduction, we need API calls.
    # BUT: Simplicity first. Let's assume Issues/Contributors ~ correlated or 0 for demo.
    # [BETTER]: Let's infer contributors from git log (authors).
    # Since we did a simple 'log' above, we missed authors.
    # Let's adjust step 2 slightly to get authors if we want better accuracy.
    # For now, to keep it robust: Contributors = Commits (Proxy) / or just 1.
    # Let's fill 0 for Issues (as obtaining them requires API key management which is complex for deployment).
    # Let's set contributors = 1 if commits > 0 else 0 (Minimal viable proxy).

    df_weekly["issues"] = 0  # Placeholder: Issues require API
    df_weekly["contributors"] = (df_weekly["commits"] > 0).astype(int)  # Placeholder

    # Compute 8-week Rolling Features
    df_weekly["commits_8w_sum"] = df_weekly["commits"].rolling(8, min_periods=1).sum()
    df_weekly["contributors_8w_unique"] = df_weekly["contributors"].rolling(8,
                                                                            min_periods=1).sum()  # Sum of active weeks as proxy
    df_weekly["issues_closed_8w_count"] = 0
    df_weekly["releases_8w_count"] = df_weekly["releases"].rolling(8, min_periods=1).sum()

    # 4. Predict
    # Features matrix
    X = df_weekly[features_list].copy()

    # Log1p
    X_log = np.log1p(X)

    # Scale
    X_z = (X_log - scaler_mean) / scaler_scale

    # PCA Project
    # X_pca = (X_z - mean) . components.T
    # Sklearn PCA centers data. StandardScaler already centered it?
    # Wait, StandardScaler centers to 0. PCA assumes centered data.
    # So we simply dot product.
    X_pca = np.dot(X_z, pca_comps.T)

    # Whiten if needed
    if pca_whiten:
        X_pca = X_pca / np.sqrt(pca_explained_var)

    # KMeans Predict
    # Distance to centroids
    # dist = ||x - c||^2
    dist_sq = np.sum((X_pca[:, np.newaxis, :] - kmeans_centroids[np.newaxis, :, :]) ** 2, axis=2)
    clusters = np.argmin(dist_sq, axis=1)

    # Map to Names
    stage_names = [label_map[c] for c in clusters]

    # Apply Dead Rule (24 weeks zero activity)
    # Re-calculate consec zero on weekly data
    final_stages = []
    c_zero = 0
    for i in range(len(df_weekly)):
        is_active = (df_weekly["commits"].iloc[i] > 0) or (df_weekly["releases"].iloc[i] > 0)
        if not is_active:
            c_zero += 1
        else:
            c_zero = 0

        if c_zero >= 24:
            final_stages.append("Dead")
        else:
            final_stages.append(stage_names[i])

    df_weekly["stage_name"] = final_stages

    # Return useful columns for plotting
    return df_weekly.reset_index().rename(columns={"index": "week_date"})