# -*- coding: utf-8 -*-
"""
p01_inference.py (Full V2 Logic)
Objective:
  1. Git clone (bare) -> Extract Commits (Full History) & Authors (for Contributors).
  2. GitHub API -> Extract Issues (exclude PRs) & Releases.
  3. Reconstruct exact v2 8-week rolling features.
  4. Apply PCA+KMeans model for inference.
"""

import os
import shutil
import pickle
import tempfile
import subprocess
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import deque, Counter


# Helper: Align timestamp to the preceding Sunday 00:00 UTC
def week_start_sunday_unix(dt_utc):
    offset = (dt_utc.weekday() + 1) % 7
    wk = dt_utc - timedelta(days=offset)
    wk = wk.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(wk.timestamp())


# API Helper: Fetch all pages from GitHub API with pagination
def fetch_github_api(url, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    results = []
    while url:
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            results.extend(data)
            # Handle pagination via 'link' header
            url = resp.links.get("next", {}).get("url")
        except Exception as e:
            print(f"[WARN] API Fetch Error: {e}")
            break
    return results


def run_git_analysis(repo_url, model_bundle_path, github_token):
    """
    Execute the Full V2 Pipeline:
    1. Clone repo to get Commits and Contributors (Emails).
    2. Query API to get Issues and Releases.
    3. Calculate 8-week rolling metrics aligned with V2 training logic.
    4. Predict lifecycle stages using the pre-trained PCA+KMeans bundle.
    """

    # 1. Load the frozen Model Bundle
    with open(model_bundle_path, "rb") as f:
        bundle = pickle.load(f)

    # Unpack model components
    scaler_mean = bundle["scaler"]["mean"]
    scaler_scale = bundle["scaler"]["scale"]
    pca_comps = bundle["pca"]["components"]
    pca_whiten = bundle["pca"]["whiten"]
    pca_explained_var = bundle["pca"]["explained_variance"]
    kmeans_centroids = bundle["kmeans_centroids"]
    label_map = bundle["label_map"]
    features_list = bundle["features"]

    # Parse Owner/Repo from URL
    if "github.com" not in repo_url:
        raise ValueError("Only GitHub URLs are supported.")
    parts = repo_url.strip("/").split("/")
    owner_repo = f"{parts[-2]}/{parts[-1]}"
    if owner_repo.endswith(".git"):
        owner_repo = owner_repo[:-4]

    # --- PHASE A: Git Clone (Commits & Contributors) ---
    temp_dir = tempfile.mkdtemp()

    commit_timestamps = []
    commit_authors = []  # List of (timestamp, email)

    try:
        # Clone bare repository to save space
        subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none", "--quiet", repo_url, temp_dir],
            check=True
        )

        # Extract Log: Timestamp + Email
        cmd = ["git", "--git-dir", temp_dir, "log", "--all", "--date-order", "--pretty=%ct|%ae"]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

        for line in result.stdout.splitlines():
            if "|" in line:
                ts_str, email = line.split("|", 1)
                ts = int(ts_str.strip())
                email = email.strip().lower()
                commit_timestamps.append(ts)
                commit_authors.append((ts, email))

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not commit_timestamps:
        raise ValueError("No commits found in repository.")

    # --- PHASE B: API Fetch (Issues & Releases) ---
    # Fetch Issues (excluding Pull Requests logic handles downstream)
    issues_url = f"https://api.github.com/repos/{owner_repo}/issues?state=all&per_page=100&sort=created&direction=asc"
    raw_issues = fetch_github_api(issues_url, github_token)

    # Filter out PRs and extract closed dates
    issue_closed_timestamps = []
    for item in raw_issues:
        # V2 Logic: Explicitly exclude PRs, only count closed issues
        if "pull_request" not in item and item["state"] == "closed" and item["closed_at"]:
            dt = datetime.fromisoformat(item["closed_at"].replace("Z", "+00:00"))
            issue_closed_timestamps.append(int(dt.timestamp()))

    # Fetch Releases
    releases_url = f"https://api.github.com/repos/{owner_repo}/releases?per_page=100"
    raw_releases = fetch_github_api(releases_url, github_token)
    release_timestamps = []
    for item in raw_releases:
        if item["published_at"]:
            dt = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
            release_timestamps.append(int(dt.timestamp()))

    # --- PHASE C: Weekly Aggregation (Aligned with V2) ---
    min_ts = min(commit_timestamps)
    max_ts = datetime.now(timezone.utc).timestamp()  # Calculate up to current date

    start_dt = datetime.fromtimestamp(week_start_sunday_unix(datetime.fromtimestamp(min_ts, timezone.utc)),
                                      timezone.utc)
    end_dt = datetime.fromtimestamp(max_ts, timezone.utc)
    grid = pd.date_range(start_dt, end_dt, freq="W-SUN", tz="UTC")

    # Prepare Buckets
    week_commits = Counter()
    week_issues = Counter()
    week_releases = Counter()
    week_authors = {dt: set() for dt in grid}  # For unique contributors calculation

    # Bucket Commits
    for ts in commit_timestamps:
        dt = datetime.fromtimestamp(ts, timezone.utc)
        wk = week_start_sunday_unix(dt)
        week_commits[wk] += 1

    # Bucket Authors (for Contributors feature)
    for ts, email in commit_authors:
        dt = datetime.fromtimestamp(ts, timezone.utc)
        wk_ts = week_start_sunday_unix(dt)
        # Convert back to grid timestamp key match
        wk_dt = pd.Timestamp(wk_ts, unit="s", tz="UTC")
        if wk_dt in week_authors:
            week_authors[wk_dt].add(email)

    # Bucket Issues
    for ts in issue_closed_timestamps:
        dt = datetime.fromtimestamp(ts, timezone.utc)
        wk = week_start_sunday_unix(dt)
        week_issues[wk] += 1

    # Bucket Releases
    for ts in release_timestamps:
        dt = datetime.fromtimestamp(ts, timezone.utc)
        wk = week_start_sunday_unix(dt)
        week_releases[wk] += 1

    # Build DataFrame
    df = pd.DataFrame(index=grid)
    df["week_unix"] = df.index.astype(np.int64) // 10 ** 9

    # Map raw counts to the weekly grid
    df["commits"] = df["week_unix"].map(week_commits).fillna(0)
    df["issues"] = df["week_unix"].map(week_issues).fillna(0)
    df["releases"] = df["week_unix"].map(week_releases).fillna(0)

    # --- PHASE D: 8-Week Rolling Features (V2 Logic) ---
    # 1. Numeric Sums (Rolling 8 weeks)
    df["commits_8w_sum"] = df["commits"].rolling(8, min_periods=1).sum()
    df["issues_closed_8w_count"] = df["issues"].rolling(8, min_periods=1).sum()
    df["releases_8w_count"] = df["releases"].rolling(8, min_periods=1).sum()

    # 2. Contributors (Sliding Union of Sets over 8 weeks)
    u8 = []
    dq = deque()
    cnt = Counter()

    for dt in grid:
        cur_set = week_authors.get(dt, set())
        dq.append(cur_set)
        for author in cur_set:
            cnt[author] += 1

        # Remove authors that fell out of the 8-week window
        if len(dq) > 8:
            out_set = dq.popleft()
            for author in out_set:
                cnt[author] -= 1
                if cnt[author] <= 0:
                    del cnt[author]

        u8.append(len(cnt))

    df["contributors_8w_unique"] = u8

    # --- PHASE E: Prediction (Inference) ---
    # Prepare Feature Matrix
    X = df[features_list].copy()

    # Preprocessing: Log1p -> Robust/Standard Scaling
    X_log = np.log1p(X)
    X_z = (X_log - scaler_mean) / scaler_scale

    # PCA Projection
    X_pca = np.dot(X_z, pca_comps.T)

    # Apply Whitening if enabled in the trained model
    if pca_whiten:
        X_pca = X_pca / np.sqrt(pca_explained_var)

    # KMeans Assignment (Nearest Centroid)
    dist_sq = np.sum((X_pca[:, np.newaxis, :] - kmeans_centroids[np.newaxis, :, :]) ** 2, axis=2)
    clusters = np.argmin(dist_sq, axis=1)
    stage_names = [label_map[c] for c in clusters]

    # Apply "Dead" Rule (24 weeks of zero activity)
    # This aligns with V1/V2 post-processing logic
    final_stages = []
    c_zero = 0
    for i in range(len(df)):
        is_active = (df["commits"].iloc[i] > 0) or (df["releases"].iloc[i] > 0)
        if not is_active:
            c_zero += 1
        else:
            c_zero = 0

        if c_zero >= 24:
            final_stages.append("Dead")
        else:
            final_stages.append(stage_names[i])

    df["stage_name"] = final_stages

    # Return result with a clean date column for plotting
    return df.reset_index().rename(columns={"index": "week_date"})