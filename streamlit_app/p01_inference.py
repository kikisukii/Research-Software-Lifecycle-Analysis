# -*- coding: utf-8 -*-
"""
p01_inference.py (Full V2 Logic + Raw Weekly Contributors)

Objective:
  1. Git clone (bare) -> Extract Commits & Authors.
  2. GitHub API -> Extract Issues & Releases.
  3. Reconstruct exact v2 8-week rolling features.
  4. Apply PCA + KMeans model for inference.
"""

import shutil
import pickle
import tempfile
import subprocess
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone
from collections import deque, Counter


def week_start_sunday_unix(dt_utc):
    """
    Convert a UTC datetime to the Unix timestamp of
    Sunday 00:00 UTC for that week.
    """
    offset = (dt_utc.weekday() + 1) % 7

    wk = dt_utc - timedelta(days=offset)

    wk = wk.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    return int(wk.timestamp())


def fetch_github_api(url, token):
    """
    Fetch all paginated results from GitHub REST API.

    Important:
    API errors are NOT converted to empty data.
    Otherwise an API failure could be incorrectly interpreted
    as zero Issues or zero Releases.
    """

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "research-software-lifecycle-detector"
    }

    results = []

    while url:
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            print(
                "[GitHub API]",
                "status=",
                resp.status_code,
                "remaining=",
                resp.headers.get("X-RateLimit-Remaining"),
                "url=",
                url
            )

            if resp.status_code == 404:
                raise RuntimeError(
                    f"GitHub API returned 404 for: {url}"
                )

            resp.raise_for_status()

            data = resp.json()

            if not data:
                break

            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected GitHub API response format for: {url}"
                )

            results.extend(data)

            url = resp.links.get(
                "next",
                {}
            ).get("url")

        except Exception as e:
            raise RuntimeError(
                f"GitHub API fetch failed for {url}: {e}"
            ) from e

    return results


def run_git_analysis(
    repo_url,
    model_bundle_path,
    github_token
):
    # ---------------------------------------------------------
    # PHASE 0: Load Model
    # ---------------------------------------------------------

    with open(model_bundle_path, "rb") as f:
        bundle = pickle.load(f)

    scaler_mean = np.asarray(
        bundle["scaler"]["mean"]
    )

    scaler_scale = np.asarray(
        bundle["scaler"]["scale"]
    )

    pca_comps = np.asarray(
        bundle["pca"]["components"]
    )

    pca_whiten = bundle["pca"]["whiten"]

    pca_explained_var = np.asarray(
        bundle["pca"]["explained_variance"]
    )

    kmeans_centroids = np.asarray(
        bundle["kmeans_centroids"]
    )

    label_map = bundle["label_map"]

    features_list = bundle["features"]

    # ---------------------------------------------------------
    # Parse Owner / Repo
    # ---------------------------------------------------------

    if "github.com" not in repo_url:
        raise ValueError(
            "Only GitHub URLs are supported."
        )

    clean_url = repo_url.strip().rstrip("/")

    parts = clean_url.split("/")

    if len(parts) < 2:
        raise ValueError(
            "Invalid GitHub repository URL."
        )

    owner_repo = f"{parts[-2]}/{parts[-1]}"

    if owner_repo.endswith(".git"):
        owner_repo = owner_repo[:-4]

    print(
        "[DEBUG] Repository:",
        owner_repo
    )

    # ---------------------------------------------------------
    # PHASE A: Git Clone
    # ---------------------------------------------------------

    temp_dir = tempfile.mkdtemp()

    commit_timestamps = []
    commit_authors = []

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--bare",
                "--filter=blob:none",
                "--quiet",
                clean_url,
                temp_dir
            ],
            check=True,
            capture_output=True,
            text=True
        )

        cmd = [
            "git",
            "--git-dir",
            temp_dir,
            "log",
            "--all",
            "--date-order",
            "--pretty=%ct|%ae"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            check=True
        )

        for line in result.stdout.splitlines():
            if "|" not in line:
                continue

            ts_str, email = line.split("|", 1)

            ts_str = ts_str.strip()
            email = email.strip().lower()

            if not ts_str:
                continue

            try:
                ts = int(ts_str)
            except ValueError:
                continue

            commit_timestamps.append(ts)

            commit_authors.append(
                (ts, email)
            )

    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""

        raise RuntimeError(
            f"Git clone/log failed: {stderr}"
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Git clone failed: {e}"
        ) from e

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

    if not commit_timestamps:
        raise ValueError(
            "No commits found."
        )

    print(
        "[DEBUG] Git commits found:",
        len(commit_timestamps)
    )

    # ---------------------------------------------------------
    # PHASE B: GitHub API Fetch
    # ---------------------------------------------------------

    issues_url = (
        f"https://api.github.com/repos/"
        f"{owner_repo}/issues"
        f"?state=all"
        f"&per_page=100"
        f"&sort=created"
        f"&direction=asc"
    )

    raw_issues = fetch_github_api(
        issues_url,
        github_token
    )

    issue_closed_timestamps = []

    for item in raw_issues:
        # GitHub /issues endpoint also includes pull requests.
        if "pull_request" in item:
            continue

        if item.get("state") != "closed":
            continue

        closed_at = item.get("closed_at")

        if not closed_at:
            continue

        try:
            dt = datetime.fromisoformat(
                closed_at.replace(
                    "Z",
                    "+00:00"
                )
            )

            issue_closed_timestamps.append(
                int(dt.timestamp())
            )

        except Exception:
            print(
                "[WARN] Could not parse issue closed_at:",
                closed_at
            )

    releases_url = (
        f"https://api.github.com/repos/"
        f"{owner_repo}/releases"
        f"?per_page=100"
    )

    raw_releases = fetch_github_api(
        releases_url,
        github_token
    )

    release_timestamps = []

    for item in raw_releases:
        published_at = item.get(
            "published_at"
        )

        # Fallback for unusual release objects.
        if not published_at:
            published_at = item.get(
                "created_at"
            )

        if not published_at:
            continue

        try:
            dt = datetime.fromisoformat(
                published_at.replace(
                    "Z",
                    "+00:00"
                )
            )

            release_timestamps.append(
                int(dt.timestamp())
            )

        except Exception:
            print(
                "[WARN] Could not parse release date:",
                published_at
            )

    print(
        "[DEBUG] Closed issues found:",
        len(issue_closed_timestamps)
    )

    print(
        "[DEBUG] Releases found:",
        len(release_timestamps)
    )

    # ---------------------------------------------------------
    # PHASE C: Weekly Aggregation
    # ---------------------------------------------------------

    min_ts = min(
        commit_timestamps
    )

    max_ts = datetime.now(
        timezone.utc
    ).timestamp()

    first_commit_dt = datetime.fromtimestamp(
        min_ts,
        timezone.utc
    )

    start_week_unix = week_start_sunday_unix(
        first_commit_dt
    )

    start_dt = datetime.fromtimestamp(
        start_week_unix,
        timezone.utc
    )

    end_dt = datetime.fromtimestamp(
        max_ts,
        timezone.utc
    )

    grid = pd.date_range(
        start=start_dt,
        end=end_dt,
        freq="W-SUN",
        tz="UTC"
    )

    if len(grid) == 0:
        raise RuntimeError(
            "Weekly date grid is empty."
        )

    week_commits = Counter()

    week_issues = Counter()

    week_releases = Counter()

    week_authors = {
        dt: set()
        for dt in grid
    }

    # ---------------------------------------------------------
    # Weekly commits
    # ---------------------------------------------------------

    for ts in commit_timestamps:
        dt = datetime.fromtimestamp(
            ts,
            timezone.utc
        )

        wk = week_start_sunday_unix(
            dt
        )

        week_commits[wk] += 1

    # ---------------------------------------------------------
    # Weekly contributors
    # ---------------------------------------------------------

    for ts, email in commit_authors:
        dt = datetime.fromtimestamp(
            ts,
            timezone.utc
        )

        wk_ts = week_start_sunday_unix(
            dt
        )

        wk_dt = pd.Timestamp(
            wk_ts,
            unit="s",
            tz="UTC"
        )

        if wk_dt in week_authors:
            week_authors[wk_dt].add(
                email
            )

    # ---------------------------------------------------------
    # Weekly closed issues
    # ---------------------------------------------------------

    for ts in issue_closed_timestamps:
        dt = datetime.fromtimestamp(
            ts,
            timezone.utc
        )

        wk = week_start_sunday_unix(
            dt
        )

        week_issues[wk] += 1

    # ---------------------------------------------------------
    # Weekly releases
    # ---------------------------------------------------------

    for ts in release_timestamps:
        dt = datetime.fromtimestamp(
            ts,
            timezone.utc
        )

        wk = week_start_sunday_unix(
            dt
        )

        week_releases[wk] += 1

    # ---------------------------------------------------------
    # Build DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(
        index=grid
    )

    # IMPORTANT:
    #
    # Do NOT use:
    #
    # df.index.astype(np.int64) // 10**9
    #
    # Pandas datetime resolution may be ns, us, etc.
    # Timestamp.timestamp() always gives Unix seconds.
    #
    # This prevents commits/issues/releases from becoming
    # all zero because of timestamp-unit mismatch.

    df["week_unix"] = np.array(
        [
            int(ts.timestamp())
            for ts in df.index
        ],
        dtype=np.int64
    )

    df["commits"] = (
        df["week_unix"]
        .map(week_commits)
        .fillna(0)
        .astype(int)
    )

    df["issues"] = (
        df["week_unix"]
        .map(week_issues)
        .fillna(0)
        .astype(int)
    )

    df["releases"] = (
        df["week_unix"]
        .map(week_releases)
        .fillna(0)
        .astype(int)
    )

    df["contributors"] = [
        len(
            week_authors.get(
                dt,
                set()
            )
        )
        for dt in df.index
    ]

    # ---------------------------------------------------------
    # Debug / Integrity checks
    # ---------------------------------------------------------

    print(
        "[DEBUG] Pandas grid dtype:",
        df.index.dtype
    )

    print(
        "[DEBUG] Total weekly commits:",
        int(
            df["commits"].sum()
        )
    )

    print(
        "[DEBUG] Total contributor-week count:",
        int(
            df["contributors"].sum()
        )
    )

    print(
        "[DEBUG] Total weekly closed issues:",
        int(
            df["issues"].sum()
        )
    )

    print(
        "[DEBUG] Total weekly releases:",
        int(
            df["releases"].sum()
        )
    )

    if week_commits:
        print(
            "[DEBUG] Example commit Counter key:",
            next(
                iter(
                    week_commits.keys()
                )
            )
        )

    if len(df) > 0:
        print(
            "[DEBUG] Example dataframe week_unix:",
            int(
                df["week_unix"].iloc[0]
            )
        )

    # Contributors and commits are extracted from the
    # same Git history.
    #
    # Therefore contributors > 0 with commits == 0
    # indicates an aggregation/alignment failure.

    if (
        df["contributors"].sum() > 0
        and
        df["commits"].sum() == 0
    ):
        raise RuntimeError(
            "Data integrity error: contributors were detected, "
            "but aggregated commit count is zero. "
            "Check weekly timestamp alignment."
        )

    # ---------------------------------------------------------
    # PHASE D: 8-Week Rolling Features
    # ---------------------------------------------------------

    df["commits_8w_sum"] = (
        df["commits"]
        .rolling(
            8,
            min_periods=1
        )
        .sum()
    )

    df["issues_closed_8w_count"] = (
        df["issues"]
        .rolling(
            8,
            min_periods=1
        )
        .sum()
    )

    df["releases_8w_count"] = (
        df["releases"]
        .rolling(
            8,
            min_periods=1
        )
        .sum()
    )

    # ---------------------------------------------------------
    # Contributors: Sliding 8-week union
    # ---------------------------------------------------------

    u8 = []

    dq = deque()

    cnt = Counter()

    for dt in grid:
        cur_set = week_authors.get(
            dt,
            set()
        )

        dq.append(
            cur_set
        )

        for author in cur_set:
            cnt[author] += 1

        if len(dq) > 8:
            out_set = dq.popleft()

            for author in out_set:
                cnt[author] -= 1

                if cnt[author] <= 0:
                    del cnt[author]

        u8.append(
            len(cnt)
        )

    df["contributors_8w_unique"] = u8

    # ---------------------------------------------------------
    # PHASE E: Prediction
    # ---------------------------------------------------------

    missing_features = [
        feature
        for feature in features_list
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Missing model features: "
            + ", ".join(
                missing_features
            )
        )

    X = df[
        features_list
    ].copy()

    X_log = np.log1p(
        X
    )

    X_z = (
        X_log
        - scaler_mean
    ) / scaler_scale

    X_pca = np.dot(
        X_z,
        pca_comps.T
    )

    if pca_whiten:
        X_pca = (
            X_pca
            / np.sqrt(
                pca_explained_var
            )
        )

    dist_sq = np.sum(
        (
            X_pca[:, np.newaxis, :]
            -
            kmeans_centroids[
                np.newaxis,
                :,
                :
            ]
        ) ** 2,
        axis=2
    )

    clusters = np.argmin(
        dist_sq,
        axis=1
    )

    stage_names = [
        label_map[c]
        for c in clusters
    ]

    # ---------------------------------------------------------
    # Dead heuristic:
    # 24 consecutive weeks without commits or releases
    # ---------------------------------------------------------

    final_stages = []

    c_zero = 0

    for i in range(
        len(df)
    ):
        is_active = (
            df["commits"].iloc[i] > 0
            or
            df["releases"].iloc[i] > 0
        )

        if not is_active:
            c_zero += 1
        else:
            c_zero = 0

        if c_zero >= 24:
            final_stages.append(
                "Dead"
            )
        else:
            final_stages.append(
                stage_names[i]
            )

    df["stage_name"] = final_stages

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------

    df = (
        df
        .reset_index()
        .rename(
            columns={
                "index": "week_date"
            }
        )
    )

    return df