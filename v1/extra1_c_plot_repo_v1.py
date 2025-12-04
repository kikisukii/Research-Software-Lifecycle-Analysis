#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extra1_c_plot_repo_v1.py


This one is for me to check how it shows on weekly.

Objective:
  Visualize the life cycle of randomly sampled repositories based on v1 labels.
  - X-axis: Time (Weeks)
  - Y-axis: Commits (from 02_gh_commit_weekly)
  - Background Color: 6 Distinct Life Cycle Stages (Custom Color Scheme V3).

Updates:
  - Title changed to "Life Activity (v1): owner/repo"
  - New Color Scheme applied (Higher contrast set).

Color Scheme (User Defined V3):
  - Initial:     Golden Orange (#f8b862)
  - Rising:      Soft Green    (#88cb7f)
  - Peak:        Jade Green    (#38b48b)
  - Maintenance: Sky Blue      (#a0d8ef)
  - Low:         Lead Gray     (#7b7c7d)
  - Dead:        Dark Charcoal (#383c3c)

Usage:
  python extra1_c_plot_repo_v1.py --sample 10
"""

import argparse
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# [CONFIG] Path Logic
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V1_DATA_ROOT = PROJECT_ROOT / "v1_data"
LABEL_DIR = V1_DATA_ROOT / "05_b_apply"
COMMIT_DIR = V1_DATA_ROOT / "02_dat"
OUTPUT_DIR = V1_DATA_ROOT / "05_c_viz"

STAMP_RE = re.compile(r"(\d{8}_\d{6})")

# [CONFIG] Visuals
SMOOTH_WINDOW = 5  # Window size for smoothing curve
PLOT_ORIGINAL = False  # If True, overlay raw noisy line
MIN_WIDTH, MAX_WIDTH = 8, 24  # Plot width constraints (inches)
WIDTH_SLOPE = 0.04  # Width per week

# Custom Color Map (Updated V3)
STAGE_COLOR = {
    "Initial": "#f8b862",  # Golden Orange
    "Rising": "#88cb7f",  # Soft Green
    "Peak": "#38b48b",  # Jade Green
    "Maintenance": "#a0d8ef",  # Sky Blue
    "Low": "#7b7c7d",  # Lead Gray
    "Dead": "#383c3c",  # Dark Charcoal
}


def infer_stamp(p: Path) -> str:
    m = STAMP_RE.search(p.name)
    if not m:
        return "00000000_000000"
    return m.group(1)


def find_latest_labels() -> Path:
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")
    cands = list(LABEL_DIR.glob("05_labels_v1_*.csv"))
    if not cands:
        raise FileNotFoundError("No label CSVs found in 05_b_apply.")
    return sorted(cands, key=lambda p: infer_stamp(p))[-1]


def to_utc(series_unix):
    return pd.to_datetime(series_unix, unit="s", utc=True)


def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("\\", "__")


def build_segments(x_ts, stages):
    """
    Constructs continuous time segments for background coloring.
    Returns [(t0, t1, stage), ...]
    """
    segs = []
    if len(x_ts) == 0: return segs
    start = 0
    for i in range(1, len(x_ts)):
        if stages[i] != stages[i - 1]:
            segs.append((x_ts[start], x_ts[i], stages[i - 1]))
            start = i
    segs.append((x_ts[start], x_ts[-1], stages[-1]))
    return segs


def smooth_series(y: pd.Series, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y.values.astype(float)
    return y.astype(float).rolling(window=window, center=True, min_periods=1).mean().values


def compute_figsize(ts_min: int, ts_max: int, height: float = 4.8):
    weeks = max(1, int(round((ts_max - ts_min) / (7 * 24 * 3600))) + 1)
    width = 4 + weeks * WIDTH_SLOPE
    width = max(MIN_WIDTH, min(MAX_WIDTH, width))
    return (width, height)


def plot_one(repo, df_repo: pd.DataFrame, outdir: Path):
    # Ensure sorted by time
    df_repo = df_repo.sort_values("week_unix").reset_index(drop=True)
    x = to_utc(df_repo["week_unix"])

    # Y-axis: Commits
    if "commits" not in df_repo.columns:
        print(f"[WARN] No commit data for {repo}, skipping plot.")
        return

    y = df_repo["commits"].fillna(0).astype(float)
    y_smooth = smooth_series(y, SMOOTH_WINDOW)

    # Dynamic Figure Size
    w, h = compute_figsize(int(df_repo["week_unix"].min()), int(df_repo["week_unix"].max()))
    fig, ax = plt.subplots(figsize=(w, h))

    # Plot Lines
    if PLOT_ORIGINAL:
        ax.plot(x, y, lw=1.0, alpha=0.35, color='#bdc3c7', label="Commits (Raw)")

    # Main Line (Dark Blue)
    ax.plot(x, y_smooth, lw=1.8, color='#2c3e50', label="Commits (Smoothed)")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Weekly Commits")

    # Title: "Life Activity (v1): owner/repo"
    ax.set_title(f"Life Activity (v1): {repo}", fontsize=12, fontweight='bold')

    # Background Colors (Stages)
    stages = df_repo["stage"].fillna("Unknown").values
    segs = build_segments(df_repo["week_unix"].values, stages)

    for (ts0, ts1, stage) in segs:
        # Default to white if stage is unknown
        color = STAGE_COLOR.get(stage, "#ffffff")
        # Alpha=0.4 allows the color to be visible but not cover the line
        ax.axvspan(to_utc([ts0])[0], to_utc([ts1])[0],
                   facecolor=color, alpha=0.4, edgecolor=None)

    # Formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, axis="y", alpha=0.3, linestyle='--')

    # Legend
    # Strictly use the keys from STAGE_COLOR
    legend_patches = [Patch(facecolor=c, edgecolor='black', linewidth=0.5, alpha=0.4, label=s)
                      for s, c in STAGE_COLOR.items()]

    leg1 = ax.legend(handles=legend_patches, loc="upper left", frameon=True, fontsize=9, title="Stages")
    ax.add_artist(leg1)

    # Line Legend
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # Save
    safe_r = safe_name(repo)
    fn = outdir / f"repo_{safe_r}.png"
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f"[OK] Plot saved: {fn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5, help="Number of random repos to plot")
    parser.add_argument("--repo", type=str, default=None, help="Specific repo name to plot (overrides sample)")
    args = parser.parse_args()

    # 1. Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Find Data
    labels_path = find_latest_labels()
    stamp = infer_stamp(labels_path)
    print(f"[INFO] Using Labels: {labels_path.name}")

    # Commits file (Raw counts for Y-axis)
    commits_path = COMMIT_DIR / f"02_gh_commit_weekly_{stamp}.csv"
    if not commits_path.exists():
        # Fallback
        try:
            commits_path = list(COMMIT_DIR.glob(f"*commit_weekly*{stamp}*.csv"))[0]
        except IndexError:
            raise FileNotFoundError(f"Commit data not found for stamp {stamp}")

    print(f"[INFO] Using Commits: {commits_path.name}")

    # 3. Load Data
    df_labels = pd.read_csv(labels_path)
    df_commits = pd.read_csv(commits_path)

    # 4. Select Repos
    all_repos = df_labels["repo"].unique()

    if args.repo:
        if args.repo not in all_repos:
            print(f"[Error] Repo '{args.repo}' not found in labels.")
            return
        target_repos = [args.repo]
    else:
        n_sample = min(args.sample, len(all_repos))
        rng = np.random.default_rng()
        target_repos = rng.choice(all_repos, size=n_sample, replace=False)
        print(f"[INFO] Randomly selected {n_sample} repositories.")

    # 5. Plot Loop
    for repo in target_repos:
        # Get labels
        sub_labels = df_labels[df_labels["repo"] == repo].copy()

        # Get commits
        sub_commits = df_commits[df_commits["repo"] == repo].copy()

        # Merge
        df_plot = pd.merge(sub_labels, sub_commits[["repo", "week_unix", "commits"]],
                           on=["repo", "week_unix"], how="left")

        # Plot
        plot_one(repo, df_plot, OUTPUT_DIR)


if __name__ == "__main__":
    main()