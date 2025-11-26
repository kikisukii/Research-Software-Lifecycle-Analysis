# -*- coding: utf-8 -*-
"""
v1/05_c_v1_8w.py — V1 Plotting aligned with V2 Style & Colors.

Objective:
  Visualize V1 results using the EXACT same visual standards and COLOR PALETTE as V2.
  This ensures that visual differences are due to the model's logic, not the color scheme.

  Mapping:
  - V1 Initial     -> V2 Baseline (#f8b862)
  - V1 Rising      -> V2 Internal Development (#38b48b)
  - V1 Peak        -> V2 Peak Activity (#e9546b)
  - V1 Maintenance -> V2 Maintenance (#89c3eb)
  - V1 Low         -> V2 Dormant (#9ea1a3)
  - V1 Dead        -> V2 Dead (#383c3c)
"""

import argparse
import re
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# [CONFIG] Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V1_DATA_ROOT = PROJECT_ROOT / "v1_data"
LABEL_DIR = V1_DATA_ROOT / "05_b_apply"
COMMIT_DIR = V1_DATA_ROOT / "02_dat"
OUTPUT_DIR = V1_DATA_ROOT / "05_c_viz_8w"

STAMP_RE = re.compile(r"(\d{8}_\d{6})")

# [CONFIG] Visuals - V1 Colors Mapped to V2 Palette
STAGE_COLOR = {
    "Initial": "#f8b862",  # Matches V2 Baseline (Orange)
    "Rising": "#38b48b",  # Matches V2 Internal Development (Green)
    "Peak": "#e9546b",  # Matches V2 Peak Activity (Red/Pink)
    "Maintenance": "#89c3eb",  # Matches V2 Maintenance (Blue)
    "Low": "#9ea1a3",  # Matches V2 Dormant (Grey)
    "Dead": "#383c3c",  # Matches V2 Dead (Dark)
}

# Define legend order for consistency
LEGEND_ORDER = ["Initial", "Rising", "Peak", "Maintenance", "Low", "Dead"]

# Plotting constants matched with V2
MIN_WIDTH, MAX_WIDTH = 8.0, 22.0
WIDTH_SLOPE = 0.04


def infer_stamp(p: Path) -> str:
    m = STAMP_RE.search(p.name)
    return m.group(1) if m else "000000"


def find_latest_labels() -> Path:
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label dir not found: {LABEL_DIR}")
    cands = sorted(LABEL_DIR.glob("05_labels_v1_*.csv"), key=lambda p: infer_stamp(p))
    if not cands:
        raise FileNotFoundError("No V1 labels found.")
    return cands[-1]


def to_utc_dt(series_unix):
    return pd.to_datetime(series_unix, unit="s", utc=True)


def compute_figsize(week_min: int, week_max: int, height: float = 5.0):
    """Same logic as V2 to ensure time-axis scale is comparable."""
    weeks = max(1, int(round((week_max - week_min) / (7 * 24 * 3600))) + 1)
    width = max(MIN_WIDTH, min(MAX_WIDTH, 4 + weeks * WIDTH_SLOPE))
    return (width, height)


def build_segments(x_ts, stages):
    segs = []
    if len(x_ts) == 0: return segs
    start = 0
    for i in range(1, len(x_ts)):
        if stages[i] != stages[i - 1]:
            segs.append((x_ts[start], x_ts[i], stages[i - 1]))
            start = i
    segs.append((x_ts[start], x_ts[-1], stages[-1]))
    return segs


def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("\\", "__")


def plot_one_8w(repo, df_repo: pd.DataFrame, outdir: Path):
    # 1. Prepare Data
    df_repo = df_repo.sort_values("week_unix").reset_index(drop=True)
    x_unix = df_repo["week_unix"].values
    x = to_utc_dt(df_repo["week_unix"])

    # Calculate 8-week rolling sum (Simulating V2 Feature Logic)
    y_raw = df_repo["commits"].fillna(0)
    y_8w = y_raw.rolling(window=8, min_periods=1).sum()

    # 2. Setup Figure (Dynamic Size)
    w, h = compute_figsize(int(x_unix.min()), int(x_unix.max()))
    fig, ax = plt.subplots(figsize=(w, h))

    # 3. Plot Background (Phases)
    stages = df_repo["stage"].fillna("Unknown").values
    segs = build_segments(x_unix, stages)
    for (ts0, ts1, stage) in segs:
        c = STAGE_COLOR.get(stage, "#eeeeee")
        # Alpha matched to V2
        ax.axvspan(to_utc_dt([ts0])[0], to_utc_dt([ts1])[0],
                   facecolor=c, alpha=0.35, edgecolor=None)

    # 4. Plot Line (Style matched to V2)
    # V2 uses dark grey line, lw=1.8
    ax.plot(x, y_8w, lw=1.8, color='#333333', label="Commits (8-week rolling)")

    # Optional: Grid
    ax.grid(True, axis="y", alpha=0.3)

    # 5. Titles and Labels
    ax.set_title(f"Life Activity (v1): {repo}", fontsize=14, y=1.02)
    ax.set_ylabel("Commits (8-week rolling)", fontsize=10)
    ax.set_xlabel("Week (UTC)")

    # 6. Formatting X-Axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # 7. Legend (Explicit Order)
    handles = []
    # Add Stage patches
    for k in LEGEND_ORDER:
        if k in STAGE_COLOR:
            handles.append(Patch(facecolor=STAGE_COLOR[k], edgecolor="none", alpha=0.6, label=k))

    # Add any missing stages found in data (e.g., Unknown)
    existing = set(stages)
    for st in existing:
        if st not in LEGEND_ORDER and st != "Unknown":
            handles.append(Patch(facecolor="#eeeeee", edgecolor="black", label=st))

    ax.legend(handles=handles, loc="upper right", fontsize=9, title="Stages (V1)", frameon=True)

    # 8. Save
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"repo_{safe_name(repo)}.png"
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f"[OK] Plot saved: {fn.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--repo", action="append", help="Specific repo(s) to plot")
    args = parser.parse_args()

    # Load Data
    labels_path = find_latest_labels()
    stamp = infer_stamp(labels_path)
    print(f"[INFO] V1 Labels: {labels_path.name}")

    # Find commits file (fuzzy search for stamp)
    try:
        commits_path = list(COMMIT_DIR.glob(f"*commit_weekly*{stamp}*.csv"))[0]
    except IndexError:
        print(f"[ERR] Could not find commit data for stamp {stamp} in {COMMIT_DIR}")
        return

    print(f"[INFO] V1 Commits: {commits_path.name}")

    df_labels = pd.read_csv(labels_path)
    df_commits = pd.read_csv(commits_path)

    # Filter Repos
    all_repos = sorted(df_labels["repo"].unique())

    if args.repo:
        targets = [r for r in args.repo if r in all_repos]
    else:
        rng = np.random.default_rng(42)  # Consistent seed with V2
        n = min(args.sample, len(all_repos))
        targets = sorted(rng.choice(all_repos, size=n, replace=False).tolist())
        print(f"[INFO] Randomly sampling {n} repos")

    # Loop
    for repo in targets:
        # Join labels with commits
        sub_lab = df_labels[df_labels["repo"] == repo].copy()
        sub_com = df_commits[df_commits["repo"] == repo].copy()

        # Merge
        if sub_com.empty:
            print(f"[WARN] No commit data for {repo}")
            continue

        df_plot = pd.merge(sub_lab, sub_com[["repo", "week_unix", "commits"]],
                           on=["repo", "week_unix"], how="left")

        plot_one_8w(repo, df_plot, OUTPUT_DIR)


if __name__ == "__main__":
    main()