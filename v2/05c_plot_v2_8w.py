# -*- coding: utf-8 -*-
"""
v2/05c_plot_repo_v2.py — Per-repo multi-metric timeline with stage-colored background.

UPDATES:
  - Title format: "Life Activity(v2): owner/repo"
  - Y-axis: Uses 8-week rolling data (Model Input).
  - Visualization Fix: Data points aligned to MID-WEEK (+3.5 days) to align properly with background phases.
  - Smoothing: Default reduced to 3.
"""

from __future__ import annotations
from pathlib import Path
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ---------- paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
D05_B = ROOT / "v2_data" / "05_b_apply"
D05_C = ROOT / "v2_data" / "05_c_viz"

# ---------- config ----------
# 1. Custom Colors
# --- Colors (V2 Standard) ---
STAGE_COLORS = {
    "Baseline": "#f8b862",
    "Internal Development": "#19448e", #"#38b48b"原来的绿色
    "Release Phase": "#9d5b8b",
    "Peak Activity": "#38b48b", ##e9546b 原来的珊瑚红
    "Maintenance": "#89c3eb",
    "Dormant": "#9ea1a3",
    "Dead": "#383c3c",
}

# 2. Legend Order
LEGEND_ORDER = [
    "Baseline",
    "Internal Development",
    "Release Phase",
    "Peak Activity",
    "Maintenance",
    "Dormant",
    "Dead"
]

# 3. Metrics (8-week rolling)
METRICS = [
    ("commits_8w_sum", "Commits (8-week rolling)"),
    ("contributors_8w_unique", "Contributors (8-week rolling)"),
    ("issues_closed_8w_count", "Issues Closed (8-week rolling)"),
    ("releases_8w_count", "Releases (8-week rolling)"),
]

MIN_WIDTH, MAX_WIDTH = 8.0, 22.0
WIDTH_SLOPE = 0.04

STAMP_RE = re.compile(r"_v2_(\d{8}_\d{6})_K\d+\.csv$")


# ---------- utils ----------
def infer_latest_assignments() -> Path:
    # Prefer the file that includes DEAD rows
    cands_dead = sorted(D05_B.glob("05b_assignments_with_stage_AND_DEAD_v2_*.csv"))
    if cands_dead:
        return cands_dead[-1]

    cands_main = sorted(D05_B.glob("05b_assignments_with_stage_v2_*.csv"))
    if not cands_main:
        raise FileNotFoundError("No 05b assignments found under v2_data/05_b_apply/")
    return cands_main[-1]


def infer_stamp_from_name(p: Path) -> str:
    m = STAMP_RE.search(p.name)
    if not m:
        m2 = re.search(r"(\d{8}_\d{6})", p.name)
        if not m2:
            return "unknown"
        return m2.group(1)
    return m.group(1)


def to_utc_dt(unix_s: pd.Series) -> pd.Series:
    return pd.to_datetime(unix_s, unit="s", utc=True)


def compute_figsize(week_min: int, week_max: int, nrows: int = 4, row_h: float = 2.0):
    weeks = max(1, int(round((week_max - week_min) / (7 * 24 * 3600))) + 1)
    width = max(MIN_WIDTH, min(MAX_WIDTH, 4 + weeks * WIDTH_SLOPE))
    height = max(5.0, nrows * row_h + 1.2)
    return (width, height)


def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return arr.astype(float)
    s = pd.Series(arr.astype(float))
    return s.rolling(window=window, center=True, min_periods=1).mean().values


def dedupe_by_week(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["repo", "week_unix"]).drop_duplicates(["repo", "week_unix"], keep="first")


def build_segments(week_unix: np.ndarray, stages: np.ndarray):
    segs = []
    if len(week_unix) == 0:
        return segs
    start = 0
    for i in range(1, len(week_unix)):
        if stages[i] != stages[i - 1]:
            segs.append((week_unix[start], week_unix[i], stages[i - 1]))
            start = i
    segs.append((week_unix[start], week_unix[-1], stages[-1]))
    return segs


def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("\\", "__")


# ---------- plotting ----------
def plot_repo(df_repo: pd.DataFrame, repo: str, stamp: str, outdir: Path, smooth: int = 5):
    df_repo = df_repo.sort_values("week_unix").reset_index(drop=True)
    need = {"repo", "week_unix", "stage_name"} | {m for m, _ in METRICS}
    miss = need - set(df_repo.columns)
    if miss:
        print(f"[WARN] {repo}: missing columns {sorted(miss)} — skip")
        return

    df_repo = dedupe_by_week(df_repo)

    # [Logic Fix] Align data points to MID-WEEK
    # x_start: Sunday 00:00 (Used for background segments)
    # x_line:  Wednesday 12:00 (Used for data line, +3.5 days)
    x_unix = df_repo["week_unix"].values
    x_start = to_utc_dt(df_repo["week_unix"])
    x_line = x_start + pd.Timedelta(days=3, hours=12)

    stages = df_repo["stage_name"].fillna("NA").values

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True,
                             figsize=compute_figsize(int(x_unix.min()), int(x_unix.max())))

    # 1. Background Spans (Aligns with x_start/Week Boundaries)
    segs = build_segments(x_unix, stages)
    for ax in axes:
        for (ts0, ts1, st) in segs:
            if st == "NA": continue
            color = STAGE_COLORS.get(st, "#eeeeee")
            ax.axvspan(to_utc_dt(pd.Series([ts0]))[0],
                       to_utc_dt(pd.Series([ts1]))[0],
                       facecolor=color, alpha=0.35, edgecolor=None)

    # 2. Data Lines (Aligns with x_line/Mid-Week)
    for ax, (col, label) in zip(axes, METRICS):
        y = df_repo[col].astype(float).values
        y_s = smooth_series(y, smooth)

        # Plot utilizing x_line to center the data visually within the week
        ax.plot(x_line, y_s, lw=1.8, label=label, color="#333333")

        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("Week (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))

    fig.suptitle(f"Life Activity(v2): {repo}", fontsize=14, y=0.98)

    # Legend
    handles = []
    for k in LEGEND_ORDER:
        if k in STAGE_COLORS:
            handles.append(Patch(facecolor=STAGE_COLORS[k], edgecolor="none", alpha=0.6, label=k))

    existing = set(stages)
    for st in existing:
        if st not in LEGEND_ORDER and st != "NA":
            handles.append(Patch(facecolor="#eeeeee", edgecolor="black", label=st))

    axes[0].legend(handles=handles, loc="upper right", fontsize=9, title="Stages", frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"repo_{safe_name(repo)}.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f"[OK] Plot saved -> {fn.name}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", help="Owner/name")
    ap.add_argument("--sample", type=int, default=10, help="Random sample size")
    # Default smooth set to 3 as discussed
    ap.add_argument("--smooth", type=int, default=3, help="Smoothing window")
    args = ap.parse_args()

    try:
        f_assign = infer_latest_assignments()
    except FileNotFoundError as e:
        print(f"[ERR] {e}")
        return

    stamp = infer_stamp_from_name(f_assign)
    print(f"[INFO] Data: {f_assign.name}")

    df = pd.read_csv(f_assign, parse_dates=["week_dt"])

    # Verify "Baseline" fix
    unique_stages = df["stage_name"].unique()
    if "Baseline" not in unique_stages and "Baseline (Solo)" in unique_stages:
        print("[WARN] Found 'Baseline (Solo)' in CSV but config uses 'Baseline'. Auto-fixing column...")
        df["stage_name"] = df["stage_name"].replace({"Baseline (Solo)": "Baseline"})

    repos_all = sorted(df["repo"].unique())

    if args.repo:
        target = [r for r in args.repo if r in repos_all]
        if not target:
            print("[WARN] No valid repos found.")
            return
    else:
        rng = np.random.default_rng(42)
        n = min(args.sample, len(repos_all))
        target = sorted(rng.choice(repos_all, size=n, replace=False).tolist())
        print(f"[INFO] Sampling {n} repos.")

    outdir = D05_C
    print(f"[INFO] Saving to: {outdir}")

    for i, r in enumerate(target, 1):
        print(f"[{i}/{len(target)}] {r}...")
        plot_repo(df[df["repo"] == r], r, stamp, outdir, smooth=args.smooth)


if __name__ == "__main__":
    main()