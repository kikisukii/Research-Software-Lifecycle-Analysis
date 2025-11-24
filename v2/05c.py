# -*- coding: utf-8 -*-
"""
v2/05c_plot_repo_v2.py — Per-repo multi-metric timeline with stage-colored background.
- One repo -> one figure with 4 stacked subplots (commits / contributors / issues / releases).
- Background colored by stage_name (from 05b assignments).
- Auto-detect latest <STAMP>. Prefer *_AND_DEAD_* file if present.
- Saves to v2_data/05_c/<STAMP>/repo_<owner__name>.png

Optional CLI:
  --repo "<owner/name>"    plot only this repo (can repeat)
  --max  N                 plot up to N repos (random sample if no --repo)
  --smooth N               smoothing window (weeks), default 5; use 1 to disable
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
D05_B = ROOT / "v2_data" / "05_b"
D05_C = ROOT / "v2_data" / "05_c"

# ---------- config ----------
STAGE_COLORS = {
    "Rising-Dev":          "#4daf4a",  # green
    "Peak-Release":        "#8e6bbd",  # purple
    "Maintenance/Triage":  "#1f78b4",  # blue
    "Cooling-Release":     "#ff8a48",  # orange
    "Low":                 "#999999",  # grey
    "Zero/Low":            "#cccccc",  # light grey
    "Dead":                "#5f6368",  # dark grey
}
METRICS = [
    ("commits_8w_sum", "Commits (8w sum)"),
    ("contributors_8w_unique", "Contributors (8w unique)"),
    ("issues_closed_8w_count", "Issues closed (8w)"),
    ("releases_8w_count", "Releases (8w)"),
]
MIN_WIDTH, MAX_WIDTH = 8.0, 22.0
WIDTH_SLOPE = 0.04  # ≈ 4 + weeks * slope

STAMP_RE = re.compile(r"_v2_(\d{8}_\d{6})_K\d+\.csv$")

# ---------- utils ----------
def infer_latest_assignments() -> Path:
    # Prefer the file that includes DEAD rows if present
    cands_dead = sorted(D05_B.glob("05b_assignments_with_stage_AND_DEAD_v2_*.csv"))
    cands_main = sorted(D05_B.glob("05b_assignments_with_stage_v2_*.csv"))
    cands = cands_dead if cands_dead else cands_main
    if not cands:
        raise FileNotFoundError("No 05b assignments found under v2_data/05_b/")
    return cands[-1]

def infer_stamp_from_name(p: Path) -> str:
    m = STAMP_RE.search(p.name)
    if not m:
        # fallback: try any 8+6 pattern
        m2 = re.search(r"(\d{8}_\d{6})", p.name)
        if not m2:
            raise RuntimeError(f"Cannot infer STAMP from filename: {p.name}")
        return m2.group(1)
    return m.group(1)

def to_utc_dt(unix_s: pd.Series) -> pd.Series:
    return pd.to_datetime(unix_s, unit="s", utc=True)

def compute_figsize(week_min: int, week_max: int, nrows: int = 4, row_h: float = 2.0):
    weeks = max(1, int(round((week_max - week_min) / (7*24*3600))) + 1)
    width = max(MIN_WIDTH, min(MAX_WIDTH, 4 + weeks * WIDTH_SLOPE))
    height = max(5.0, nrows * row_h + 1.2)
    return (width, height)

def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return arr.astype(float)
    s = pd.Series(arr.astype(float))
    return s.rolling(window=window, center=True, min_periods=1).mean().values

def dedupe_by_week(df: pd.DataFrame) -> pd.DataFrame:
    # keep first record per (repo, week_unix)
    return df.sort_values(["repo","week_unix"]).drop_duplicates(["repo","week_unix"], keep="first")

def build_segments(week_unix: np.ndarray, stages: np.ndarray):
    segs = []
    if len(week_unix) == 0:
        return segs
    start = 0
    for i in range(1, len(week_unix)):
        if stages[i] != stages[i-1]:
            segs.append((week_unix[start], week_unix[i], stages[i-1]))
            start = i
    segs.append((week_unix[start], week_unix[-1], stages[-1]))
    return segs

def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("\\", "__")

# ---------- plotting ----------
def plot_repo(df_repo: pd.DataFrame, repo: str, stamp: str, outdir: Path, smooth:int=5):
    df_repo = df_repo.sort_values("week_unix").reset_index(drop=True)
    # ensure required columns
    need = {"repo","week_unix","stage_name"} | {m for m,_ in METRICS}
    miss = need - set(df_repo.columns)
    if miss:
        print(f"[WARN] {repo}: missing columns {sorted(miss)} — skip")
        return

    # dedupe per week in case of duplicates
    df_repo = dedupe_by_week(df_repo)

    x_unix = df_repo["week_unix"].values
    x = to_utc_dt(df_repo["week_unix"])
    stages = df_repo["stage_name"].fillna("NA").values

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True,
                             figsize=compute_figsize(int(x_unix.min()), int(x_unix.max())))

    # background spans on each subplot
    segs = build_segments(x_unix, stages)
    for ax in axes:
        for (ts0, ts1, st) in segs:
            if st == "NA":
                continue
            color = STAGE_COLORS.get(st, "#eeeeee")
            ax.axvspan(to_utc_dt(pd.Series([ts0]))[0],
                       to_utc_dt(pd.Series([ts1]))[0],
                       facecolor=color, alpha=0.20, edgecolor=None)

    # plot each metric
    for ax, (col, label) in zip(axes, METRICS):
        y = df_repo[col].astype(float).values
        y_s = smooth_series(y, smooth)
        ax.plot(x, y_s, lw=1.8, label=label)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Week (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    fig.suptitle(f"{repo} — 8-week window metrics with stage background (STAMP={stamp})", fontsize=12)

    # Stage legend (single, top-left)
    handles = [Patch(facecolor=STAGE_COLORS.get(k, "#eee"), edgecolor="none", alpha=0.6, label=k)
               for k in ["Rising-Dev","Peak-Release","Maintenance/Triage","Cooling-Release","Low","Zero/Low","Dead"]]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, title="Stage colors")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"repo_{safe_name(repo)}.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f"[OK] {repo} -> {fn}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", help="Owner/name; can repeat.", default=None)
    ap.add_argument("--max", type=int, default=8, help="Max repos to plot when no --repo is given.")
    ap.add_argument("--smooth", type=int, default=5, help="Smoothing window (weeks). Use 1 to disable.")
    args = ap.parse_args()

    f_assign = infer_latest_assignments()
    stamp = infer_stamp_from_name(f_assign)

    df = pd.read_csv(f_assign, parse_dates=["week_dt"])
    # only keep columns we need
    keep_cols = ["repo","week_unix","stage_name"] + [m for m,_ in METRICS]
    df = df[keep_cols].copy()

    repos_all = sorted(df["repo"].unique())
    if not repos_all:
        raise ValueError("No repos found in assignments.")

    if args.repo:
        target = []
        for r in args.repo:
            if r in repos_all:
                target.append(r)
            else:
                print(f"[WARN] repo not found: {r}")
        if not target:
            print("[WARN] no valid repos from --repo; exit")
            return
    else:
        rng = np.random.default_rng(123)  # deterministic sample
        n = min(args.max, len(repos_all))
        target = sorted(rng.choice(repos_all, size=n, replace=False).tolist())

    outdir = D05_C / stamp
    print(f"[INFO] Assignments: {f_assign.name} | STAMP={stamp} | repos={len(repos_all)} | pick={len(target)}")
    for i, r in enumerate(target, 1):
        print(f"[{i}/{len(target)}] plotting {r} ...")
        plot_repo(df[df["repo"] == r], r, stamp, outdir, smooth=args.smooth)

if __name__ == "__main__":
    main()
