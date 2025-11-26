# -*- coding: utf-8 -*-
"""
v2/05c_plot_repo_v2_weekly.py — Per-repo WEEKLY metric timeline with stage-colored background.
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
# 输出到专门的测试文件夹，以免混淆
D05_C = ROOT / "v2_data" / "05_c_weekly_test"

# ---------- config ----------
# 1. Custom Colors
STAGE_COLORS = {
    "Baseline": "#f8b862",
    "Internal Development": "#38b48b",
    "Release Phase": "#9d5b8b",
    "Peak Activity": "#e9546b",
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

# 核心修改：使用 Weekly 列名
METRICS = [
    ("commits_weekly", "Commits (Weekly)"),
    ("contributors_weekly", "Active Contributors (Weekly)"),
    ("issues_weekly", "Issues Closed (Weekly)"),
    ("releases_weekly", "Releases (Weekly)"),
]

MIN_WIDTH, MAX_WIDTH = 8.0, 22.0
WIDTH_SLOPE = 0.04
STAMP_RE = re.compile(r"_v2_(\d{8}_\d{6})_K\d+\.csv$")


# ---------- utils ----------
def infer_latest_assignments() -> Path:
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
        if not m2: return "unknown"
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
    # center=True 很重要，保证波峰不滞后
    return s.rolling(window=window, center=True, min_periods=1).mean().values


def dedupe_by_week(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["repo", "week_unix"]).drop_duplicates(["repo", "week_unix"], keep="first")


def build_segments(week_unix: np.ndarray, stages: np.ndarray):
    segs = []
    if len(week_unix) == 0: return segs
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

    # 检查列是否存在
    required_cols = {m for m, _ in METRICS}
    if not required_cols.issubset(df_repo.columns):
        print(f"[ERR] {repo}: 缺少 Weekly 列。请确保已重跑 03 和 05b。")
        return

    df_repo = dedupe_by_week(df_repo)
    x_unix = df_repo["week_unix"].values
    x = to_utc_dt(df_repo["week_unix"])
    stages = df_repo["stage_name"].fillna("NA").values

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True,
                             figsize=compute_figsize(int(x_unix.min()), int(x_unix.max())))

    # 1. 画背景 (Stages)
    segs = build_segments(x_unix, stages)
    for ax in axes:
        for (ts0, ts1, st) in segs:
            if st == "NA": continue
            color = STAGE_COLORS.get(st, "#eeeeee")
            ax.axvspan(to_utc_dt(pd.Series([ts0]))[0],
                       to_utc_dt(pd.Series([ts1]))[0],
                       facecolor=color, alpha=0.35, edgecolor=None)

    # 2. 画折线 (Weekly Data)
    for ax, (col, label) in zip(axes, METRICS):
        y = df_repo[col].fillna(0).astype(float).values
        # 平滑处理，让 Weekly 数据不那么刺眼
        y_s = smooth_series(y, smooth)

        ax.plot(x, y_s, lw=1.5, label=label, color="#333333")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("Week (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))

    fig.suptitle(f"Life Activity (v2 Weekly): {repo}", fontsize=14, y=0.98)

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
    # [修改] 将默认平滑窗口从 5 改为 10，适配 Weekly 数据
    ap.add_argument("--smooth", type=int, default=10, help="Smoothing window")  # 默认平滑5周
    args = ap.parse_args()

    try:
        f_assign = infer_latest_assignments()
    except FileNotFoundError as e:
        print(f"[ERR] {e}")
        return

    stamp = infer_stamp_from_name(f_assign)
    print(f"[INFO] Data: {f_assign.name}")

    df = pd.read_csv(f_assign, parse_dates=["week_dt"])

    # 列检查
    test_cols = [m[0] for m in METRICS]
    missing = [c for c in test_cols if c not in df.columns]
    if missing:
        print(f"[CRITICAL] 输入文件缺少 Weekly 列: {missing}")
        print("请重新运行 03 (需修改代码) -> 05b。")
        return

    # 修复 Baseline 命名
    if "Baseline (Solo)" in df["stage_name"].unique():
        df["stage_name"] = df["stage_name"].replace({"Baseline (Solo)": "Baseline"})

    repos_all = sorted(df["repo"].unique())

    if args.repo:
        target = [r for r in args.repo if r in repos_all]
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