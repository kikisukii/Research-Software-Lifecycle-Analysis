#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_c_plot_random_v2.py
零参数：自动定位最新 <STAMP> 的 labels；随机抽 5 个 repo 绘图（UTC，背景=stage_display）。
y 轴优先用 commits(02_gh_commit_weekly_<STAMP>.csv)，缺则回退到 c8_total→M8_24。
去掉“顶部次标签带”；改用高对比配色；图宽度随时间跨度自适应；折线做轻微平滑。

输出：data/05_c_viz/<STAMP>/random/repo_<owner__name>_<metric>.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ===== 可调参数（仅影响可视化，不影响任何计算/标签） =====
SMOOTH_WINDOW = 5            # 平滑窗口（周），centered rolling；设为 1 即无平滑
PLOT_ORIGINAL  = False       # 同时叠一条原始折线（淡色）
MIN_WIDTH, MAX_WIDTH = 8, 24 # 图宽限制（英寸）
WIDTH_SLOPE = 0.04           # 宽度与周数的线性系数（宽度≈4 + 周数*WIDTH_SLOPE，再夹在[min,max]）

# 高对比配色（Initial 与 Dead 明显区分）
STAGE_COLOR = {
    "Initial": "#cfe8ff",  # 浅蓝
    "Rising":  "#4daf4a",  # 绿
    "Peak":    "#8e6bbd",  # 紫
    "Cooling": "#ff8a48",  # 橙
    "Dead":    "#5f6368",  # 深灰
}
STAMP_RE = re.compile(r"(\d{8}_\d{6})")

def to_utc(series_unix):
    return pd.to_datetime(series_unix, unit="s", utc=True)

def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("\\", "__")

def find_latest_labels() -> Path:
    cands = list(Path("../data/05_b_apply").glob("*/05_labels_v1_*_K*.csv"))
    if not cands:
        raise FileNotFoundError("未找到 labels：data/05_b_apply/*/05_labels_v1_*_K*.csv")
    def key_fn(p: Path):
        m = STAMP_RE.search(p.name)
        return m.group(1) if m else "00000000_000000"
    cands.sort(key=key_fn)
    return cands[-1]

def infer_stamp(p: Path) -> str:
    m = STAMP_RE.search(p.name) or STAMP_RE.search(p.as_posix())
    if not m:
        raise ValueError(f"无法从路径推断 <STAMP>: {p}")
    return m.group(1)

def build_segments(x_ts, stages):
    """根据 stage 序列构造连续分段（忽略 'NA'）。返回 [(t0,t1,stage), ...]，t1 为下一点的时间。"""
    segs = []
    if len(x_ts) == 0: return segs
    start = 0
    for i in range(1, len(x_ts)):
        if stages[i] != stages[i-1]:
            segs.append((x_ts[start], x_ts[i], stages[i-1]))
            start = i
    segs.append((x_ts[start], x_ts[-1], stages[-1]))
    return [(t0, t1, s) for (t0, t1, s) in segs if s != "NA"]

def prepare_labels_for_repo(dfL_repo: pd.DataFrame) -> pd.DataFrame:
    """
    清理同周重复标签：保留“第一个标签”为主标签 stage_base（无次标签）
    """
    def uniq_first(vals):
        vals = [x for x in vals if pd.notna(x)]
        u = np.unique(vals)
        return u[0] if len(u) else np.nan
    g = (dfL_repo.groupby(["repo","week_unix"])["stage_display"]
                 .apply(lambda s: uniq_first(list(s)))
                 .reset_index(name="stage_base"))
    return g

def smooth_series(y: pd.Series, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y.values.astype(float)
    return y.astype(float).rolling(window=window, center=True, min_periods=1).mean().values

def compute_figsize(ts_min: int, ts_max: int, height: float = 4.8):
    # 以周数估算宽度：width ≈ 4 + 周数 * WIDTH_SLOPE，之后夹在 [MIN_WIDTH, MAX_WIDTH]
    weeks = max(1, int(round((ts_max - ts_min) / (7*24*3600))) + 1)
    width = 4 + weeks * WIDTH_SLOPE
    width = max(MIN_WIDTH, min(MAX_WIDTH, width))
    return (width, height)

def plot_one(repo, df_repo: pd.DataFrame, y_col, outdir: Path):
    df_repo = df_repo.sort_values("week_unix").reset_index(drop=True)
    x = to_utc(df_repo["week_unix"])
    y = df_repo[y_col].astype(float)

    # 平滑（仅用于可视化）
    y_smooth = smooth_series(pd.Series(y), SMOOTH_WINDOW)

    # 动态尺寸
    w, h = compute_figsize(int(df_repo["week_unix"].min()), int(df_repo["week_unix"].max()))

    fig, ax = plt.subplots(figsize=(w, h))
    if PLOT_ORIGINAL:
        ax.plot(x, y, lw=1.0, alpha=0.35, label=f"{y_col} (raw)")
    ax.plot(x, y_smooth, lw=1.8, label=f"{y_col} (smoothed)")

    ax.set_xlabel("Week (UTC)")
    ax.set_ylabel(y_col)

    # 背景：按主标签分段
    base = df_repo["stage_base"].fillna("NA").values
    segs_base = build_segments(df_repo["week_unix"].values, base)
    for (ts0, ts1, stage) in segs_base:
        ax.axvspan(to_utc([ts0])[0], to_utc([ts1])[0],
                   facecolor=STAGE_COLOR.get(stage, "#eee"), alpha=0.22, edgecolor=None)

    # 阶段切换竖线
    for i in range(1, len(df_repo)):
        if base[i] != base[i-1]:
            ax.axvline(x[i], ls="--", lw=0.8, alpha=0.55)

    # 轴样式
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"{repo} — {y_col} (UTC)")

    # 图例：颜色 → 周期；线条 → raw/smoothed
    handles = [Patch(facecolor=STAGE_COLOR[k], edgecolor='none', alpha=0.6, label=k)
               for k in ["Initial","Rising","Peak","Cooling","Dead"]]
    leg1 = ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8, title="Stage colors")
    leg1._legend_box.align = "left"
    line_labels = ([f"{y_col} (raw)"] if PLOT_ORIGINAL else []) + [f"{y_col} (smoothed)"]
    ax.legend(labels=line_labels, loc="upper right", frameon=True, fontsize=8, title="Series")
    ax.add_artist(leg1)

    fn = outdir / f"repo_{safe_name(repo)}_{y_col}.png"
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)
    print(f"[OK] {repo} -> {fn}")

def main():
    # 1) 找最新批次
    labels_fp = find_latest_labels()
    stamp = infer_stamp(labels_fp)
    outdir = Path(f"data/05_c_viz/{stamp}/random"); outdir.mkdir(parents=True, exist_ok=True)

    # 2) 同批 features / commits（features 优先 full，再退 filtered）
    feat_full = Path(f"data/03_features_weekly_v1_{stamp}.csv")
    feat_filt = Path(f"data/03_features_weekly_v1_{stamp}_filtered.csv")
    if feat_full.exists():
        features_fp = feat_full
    elif feat_filt.exists():
        features_fp = feat_filt
        print("[WARN] 未找到 full 特征表，使用 filtered（可能没有 Initial）")
    else:
        raise FileNotFoundError(f"未找到特征表：{feat_full} 或 {feat_filt}")

    commits_fp = Path(f"data/02_gh_commit_weekly_{stamp}.csv")  # 原始每周提交
    has_commits = commits_fp.exists()

    # 3) 读数据
    dfL = pd.read_csv(labels_fp)   # 需含：repo, week_unix, stage_display
    need = {"repo","week_unix","stage_display"}
    if not need.issubset(dfL.columns):
        raise ValueError(f"labels 缺列：{need - set(dfL.columns)}")
    dfF = pd.read_csv(features_fp) # 需含：repo, week_unix, (+ c8_total/M8_24)
    dfC = pd.read_csv(commits_fp) if has_commits else None
    if dfC is not None and "commits" not in dfC.columns:
        print("[WARN] commits_csv 不含 'commits' 列，将忽略 commits"); dfC = None

    # 4) 随机抽 5 个 repo（每次不同）
    repos = sorted(dfL["repo"].unique())
    if not repos: raise ValueError("labels 中没有 repo")
    n_pick = min(5, len(repos))
    rng = np.random.default_rng()  # 无种子：每次不同
    pick = rng.choice(repos, size=n_pick, replace=False)

    print(f"[INFO] 使用批次 <STAMP>={stamp}；抽样 {n_pick}/{len(repos)} 个仓")
    for r in pick:
        # 标签：保留同周首个标签为主
        lab_raw = dfL[dfL["repo"] == r][["repo","week_unix","stage_display"]]
        lab = prepare_labels_for_repo(lab_raw)

        # 合并特征
        feat_cols = ["repo","week_unix"] + [c for c in ["c8_total","M8_24"] if c in dfF.columns]
        feat = dfF[dfF["repo"] == r][feat_cols]
        data = pd.merge(lab, feat, on=["repo","week_unix"], how="left")

        # 合并 commits（优先作为 y）
        y_col = None
        if dfC is not None:
            com = dfC[dfC["repo"] == r][["repo","week_unix","commits"]]
            data = pd.merge(data, com, on=["repo","week_unix"], how="left")
            if "commits" in data.columns and not data["commits"].isna().all():
                y_col = "commits"
        if y_col is None:
            if "c8_total" in data.columns and not data["c8_total"].isna().all():
                y_col = "c8_total"
            elif "M8_24" in data.columns and not data["M8_24"].isna().all():
                y_col = "M8_24"
            else:
                print(f"[WARN] {r} 无可用 y 列（commits/c8_total/M8_24），跳过"); continue

        plot_one(r, data, y_col, outdir)

if __name__ == "__main__":
    main()
