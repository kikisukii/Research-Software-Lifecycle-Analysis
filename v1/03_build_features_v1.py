# 03_build_features_v1.py
# v1 (commit-side only): build weekly features from commit-weekly CSV
# Outputs:
#   data/03_features_weekly_v1_<STAMP>.csv
#   data/03_features_weekly_v1_<STAMP>_filtered.csv            # only rows with window_ok_24w == True
#   data/03_features_weekly_v1_<STAMP>.meta.json               # provenance / snapshot

from pathlib import Path
from datetime import datetime, timezone
import argparse
import pandas as pd
import numpy as np
import json
import re

WEEK_SEC = 7 * 24 * 60 * 60
SHORT_W, LONG_W = 8, 24               # 窗口：8周 / 24周
EPS = 1e-6

# ---------------- utils ----------------

def latest_csv(pattern: str) -> str:
    """
    选取 pattern 下“最新”的 CSV：
    - 优先用文件名里的 _YYYYMMDD_HHMMSS 作为时间戳
    - 否则退回按修改时间 mtime
    """
    files = list(Path("..").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")

    def ts_key(p: Path) -> float:
        m = re.search(r"_(\d{8})_(\d{6})\.csv$", p.name)
        if m:
            return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S").timestamp()
        return p.stat().st_mtime

    return max(files, key=ts_key).as_posix()


def stamp_from_name(csv_path: str) -> str:
    """从文件名提取 _YYYYMMDD_HHMMSS；若无则用当前UTC时间。"""
    name = Path(csv_path).name
    m = re.search(r"_(\d{8})_(\d{6})\.csv$", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# ---------------- IO ----------------

def load_commits(commits_csv: str) -> pd.DataFrame:
    df = pd.read_csv(commits_csv)
    need = {"repo", "week_unix", "commits"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{commits_csv} missing columns: {miss}")
    # 类型清洗
    df["repo"] = df["repo"].astype(str)
    df["week_unix"] = pd.to_numeric(df["week_unix"], errors="coerce").astype("Int64")
    df["commits"] = pd.to_numeric(df["commits"], errors="coerce").fillna(0).astype("Int64")
    # 聚合去重（防同一周重复行）
    df = df.groupby(["repo","week_unix"], as_index=False, sort=False)["commits"].sum()
    # 转成 numpy int64，便于后续计算
    df["week_unix"] = df["week_unix"].astype(np.int64)
    df["commits"] = df["commits"].astype(np.int64)
    return df


def load_repo_meta_optional(meta_glob: str) -> tuple[str | None, set[str]]:
    """尝试加载最新的 repo_meta CSV（可选）。返回 (路径或None, 仓库集合)。"""
    try:
        meta_csv = latest_csv(meta_glob)
    except FileNotFoundError:
        return None, set()
    meta = pd.read_csv(meta_csv)
    if "full_name" in meta.columns:
        repos = set(meta["full_name"].astype(str))
    elif "repo" in meta.columns:
        repos = set(meta["repo"].astype(str))
    else:
        repos = set()
    return meta_csv, repos

# ---------------- feature building ----------------

def build_week_grid(df_commits: pd.DataFrame) -> pd.DataFrame:
    """按 repo 补连续周轴（周日00:00:00 UTC 对齐）；缺周 commits=0。"""
    crates = []
    for repo, g in df_commits.groupby("repo", sort=False):
        g = g.sort_values("week_unix")
        start, end = int(g["week_unix"].min()), int(g["week_unix"].max())
        weeks = np.arange(start, end + 1, WEEK_SEC, dtype=np.int64)
        base = pd.DataFrame({"repo": repo, "week_unix": weeks})
        merged = base.merge(g[["week_unix", "commits"]], on="week_unix", how="left")
        merged["commits"] = merged["commits"].fillna(0).astype(np.int64)
        crates.append(merged)
    return pd.concat(crates, ignore_index=True)


def _first_nonzero_week(g: pd.DataFrame) -> int:
    nz = g.loc[g["commits"] > 0, "week_unix"]
    return int(nz.min()) if len(nz) else int(g["week_unix"].min())


def add_roll_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    每个 repo：
      c8_total      = 最近8周提交总数（含本周）
      c24_total     = 最近24周提交总数（含本周）
      M8_24         = c8_total / max(c24_total/3, 1e-6)   # 总数口径动量
      dead_flag     = 1 若 c24_total == 0，否则 0
      age_weeks     = 距首次非零提交周的周数（>=0）
      week_date     = 可读日期（UTC）
      window_ok_24w = age_weeks >= 24   # 聚类/分类建议只用 True 的行
    """
    def per_repo(g: pd.DataFrame) -> pd.DataFrame:
        repo = g.name  # include_groups=False 时，用分组键
        g = g.sort_values("week_unix").copy()
        g["repo"] = repo

        g["c8_total"]  = g["commits"].rolling(SHORT_W, min_periods=1).sum()
        g["c24_total"] = g["commits"].rolling(LONG_W,  min_periods=1).sum()
        g["M8_24"] = g["c8_total"] / np.maximum(g["c24_total"] / 3.0, EPS)
        g["dead_flag"] = (g["c24_total"] == 0).astype(np.int8)

        first_w = _first_nonzero_week(g)
        g["age_weeks"] = ((g["week_unix"] - first_w) // WEEK_SEC).clip(lower=0).astype(np.int64)
        g["week_date"] = pd.to_datetime(g["week_unix"], unit="s", utc=True)
        g["window_ok_24w"] = (g["age_weeks"] >= 24)

        # 列类型更省内存
        g["c8_total"]  = g["c8_total"].astype(np.int64)
        g["c24_total"] = g["c24_total"].astype(np.int64)
        return g

    # include_groups=False 消除 FutureWarning
    feats = df_full.groupby("repo", group_keys=False).apply(per_repo, include_groups=False)
    # 统一列顺序
    cols = ["repo","week_unix","week_date","commits","c8_total","c24_total","M8_24","dead_flag","age_weeks","window_ok_24w"]
    return feats[cols]

# ---------------- main ----------------

def main(commits_glob: str, meta_glob: str | None, out_path: str | None):
    # 选择最新输入 + 构造戳
    commits_csv = latest_csv(commits_glob)
    stamp = stamp_from_name(commits_csv)

    # 输出名（自动 v1 + 源时间戳）
    if not out_path or out_path.lower() == "auto":
        out_csv_full = f"data/03_features_weekly_v1_{stamp}.csv"
        out_csv_filt = f"data/03_features_weekly_v1_{stamp}_filtered.csv"
    else:
        out_csv_full = out_path
        stem = Path(out_csv_full).with_suffix("").as_posix()
        out_csv_filt = f"{stem}_filtered.csv"
    meta_json = Path(out_csv_full).with_suffix(".meta.json")

    print(f"[use] commits_csv = {commits_csv}")

    # 读 commits，补轴 & 特征
    df_raw  = load_commits(commits_csv)
    df_full = build_week_grid(df_raw)
    feats   = add_roll_features(df_full)

    # 过滤视图（供聚类/分类使用）
    feats_filt = feats.loc[feats["window_ok_24w"]].copy()

    # 可选加载 repo_meta（覆盖率对比）
    meta_csv_used, meta_repos = (None, set())
    if meta_glob:
        meta_csv_used, meta_repos = load_repo_meta_optional(meta_glob)
    elif Path("../data").exists():
        # 若用户没传，但 data/ 下常规命名存在，也尝试加载
        try:
            meta_csv_used, meta_repos = load_repo_meta_optional("data/02_gh_repo_meta_*.csv")
        except FileNotFoundError:
            meta_csv_used, meta_repos = (None, set())

    commits_repos = set(feats["repo"].unique())
    missing_in_commits = sorted(list(meta_repos - commits_repos))[:30] if meta_repos else []  # 只截前30个做预览

    # 落盘
    Path(out_csv_full).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_csv_full, index=False)
    feats_filt.to_csv(out_csv_filt, index=False)

    # 元数据快照
    meta = {
        "version": "v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commits_csv": commits_csv,
        "source_stamp": stamp,
        "windows_weeks": {"short": SHORT_W, "long": LONG_W},
        "momentum_formula": "M8_24 = c8_total / max(c24_total/3, 1e-6)  # totals-based",
        "dead_rule": "dead_flag = (c24_total == 0)  # last 24 weeks no commits",
        "filter_flag": "window_ok_24w = (age_weeks >= 24)  # use filtered CSV for clustering/classification",
        "rows_full": int(len(feats)),
        "rows_filtered": int(len(feats_filt)),
        "repos_in_features": int(feats['repo'].nunique()),
        "repo_meta_csv": meta_csv_used,
        "repo_meta_repos": int(len(meta_repos)) if meta_repos else None,
        "meta_minus_commits_preview": missing_in_commits if missing_in_commits else None,
        "date_range_utc": {
            "min_week": str(pd.to_datetime(int(feats["week_unix"].min()), unit="s", utc=True)),
            "max_week": str(pd.to_datetime(int(feats["week_unix"].max()), unit="s", utc=True)),
        }
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved  {out_csv_full}")
    print(f"[ok] saved  {out_csv_filt}  # use this for clustering/classification")
    print(f"[ok] meta   {meta_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build v1 weekly features from commit-weekly CSV.")
    parser.add_argument("--commits-glob", default="data/02_gh_commit_weekly_*.csv",
                        help="Glob pattern to locate commit-weekly CSVs (auto-pick latest)")
    parser.add_argument("--repo-meta-glob", default="data/02_gh_repo_meta_*.csv",
                        help="(Optional) Glob for repo meta CSV; used only for coverage reporting")
    parser.add_argument("--out", default="auto",
                        help="Output CSV path; use 'auto' to name as data/03_features_weekly_v1_<STAMP>.csv")
    args = parser.parse_args()
    main(commits_glob=args.commits_glob, meta_glob=args.repo_meta_glob, out_path=args.out)
