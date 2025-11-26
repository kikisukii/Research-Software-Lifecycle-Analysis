# 03_build_features_v1.py
# v1 (commit-side only): build weekly features from commit-weekly CSV
# Outputs:
#   ../v1_data/03_features/03_features_weekly_v1_<STAMP>.csv
#   ../v1_data/03_features/03_features_weekly_v1_<STAMP>_filtered.csv
#   ../v1_data/03_features/03_features_weekly_v1_<STAMP>.meta.json

from pathlib import Path
from datetime import datetime, timezone
import argparse
import pandas as pd
import numpy as np
import json
import re
import sys

# ==========================================
# Config & Path Management (Fixed)
# ==========================================

# 1. Locate this script: .../project_root/v1/03_build_features_v1.py
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# 2. Locate Project Root: .../project_root/
PROJECT_ROOT = SCRIPT_DIR.parent

# 3. Locate v1_data (Sibling to v1): .../project_root/v1_data
# This ensures we don't accidentally create v1/v1_data if running from inside v1
V1_DATA_ROOT = PROJECT_ROOT / "v1_data"

# Subdirectories
INPUT_DIR = V1_DATA_ROOT / "02_dat"
OUTPUT_DIR = V1_DATA_ROOT / "03_features"

WEEK_SEC = 7 * 24 * 60 * 60
SHORT_W, LONG_W = 8, 24
EPS = 1e-6


# ---------------- utils ----------------

def latest_csv(pattern: str) -> str:
    """
    Selects the 'latest' CSV matching the pattern.
    The pattern is applied relative to V1_DATA_ROOT if not absolute.
    """
    # Construct search path: .../v1_data/02_dat/pattern
    # We search specifically in the Input Directory defined above
    target_dir = INPUT_DIR

    # Check if directory exists
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {target_dir}\nMake sure you have run step 02 and the folder 'v1_data/02_dat' exists next to 'v1'.")

    files = list(target_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching: {pattern} in {target_dir}")

    def ts_key(p: Path) -> float:
        m = re.search(r"_(\d{8})_(\d{6})\.csv$", p.name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        return p.stat().st_mtime

    return max(files, key=ts_key).as_posix()


def stamp_from_name(csv_path: str) -> str:
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

    df["repo"] = df["repo"].astype(str)
    df["week_unix"] = pd.to_numeric(df["week_unix"], errors="coerce").astype("Int64")
    df["commits"] = pd.to_numeric(df["commits"], errors="coerce").fillna(0).astype("Int64")

    # Aggregate duplicates
    df = df.groupby(["repo", "week_unix"], as_index=False, sort=False)["commits"].sum()

    df["week_unix"] = df["week_unix"].astype(np.int64)
    df["commits"] = df["commits"].astype(np.int64)
    return df


def load_repo_meta_optional(meta_pattern: str) -> tuple[str | None, set[str]]:
    """Attempts to load the latest repo_meta CSV from INPUT_DIR."""
    try:
        # Re-use latest_csv logic but for meta pattern
        meta_csv = latest_csv(meta_pattern)
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
    def per_repo(g: pd.DataFrame) -> pd.DataFrame:
        repo = g.name
        g = g.sort_values("week_unix").copy()
        g["repo"] = repo

        g["c8_total"] = g["commits"].rolling(SHORT_W, min_periods=1).sum()
        g["c24_total"] = g["commits"].rolling(LONG_W, min_periods=1).sum()
        g["M8_24"] = g["c8_total"] / np.maximum(g["c24_total"] / 3.0, EPS)
        g["dead_flag"] = (g["c24_total"] == 0).astype(np.int8)

        first_w = _first_nonzero_week(g)
        g["age_weeks"] = ((g["week_unix"] - first_w) // WEEK_SEC).clip(lower=0).astype(np.int64)
        g["week_date"] = pd.to_datetime(g["week_unix"], unit="s", utc=True)
        g["window_ok_24w"] = (g["age_weeks"] >= 24)

        g["c8_total"] = g["c8_total"].astype(np.int64)
        g["c24_total"] = g["c24_total"].astype(np.int64)
        return g

    feats = df_full.groupby("repo", group_keys=False).apply(per_repo, include_groups=False)
    cols = ["repo", "week_unix", "week_date", "commits", "c8_total", "c24_total", "M8_24", "dead_flag", "age_weeks",
            "window_ok_24w"]
    return feats[cols]


# ---------------- main ----------------

def main(commits_pattern: str, meta_pattern: str | None, out_path_arg: str | None):
    # 1. Resolve Input Path
    # It will look inside V1_DATA_ROOT/02_dat/ for this pattern
    try:
        commits_csv = latest_csv(commits_pattern)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    stamp = stamp_from_name(commits_csv)

    # 2. Resolve Output Path
    # If auto, use V1_DATA_ROOT/03_features/...
    if not out_path_arg or out_path_arg.lower() == "auto":
        out_csv_full = OUTPUT_DIR / f"03_features_weekly_v1_{stamp}.csv"
        out_csv_filt = OUTPUT_DIR / f"03_features_weekly_v1_{stamp}_filtered.csv"
    else:
        # If user provides a path, use it as is
        out_csv_full = Path(out_path_arg)
        stem = out_csv_full.parent / out_csv_full.stem
        out_csv_filt = Path(f"{stem}_filtered.csv")

    meta_json = out_csv_full.with_suffix(".meta.json")

    print(f"[config] Script location: {SCRIPT_DIR}")
    print(f"[config] Data root:      {V1_DATA_ROOT}")
    print(f"[input]  Commits CSV:    {commits_csv}")

    # Process
    df_raw = load_commits(commits_csv)
    df_full = build_week_grid(df_raw)
    feats = add_roll_features(df_full)
    feats_filt = feats.loc[feats["window_ok_24w"]].copy()

    # Meta coverage check
    meta_csv_used, meta_repos = (None, set())
    if meta_pattern:
        try:
            meta_csv_used, meta_repos = load_repo_meta_optional(meta_pattern)
        except FileNotFoundError:
            pass  # It's optional

    commits_repos = set(feats["repo"].unique())
    missing_in_commits = sorted(list(meta_repos - commits_repos))[:30] if meta_repos else []

    # Ensure output directory exists
    out_csv_full.parent.mkdir(parents=True, exist_ok=True)

    feats.to_csv(out_csv_full, index=False)
    feats_filt.to_csv(out_csv_filt, index=False)

    meta = {
        "version": "v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commits_csv": commits_csv,
        "source_stamp": stamp,
        "windows_weeks": {"short": SHORT_W, "long": LONG_W},
        "dead_rule": "dead_flag = (c24_total == 0)",
        "rows_full": int(len(feats)),
        "rows_filtered": int(len(feats_filt)),
        "repos_in_features": int(feats['repo'].nunique()),
        "repo_meta_csv": meta_csv_used,
        "date_range_utc": {
            "min_week": str(pd.to_datetime(int(feats["week_unix"].min()), unit="s", utc=True)),
            "max_week": str(pd.to_datetime(int(feats["week_unix"].max()), unit="s", utc=True)),
        }
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved  {out_csv_full}")
    print(f"[ok] saved  {out_csv_filt}")
    print(f"[ok] meta   {meta_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build v1 weekly features from commit-weekly CSV.")
    # Arguments are now just filenames/patterns, NOT full paths.
    # The script knows where to look (v1_data/02_dat).
    parser.add_argument("--commits-pattern", default="02_gh_commit_weekly_*.csv",
                        help="Filename pattern for commit CSV (inside v1_data/02_dat)")
    parser.add_argument("--repo-meta-pattern", default="02_gh_repo_meta_*.csv",
                        help="(Optional) Filename pattern for meta CSV (inside v1_data/02_dat)")
    parser.add_argument("--out", default="auto",
                        help="Output path. Default: auto-generated in v1_data/03_features")
    args = parser.parse_args()

    main(commits_pattern=args.commits_pattern, meta_pattern=args.repo_meta_pattern, out_path_arg=args.out)