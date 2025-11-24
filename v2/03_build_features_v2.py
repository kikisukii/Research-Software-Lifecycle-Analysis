# -*- coding: utf-8 -*-
"""
v2/03_build_features_v2.py

Build 8-week (8w) features from step-02 artifacts for v2.

Key decisions:
- Auto-detect latest STAMP under v2_data/02_dat/.
- Per-repo weekly grid anchored at Sunday 00:00:00 UTC.
- Fill 0 only for true zero activity; "not_available" stays NA.
- issues: not_attempted -> 0; not_available -> NA.
- releases: not_available -> NA.
- contributors_8w_unique from contributors_raw via sliding 8w union (email-first, fallback name).
- Complete-case over four numeric 8w features; record dropped windows (with reasons).
- Zero/Low windows included; dead_flag (24→25 zero-commit weeks) for reporting only.

Outputs:
  - v2_data/03_dat/03_features_weekly_v2_<STAMP>.csv
  - v2_data/03_dat/03_features_qc_v2_<STAMP>.json
  - v2_data/03_dat/03_features_dropped_v2_<STAMP>.csv
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter, deque
from typing import TextIO
import re, json
import numpy as np
import pandas as pd

# ---------------- Paths & STAMP discovery ----------------
THIS_FILE = Path(__file__).resolve()
THIS_DIR  = THIS_FILE.parent
DATA_ROOT = (THIS_DIR.parent / "v2_data").resolve()
D02 = DATA_ROOT / "02_dat"
D03 = DATA_ROOT / "03_dat"
D03.mkdir(parents=True, exist_ok=True)

STAMP_RE = re.compile(r"_(\d{8}_\d{6})\.")

def find_latest_stamp(d: Path) -> str:
    stamps = []
    for p in d.glob("02_*.*"):
        m = STAMP_RE.search(p.name)
        if m:
            stamps.append(m.group(1))
    if not stamps:
        raise FileNotFoundError("No timestamped 02_* files under v2_data/02_dat/")
    return sorted(set(stamps))[-1]

STAMP = find_latest_stamp(D02)

# ---------------- Utilities ----------------
def to_utc_ts(x):
    """Scalar/array-like to UTC Timestamp(s). Invalid -> NaT."""
    return pd.to_datetime(x, utc=True, errors="coerce")

def sunday_anchor_utc(x):
    """
    Map timestamps to last Sunday's 00:00:00 UTC.
    Accepts scalar or vector (Series/Index/array-like).
    """
    ts = to_utc_ts(x)
    if isinstance(ts, pd.Timestamp):  # scalar path
        if pd.isna(ts):
            return ts
        wd = (ts.weekday() + 1) % 7  # Mon=0,...,Sun=6
        return (ts - pd.Timedelta(days=wd)).normalize()
    # vector path
    delta = (ts.dt.weekday + 1) % 7
    return (ts - pd.to_timedelta(delta, unit="D")).dt.normalize()

def week_unix_from_dtindex(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.asi8 // 10**9).astype("int64")

def ensure_cols(df: pd.DataFrame, candidates: list[str], logical_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    head = list(df.columns)[:16]
    raise KeyError(f"Missing column for {logical_name}. Tried {candidates}. Got {head} ...")

def pstats(ser: pd.Series) -> dict:
    ser = ser.dropna()
    if ser.empty:
        return {}
    q = ser.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": float(ser.min()),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "max": float(ser.max()),
    }

def repo_week_grid(min_dt, max_dt) -> pd.DatetimeIndex:
    start = sunday_anchor_utc(pd.Timestamp(min_dt))
    end   = sunday_anchor_utc(pd.Timestamp(max_dt))
    return pd.date_range(start, end, freq="W-SUN", tz="UTC")

# ---------------- Read step-02 inputs ----------------
f_commits   = D02 / f"02_gh_commit_weekly_{STAMP}.csv"
f_contrib_r = D02 / f"02_contributors_raw_{STAMP}.csv"
f_issues    = D02 / f"02_gh_issues_{STAMP}.csv"
f_releases  = D02 / f"02_gh_releases_{STAMP}.csv"
f_status    = D02 / f"02_api_status_{STAMP}.csv"

df_comm = pd.read_csv(f_commits)
df_raw  = pd.read_csv(f_contrib_r)
df_iss  = pd.read_csv(f_issues)
df_rel  = pd.read_csv(f_releases)
df_sta  = pd.read_csv(f_status)

# Column mappings (robust)
col_repo_c = ensure_cols(df_comm, ["repo","repository","full_name"], "repo(commits)")
col_week_c = ensure_cols(df_comm, ["week_unix","week","week_start_unix"], "week_unix(commits)")
col_comm   = ensure_cols(df_comm, ["commits","n_commits","commit_count","count"], "commits")

col_repo_r = ensure_cols(df_raw,  ["repo","repository","full_name"], "repo(raw)")
col_eml    = ensure_cols(df_raw,  ["author_email","email","committer_email"], "email(raw)")
col_name   = ensure_cols(df_raw,  ["author_name","name","committer_name"], "name(raw)")
col_cdate  = ensure_cols(df_raw,  ["commit_date_utc","commit_date","committed_date","author_date"], "commit_date(raw)")

col_repo_i = ensure_cols(df_iss,  ["repo","repository","full_name"], "repo(issues)")
col_iclose = ensure_cols(df_iss,  ["closed_at","closed_at_utc","closed"], "closed_at(issues)")

col_repo_l = ensure_cols(df_rel,  ["repo","repository","full_name"], "repo(releases)")
col_rdate  = ensure_cols(df_rel,  ["published_at","published_at_utc","created_at"], "published_at(releases)")

col_repo_s = ensure_cols(df_sta,  ["repo","repository","full_name"], "repo(status)")
col_ista   = ensure_cols(df_sta,  ["issues_api_status","issues_status"], "issues_api_status")
col_rsta   = ensure_cols(df_sta,  ["releases_api_status","releases_status"], "releases_api_status")

# Normalize
df_comm = df_comm[[col_repo_c, col_week_c, col_comm]].copy()
df_comm.rename(columns={col_repo_c:"repo", col_week_c:"week_unix", col_comm:"commits"}, inplace=True)
df_comm["week_unix"] = df_comm["week_unix"].astype("int64")

df_raw = df_raw[[col_repo_r, col_eml, col_name, col_cdate]].copy()
df_raw.rename(columns={col_repo_r:"repo", col_eml:"email", col_name:"name", col_cdate:"commit_dt"}, inplace=True)
df_raw["commit_dt"] = to_utc_ts(df_raw["commit_dt"])
df_raw["week_dt"]   = sunday_anchor_utc(df_raw["commit_dt"])

emails = df_raw["email"].astype("string").str.strip().str.lower()
mask_blank = emails.isna() | (emails == "")
pid = emails.copy()
pid[mask_blank] = "name:" + df_raw["name"].astype("string").str.strip().str.lower()
df_raw["pid"] = pid

df_iss = df_iss[[col_repo_i, col_iclose]].copy()
df_iss.rename(columns={col_repo_i:"repo", col_iclose:"closed_dt"}, inplace=True)
df_iss["closed_dt"] = to_utc_ts(df_iss["closed_dt"])
df_iss["week_dt"]   = sunday_anchor_utc(df_iss["closed_dt"])

df_rel = df_rel[[col_repo_l, col_rdate]].copy()
df_rel.rename(columns={col_repo_l:"repo", col_rdate:"pub_dt"}, inplace=True)
df_rel["pub_dt"]  = to_utc_ts(df_rel["pub_dt"])
df_rel["week_dt"] = sunday_anchor_utc(df_rel["pub_dt"])

df_sta = df_sta[[col_repo_s, col_ista, col_rsta]].copy()
df_sta.rename(columns={col_repo_s:"repo", col_ista:"issues_status", col_rsta:"releases_status"}, inplace=True)

# ---------------- Build weekly series ----------------
comm_by_repo = {r: g.set_index("week_unix")["commits"].sort_index()
                for r, g in df_comm.groupby("repo")}

iss_by_repo = {}
for r, g in df_iss.groupby("repo"):
    iss_by_repo[r] = g.groupby("week_dt").size().astype(int)

rel_by_repo = {}
for r, g in df_rel.groupby("repo"):
    rel_by_repo[r] = g.groupby("week_dt").size().astype(int)

raw_by_repo = {}
for r, g in df_raw.groupby("repo"):
    raw_by_repo[r] = g.groupby("week_dt")["pid"].apply(lambda vals: set(vals.dropna()))

sta = df_sta.set_index("repo").to_dict(orient="index")
repos = sorted(set(df_comm["repo"]))

# ---------------- Main loop ----------------
rows, dropped = [], []

for repo in repos:
    # bounds from any available signal
    bounds = []
    if repo in comm_by_repo and not comm_by_repo[repo].empty:
        w = pd.to_datetime(comm_by_repo[repo].index, unit="s", utc=True)
        bounds += [w.min(), w.max()]
    if repo in iss_by_repo and not iss_by_repo[repo].empty:
        bounds += [iss_by_repo[repo].index.min(), iss_by_repo[repo].index.max()]
    if repo in rel_by_repo and not rel_by_repo[repo].empty:
        bounds += [rel_by_repo[repo].index.min(), rel_by_repo[repo].index.max()]
    if repo in raw_by_repo and not raw_by_repo[repo].empty:
        bounds += [raw_by_repo[repo].index.min(), raw_by_repo[repo].index.max()]
    if not bounds:
        continue

    grid = repo_week_grid(min(bounds), max(bounds))
    grid_unix = week_unix_from_dtindex(grid)

    # commits: zero-filled weekly series
    comm = pd.Series(0, index=grid, dtype="int64")
    if repo in comm_by_repo:
        tmp = comm_by_repo[repo]
        comm.loc[pd.to_datetime(tmp.index, unit="s", utc=True)] = tmp.values

    # issues
    iss_status = sta.get(repo, {}).get("issues_status", "ok")
    if iss_status == "not_available":
        iss = pd.Series(np.nan, index=grid, dtype="float")
    else:
        iss = pd.Series(0, index=grid, dtype="int64")
        if repo in iss_by_repo:
            iss.loc[iss_by_repo[repo].index] = iss_by_repo[repo].values

    # releases
    rel_status = sta.get(repo, {}).get("releases_status", "ok")
    if rel_status == "not_available":
        rel = pd.Series(np.nan, index=grid, dtype="float")
    else:
        rel = pd.Series(0, index=grid, dtype="int64")
        if repo in rel_by_repo:
            rel.loc[rel_by_repo[repo].index] = rel_by_repo[repo].values

    # contributors weekly sets (fill missing with empty set)
    weekly_sets = {dt: set() for dt in grid}
    if repo in raw_by_repo:
        weekly_sets.update(raw_by_repo[repo].to_dict())

    # dead_flag: from 25th consecutive zero-commit week onwards
    consec_zero = 0
    dead_week = pd.Series(0, index=grid, dtype="int8")
    for i in range(len(grid)):
        consec_zero = consec_zero + 1 if comm.iloc[i] == 0 else 0
        if consec_zero >= 25:
            dead_week.iloc[i] = 1

    # ---------- FIX: compute rolling series directly (no slice assignment) ----------
    c8 = comm.rolling(8, min_periods=8).sum().astype(float)
    i8 = iss.rolling(8, min_periods=8).sum()           # all-NA remains all-NA when status=not_available
    r8 = rel.rolling(8, min_periods=8).sum()

    # contributors 8w unique via sliding union
    u8 = pd.Series(np.nan, index=grid, dtype="float")
    dq = deque()
    cnt = Counter()
    for i, dt in enumerate(grid):
        cur = weekly_sets[dt]
        dq.append(cur)
        for pid in cur:
            cnt[pid] += 1
        if len(dq) > 8:
            out = dq.popleft()
            for pid in out:
                cnt[pid] -= 1
                if cnt[pid] <= 0:
                    del cnt[pid]
        if len(dq) == 8:
            u8.iloc[i] = float(len(cnt))

    # assemble rows (only after week 8)
    for i, dt in enumerate(grid):
        if i < 7:
            continue
        row = {
            "repo": repo,
            "week_unix": int(grid_unix[i]),
            "week_dt":   str(dt),
            "commits_8w_sum": c8.iloc[i],
            "contributors_8w_unique": u8.iloc[i],
            "issues_closed_8w_count": i8.iloc[i],
            "releases_8w_count": r8.iloc[i],
            "has_release_8w": (pd.NA if pd.isna(r8.iloc[i]) else int(r8.iloc[i] > 0)),
            "dead_flag": int(dead_week.iloc[i]),
        }
        num_cols = ["commits_8w_sum","contributors_8w_unique","issues_closed_8w_count","releases_8w_count"]
        na_cols = [c for c in num_cols if pd.isna(row[c])]
        if na_cols:
            dropped.append((repo, int(grid_unix[i]), ",".join(na_cols)))
            continue
        rows.append(row)

# ---------------- Build DataFrames ----------------
feat = pd.DataFrame(rows).sort_values(["repo","week_unix"]).reset_index(drop=True)
dropped_df = pd.DataFrame(dropped, columns=["repo","week_unix","na_cols"]) if dropped else \
             pd.DataFrame(columns=["repo","week_unix","na_cols"])

# ---------------- QC summary ----------------
_zero_mask = (
    (feat[["commits_8w_sum","contributors_8w_unique","issues_closed_8w_count","releases_8w_count"]] == 0)
    .all(axis=1)
) if not feat.empty else pd.Series(dtype=bool)

qc = {
    "stamp": STAMP,
    "n_rows": int(len(feat)),
    "n_repos": int(feat["repo"].nunique()) if not feat.empty else 0,
    "dropped_complete_case_rows": int(len(dropped_df)),
    "dropped_ratio": float(len(dropped_df) / (len(feat) + len(dropped_df) + 1e-9)),
    "distributions": {
        "commits_8w_sum": pstats(feat["commits_8w_sum"]) if not feat.empty else {},
        "contributors_8w_unique": pstats(feat["contributors_8w_unique"]) if not feat.empty else {},
        "issues_closed_8w_count": pstats(feat["issues_closed_8w_count"]) if not feat.empty else {},
        "releases_8w_count": pstats(feat["releases_8w_count"]) if not feat.empty else {},
    },
    "has_release_8w_ratio": float(feat["has_release_8w"].eq(1).astype(float).mean()) if not feat.empty else 0.0,
    "zero_8w_ratio": float(_zero_mask.astype(float).mean()) if not feat.empty else 0.0,
    "dead_flag_ratio": float((feat["dead_flag"]==1).astype(float).mean()) if not feat.empty else 0.0,
}

# ---------------- Save ----------------
out_csv = D03 / f"03_features_weekly_v2_{STAMP}.csv"
feat.to_csv(out_csv, index=False)

out_qc  = D03 / f"03_features_qc_v2_{STAMP}.json"
with open(out_qc, "w", encoding="utf-8") as f:  # type: TextIO
    json.dump(qc, f, ensure_ascii=False, indent=2)

if len(dropped_df):
    out_drop = D03 / f"03_features_dropped_v2_{STAMP}.csv"
    dropped_df.to_csv(out_drop, index=False)
    print(f"[INFO] dropped complete-case rows: {len(dropped_df)} -> {out_drop}")

print(f"[OK] 03 features saved: {out_csv}")
print(f"[OK] QC summary saved: {out_qc}")
