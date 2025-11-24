# -*- coding: utf-8 -*-
"""
v2/05b_name_and_label_v2.py — Stage naming for clusters (K fixed, here K=6).
Reads 05a outputs from v2_data/05_a/, writes labeled tables to v2_data/05_b/.

Outputs:
  1) v2_data/05_b/05b_cluster_label_map_v2_<STAMP>_K6.csv
     (cluster -> suggested_stage + final stage_name; editable for future runs)
  2) v2_data/05_b/05b_profiles_with_stage_v2_<STAMP>_K6.csv
     (05a profiles + stage_name + typical duration stats)
  3) v2_data/05_b/05b_assignments_with_stage_v2_<STAMP>_K6.csv
     (05a assignments + stage_name; ORIGINAL features preserved — NOT PCA)

Naming:
  - Suggested (auditable) rules:
      Zero/Low: (zero_8w_rate >= 0.80) OR (all four p75 == 0)
      Peak:     commits_p50 >= median(commits_p50 across clusters)
                AND has_release_8w_rate >= median(has_release_8w_rate across clusters)
      Rising:   commits_p50 >= median AND has_release_8w_rate < median
      Cooling:  commits_p50 <  median AND has_release_8w_rate >= median
      Low:      otherwise
  - Final names (this batch, per your decision): use cluster_rank_by_commits order:
      rank 0..5 -> ["Rising-Dev","Peak-Release","Maintenance/Triage","Cooling-Release","Low","Zero/Low"]
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from typing import List

# ---------------- paths & latest stamp ----------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
D05_A = ROOT / "v2_data" / "05_a"
D05_B = ROOT / "v2_data" / "05_b"
D05_B.mkdir(parents=True, exist_ok=True)

def latest_stamp_from_05a(d: Path):
    cands = sorted(d.glob("05a_cluster_assignments_v2_*_K*.csv"))
    if not cands:
        raise FileNotFoundError("No 05a assignments under v2_data/05_a/")
    m = re.search(r"_v2_(\d{8}_\d{6})_K(\d+)\.csv$", cands[-1].name)
    if not m:
        raise RuntimeError(f"Cannot parse STAMP/K from: {cands[-1].name}")
    return m.group(1), int(m.group(2))

STAMP, K = latest_stamp_from_05a(D05_A)

f_assign = D05_A / f"05a_cluster_assignments_v2_{STAMP}_K{K}.csv"
f_prof   = D05_A / f"05a_cluster_profiles_v2_{STAMP}_K{K}.csv"

assign = pd.read_csv(f_assign, parse_dates=["week_dt"])
prof   = pd.read_csv(f_prof)

# ---------------- sanity checks ----------------
prof_required = {
    "cluster","cluster_rank_by_commits","n_windows","has_release_8w_rate","zero_8w_rate",
    "commits_8w_sum_p25","commits_8w_sum_p50","commits_8w_sum_p75",
    "contributors_8w_unique_p25","contributors_8w_unique_p50","contributors_8w_unique_p75",
    "issues_closed_8w_count_p25","issues_closed_8w_count_p50","issues_closed_8w_count_p75",
    "releases_8w_count_p25","releases_8w_count_p50","releases_8w_count_p75",
}
missing = prof_required - set(prof.columns)
if missing:
    raise KeyError(f"Profiles missing columns: {sorted(missing)}")

assign_required = {
    "repo","week_unix","week_dt","cluster","cluster_rank_by_commits",
    "commits_8w_sum","contributors_8w_unique","issues_closed_8w_count","releases_8w_count"
}
miss2 = assign_required - set(assign.columns)
if miss2:
    raise KeyError(f"Assignments missing columns: {sorted(miss2)}")

# ---------------- suggested names (auditable rules) ----------------
def all_four_p75_zero(r):
    return (
        (r["commits_8w_sum_p75"] == 0) and
        (r["contributors_8w_unique_p75"] == 0) and
        (r["issues_closed_8w_count_p75"] == 0) and
        (r["releases_8w_count_p75"] == 0)
    )

commits_med = float(prof["commits_8w_sum_p50"].median())
hasrel_med  = float(prof["has_release_8w_rate"].median())

def suggest_stage(r):
    if (r["zero_8w_rate"] >= 0.80) or all_four_p75_zero(r):
        return "Zero/Low"
    c50 = float(r["commits_8w_sum_p50"])
    hasrel = float(r["has_release_8w_rate"])
    if (c50 >= commits_med) and (hasrel >= hasrel_med):
        return "Peak"
    if (c50 >= commits_med) and (hasrel <  hasrel_med):
        return "Rising"
    if (c50 <  commits_med) and (hasrel >= hasrel_med):
        return "Cooling"
    return "Low"

prof["stage_name_suggested"] = prof.apply(suggest_stage, axis=1)

# ---------------- final names for THIS BATCH (your decided version) ----------------
# Map by cluster_rank_by_commits order (0..5), making it robust to raw cluster ids.
names_by_rank = ["Rising-Dev","Peak-Release","Maintenance/Triage","Cooling-Release","Low","Zero/Low"]
rank2name = {rank: names_by_rank[rank] for rank in range(len(names_by_rank))}

prof["stage_name"] = prof["cluster_rank_by_commits"].map(rank2name)
# Fall back to suggested if for any reason rank is out of range
prof["stage_name"] = prof["stage_name"].fillna(prof["stage_name_suggested"])

# ---------------- optional: typical duration (weeks) per cluster ----------------
# For each repo, sort by week_unix and compute consecutive run lengths in the same cluster.
def compute_durations(df_assign: pd.DataFrame) -> pd.DataFrame:
    df = df_assign.sort_values(["repo","week_unix"]).copy()
    # identify breaks where cluster changes (within repo)
    change = (df["repo"] != df["repo"].shift(1)) | (df["cluster"] != df["cluster"].shift(1))
    run_id = change.cumsum()
    runs = df.groupby(run_id).agg(
        repo=("repo","first"),
        cluster=("cluster","first"),
        start_week=("week_unix","min"),
        end_week=("week_unix","max"),
        n_weeks=("week_unix","size")
    ).reset_index(drop=True)
    return runs

runs = compute_durations(assign)
dur_stats = (
    runs.groupby("cluster")["n_weeks"]
        .agg(dur_p25_weeks=lambda s: float(s.quantile(0.25)),
             dur_p50_weeks="median",
             dur_p75_weeks=lambda s: float(s.quantile(0.75)))
        .reset_index()
)

prof2 = prof.merge(dur_stats, on="cluster", how="left")

# ---------------- build editable label map ----------------
map_cols = [
    "cluster","cluster_rank_by_commits","n_windows",
    "has_release_8w_rate","zero_8w_rate",
    "commits_8w_sum_p50","contributors_8w_unique_p50","issues_closed_8w_count_p50","releases_8w_count_p50",
    "stage_name_suggested","stage_name"
]
label_map = prof2[map_cols].copy()
map_csv = D05_B / f"05b_cluster_label_map_v2_{STAMP}_K{K}.csv"
label_map.to_csv(map_csv, index=False)

# ---------------- merge stage back to profiles & assignments ----------------
out_prof   = D05_B / f"05b_profiles_with_stage_v2_{STAMP}_K{K}.csv"
out_assign = D05_B / f"05b_assignments_with_stage_v2_{STAMP}_K{K}.csv"

assign2 = assign.merge(label_map[["cluster","stage_name"]], on="cluster", how="left")
assign2["stage_name"] = assign2["stage_name"].fillna("Unlabeled")

prof2.to_csv(out_prof, index=False)
assign2.to_csv(out_assign, index=False)

print(f"[OK] Saved label map (editable) -> {map_csv}")
print(f"[OK] Saved profiles + stage     -> {out_prof}")
print(f"[OK] Saved assignments + stage  -> {out_assign}")
print(f"[INFO] STAMP={STAMP} | K={K} | clusters={prof['cluster'].nunique()}")

# ===== [Optional] append DEAD windows from 03 into an extra file =====
from pathlib import Path

D03 = ROOT / "v2_data" / "03_dat"
f_feat03 = D03 / f"03_features_weekly_v2_{STAMP}.csv"
if f_feat03.exists():
    feat03 = pd.read_csv(f_feat03, parse_dates=["week_dt"])
    # ensure required columns exist
    need_cols = {
        "repo","week_unix","week_dt","dead_flag",
        "commits_8w_sum","contributors_8w_unique",
        "issues_closed_8w_count","releases_8w_count"
    }
    miss = need_cols - set(feat03.columns)
    if not miss:
        dead = feat03[feat03["dead_flag"] == 1].copy()
        # ensure aux columns to align with assignments
        if "has_release_8w" not in dead.columns:
            dead["has_release_8w"] = (dead["releases_8w_count"] > 0).astype(int)
        dead["zero_8w"] = (
            (dead["commits_8w_sum"]==0) &
            (dead["contributors_8w_unique"]==0) &
            (dead["issues_closed_8w_count"]==0) &
            (dead["releases_8w_count"]==0)
        ).astype(int)

        # align columns with assign2; mark cluster=-1, stage_name='Dead'
        cols_base = [
            "repo","week_unix","week_dt",
            "commits_8w_sum","contributors_8w_unique",
            "issues_closed_8w_count","releases_8w_count",
            "has_release_8w","zero_8w"
        ]
        dead_aligned = dead[cols_base].copy()
        dead_aligned["cluster"] = -1
        dead_aligned["cluster_rank_by_commits"] = -1
        dead_aligned["stage_name"] = "Dead"

        # concat and save an extra file (DOES NOT replace main 05b output)
        assign_with_dead = pd.concat(
            [assign2[dead_aligned.columns], dead_aligned],
            axis=0, ignore_index=True
        ).sort_values(["repo","week_unix"])
        out_assign_dead = D05_B / f"05b_assignments_with_stage_AND_DEAD_v2_{STAMP}_K{K}.csv"
        assign_with_dead.to_csv(out_assign_dead, index=False)
        print(f"[OK] Saved assignments + stage + DEAD -> {out_assign_dead}")
    else:
        print(f"[WARN] Cannot append DEAD (03 missing cols): {sorted(miss)}")
else:
    print(f"[WARN] Cannot append DEAD (03 file missing): {f_feat03}")
