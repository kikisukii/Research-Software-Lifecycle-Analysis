# -*- coding: utf-8 -*-
"""
v2/05b_name_and_label_v2.py — Stage naming & DATA RECOVERY.

CRITICAL FIX:
  - Merges ALL columns (Raw + 8w) from 03_features into the assignments.
  - Ensures 05c has access to 'commits' (weekly) AND 'commits_8w_sum'.
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ---------------- paths ----------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
D03 = ROOT / "v2_data" / "03_dat"
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
print(f"[INFO] Processing STAMP={STAMP} | K={K}")

f_assign = D05_A / f"05a_cluster_assignments_v2_{STAMP}_K{K}.csv"
f_prof = D05_A / f"05a_cluster_profiles_v2_{STAMP}_K{K}.csv"

# 1. Load Clusters
assign_raw = pd.read_csv(f_assign, parse_dates=["week_dt"])
prof = pd.read_csv(f_prof)

# 2. Load Features (This contains RAW metrics + 8w metrics)
cands_03 = sorted(D03.glob(f"03_features_weekly_v2_*.csv"))
f_feat03 = None
for f in cands_03:
    if STAMP in f.name:
        f_feat03 = f
        break
if not f_feat03 and cands_03: f_feat03 = cands_03[-1]

if not f_feat03 or not f_feat03.exists():
    raise FileNotFoundError(f"Missing 03 features file for stamp {STAMP}")

print(f"[INFO] Loading features from: {f_feat03.name}")
feat03 = pd.read_csv(f_feat03, parse_dates=["week_dt"])

# 3. MERGE Features into Assignments (The Fix)
# We merge everything from feat03 into assign_raw
print("[INFO] Merging metric columns into assignments...")
# Identify columns to add (exclude join keys)
# [Fix] Check existing columns in the left DF to prevent duplicates and '_x' suffixes
existing_cols = set(assign_raw.columns)
cols_to_add = [c for c in feat03.columns
               if c not in existing_cols
               and c not in ["repo", "week_unix", "week_dt", "dead_flag"]]
assign = pd.merge(assign_raw, feat03[["repo", "week_unix"] + cols_to_add],
                  on=["repo", "week_unix"], how="left")

# 4. MAPPING
CLUSTER_LABEL_MAP = {
    3: "Peak Activity",
    1: "Internal Development",
    2: "Release Phase",
    5: "Maintenance",
    4: "Baseline",
    0: "Dormant"
}

prof["stage_name"] = prof["cluster"].map(CLUSTER_LABEL_MAP)
assign["stage_name"] = assign["cluster"].map(CLUSTER_LABEL_MAP)


# Durations
def compute_durations(df_assign: pd.DataFrame) -> pd.DataFrame:
    df = df_assign.sort_values(["repo", "week_unix"]).copy()
    change = (df["repo"] != df["repo"].shift(1)) | (df["cluster"] != df["cluster"].shift(1))
    run_id = change.cumsum()
    runs = df.groupby(run_id).agg(cluster=("cluster", "first"), n_weeks=("week_unix", "size"))
    return runs


runs = compute_durations(assign)
dur_stats = runs.groupby("cluster")["n_weeks"].agg(
    dur_p50_weeks="median"
).reset_index()
prof = prof.merge(dur_stats, on="cluster", how="left")

# 5. MERGE DEAD ROWS
dead_rows = feat03[feat03["dead_flag"] == 1].copy()

if not dead_rows.empty:
    print(f"[INFO] Merging {len(dead_rows)} DEAD rows...")
    dead_final = dead_rows.copy()
    dead_final["cluster"] = -1
    dead_final["cluster_rank_by_commits"] = -1
    dead_final["stage_name"] = "Dead"

    # Ensure columns match
    common_cols = [c for c in assign.columns if c in dead_final.columns]
    full_df = pd.concat([assign[common_cols], dead_final[common_cols]], ignore_index=True)
else:
    print("[INFO] No DEAD rows found.")
    full_df = assign.copy()

full_df = full_df.sort_values(["repo", "week_unix"])

# 6. OUTPUTS
map_csv = D05_B / f"05b_cluster_label_map_v2_{STAMP}_K{K}.csv"
# Only save available columns
map_save_cols = [c for c in ["cluster", "stage_name", "n_windows", "commits_8w_sum_p50"] if c in prof.columns]
prof[map_save_cols].to_csv(map_csv, index=False)

out_prof = D05_B / f"05b_profiles_with_stage_v2_{STAMP}_K{K}.csv"
prof.to_csv(out_prof, index=False)

out_full = D05_B / f"05b_assignments_with_stage_AND_DEAD_v2_{STAMP}_K{K}.csv"
full_df.to_csv(out_full, index=False)

# Summary
current_stage_df = full_df.sort_values("week_unix").groupby("repo").last().reset_index()
summ_cols = [c for c in ["repo", "week_unix", "stage_name", "cluster", "commits_8w_sum"] if
             c in current_stage_df.columns]
current_stage_df[summ_cols].to_csv(D05_B / f"05b_current_stage_summary_v2_{STAMP}_K{K}.csv", index=False)

print(f"[OK] Full Timeline Saved: {out_full.name}")