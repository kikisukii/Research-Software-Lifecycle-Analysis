# check_issues_extreme.py
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent
D02 = ROOT / "v2_data" / "02_dat"
D03 = ROOT / "v2_data" / "03_dat"

# --- auto-detect STAMP from features file ---
cands = sorted(D03.glob("03_features_weekly_v2_*.csv"))
assert cands, "No features file found in v2_data/03_dat/"
m = re.search(r"_(\d{8}_\d{6})\.csv$", cands[-1].name)
STAMP = m.group(1)

f_feat   = D03 / f"03_features_weekly_v2_{STAMP}.csv"
f_issues = D02 / f"02_gh_issues_{STAMP}.csv"

# --- load features; ensure UTC awareness on week_dt ---
df = pd.read_csv(f_feat)
df["week_dt"] = pd.to_datetime(df["week_dt"], utc=True, errors="coerce")

# pick the row with the maximum issues_closed_8w_count
row = df.sort_values("issues_closed_8w_count", ascending=False).iloc[0]
repo     = row["repo"]
week_dt  = row["week_dt"]  # Sunday 00:00:00 UTC anchor of the LAST week in the 8w window
max_count = int(row["issues_closed_8w_count"])

# 8w window aligned with step-03 logic:
# weeks: [week_dt-7w, ..., week_dt], each week is [Sunday 00:00, next Sunday 00:00)
win_start = week_dt - pd.Timedelta(weeks=7)      # inclusive
win_end   = week_dt + pd.Timedelta(weeks=1)      # exclusive (right-open)

print(f"STAMP = {STAMP}")
print(f"Top repo: {repo}")
print(f"issues_closed_8w_count (from 03) = {max_count}")
print(f"8w window (UTC, right-open): [{win_start} , {win_end})")

# --- load issues and filter to the same 8w window ---
iss = pd.read_csv(f_issues)
iss["closed_at"] = pd.to_datetime(iss["closed_at"], utc=True, errors="coerce")

sub = iss[
    (iss["repo"] == repo)
    & (iss["closed_at"] >= win_start)
    & (iss["closed_at"] <  win_end)   # RIGHT-OPEN to match weekly aggregation
]

print(f"02 issues closed in window (recount) = {len(sub)}")
if len(sub) != max_count:
    print(f"[WARN] Recount != features max ({len(sub)} vs {max_count}). "
          f"Check time bounds / tz or duplicates if any.")

# show a small sample for manual spot-check on GitHub
cols = [c for c in ["number","title","state","created_at","closed_at","user_login"] if c in sub.columns]
print(sub[cols].head(10).to_string(index=False))

# Optional: quick sanity—top closers in this window (often bots for bulk closes)
if "user_login" in sub.columns:
    top_closers = sub["user_login"].value_counts().head(5)
    print("\nTop closers in window:")
    print(top_closers.to_string())
