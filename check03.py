# check_na_entrypoints.py
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent
D02 = ROOT / "v2_data" / "02_dat"
D03 = ROOT / "v2_data" / "03_dat"

# auto-detect STAMP from features file
cands = sorted(D03.glob("03_features_weekly_v2_*.csv"))
assert cands, "No features file found in v2_data/03_dat/"
m = re.search(r"_(\d{8}_\d{6})\.csv$", cands[-1].name)
STAMP = m.group(1)

f_api = D02 / f"02_api_status_{STAMP}.csv"
f_feat = D03 / f"03_features_weekly_v2_{STAMP}.csv"
f_drop = D03 / f"03_features_dropped_v2_{STAMP}.csv"  # may not exist if 0 rows

df_api = pd.read_csv(f_api)
df_feat = pd.read_csv(f_feat)

# repos with not_available (issues / releases)
na_issues_repos   = set(df_api.loc[df_api["issues_api_status"]=="not_available","repo"])
na_releases_repos = set(df_api.loc[df_api["releases_api_status"]=="not_available","repo"])
feat_repos        = set(df_feat["repo"].unique())

print(f"STAMP = {STAMP}")
print(f"02: issues not_available repos = {len(na_issues_repos)}")
print(f"02: releases not_available repos = {len(na_releases_repos)}")
print(f"03: repos in features          = {len(feat_repos)}")

# intersections
int_iss = na_issues_repos & feat_repos
int_rel = na_releases_repos & feat_repos
print(f"INTERSECTION (issues NA ∩ features)   = {len(int_iss)}")
print(f"INTERSECTION (releases NA ∩ features) = {len(int_rel)}")

# dropped windows (if file exists)
if f_drop.exists():
    df_drop = pd.read_csv(f_drop)
    print(f"dropped rows: {len(df_drop)}")
    if len(df_drop):
        print("Top-5 reasons:")
        print(df_drop["na_cols"].value_counts().head(5))
else:
    print("No dropped file exists (consistent with qc=0).")
