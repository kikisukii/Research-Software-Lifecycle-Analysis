# -*- coding: utf-8 -*-
"""
06_full_compare_run.py
Function: Automatically finds common repositories between V1 and V2, randomly samples 10,
and drives both plotting scripts to generate comparison charts.
"""

import sys
import subprocess
import random
from pathlib import Path
import pandas as pd
import numpy as np

# --- Configuration ---
# V1 Plotting Script Path
SCRIPT_V1 = Path("v1/05_c_v1_8w.py") # Note: Updated to match previous file name provided
# V2 Plotting Script Path (Ensure this is the version with updated labels)
SCRIPT_V2 = Path("v2/05c_plot.py")

# Data Directories (Used to find common repos)
DIR_V1_DATA = Path("v1_data/05_b_apply")
DIR_V2_DATA = Path("v2_data/05_b")


def get_v1_repos():
    # Find the latest V1 labels file
    cands = sorted(DIR_V1_DATA.glob("05_labels_v1_*.csv"))
    if not cands:
        print("[Error] V1 data not found (v1_data/05_b_apply/05_labels_v1_*.csv)")
        return set()
    print(f"[Data] V1: {cands[-1].name}")
    df = pd.read_csv(cands[-1])
    return set(df["repo"].unique())


def get_v2_repos():
    # Find the latest V2 assignments file
    cands = sorted(DIR_V2_DATA.glob("05b_assignments_with_stage_AND_DEAD_v2_*.csv"))
    if not cands:
        # Try finding the version without DEAD rows
        cands = sorted(DIR_V2_DATA.glob("05b_assignments_with_stage_v2_*.csv"))

    if not cands:
        print("[Error] V2 data not found (v2_data/05_b/...)")
        return set()

    print(f"[Data] V2: {cands[-1].name}")
    df = pd.read_csv(cands[-1])
    return set(df["repo"].unique())


def main():
    print(">>> Comparing V1 and V2 repository lists...")
    repos_v1 = get_v1_repos()
    repos_v2 = get_v2_repos()

    if not repos_v1 or not repos_v2:
        print("Data missing, cannot proceed.")
        sys.exit(1)

    # 1. Intersection: Ensure selected repos exist in both datasets
    common_repos = sorted(list(repos_v1 & repos_v2))
    print(f"    V1 Repo Count: {len(repos_v1)}")
    print(f"    V2 Repo Count: {len(repos_v2)}")
    print(f"    Common Repo Count: {len(common_repos)}")

    if not common_repos:
        print("[Error] No common repos between V1 and V2! Check data sources.")
        sys.exit(1)

    # 2. Sampling (Or you can hardcode a list here)
    # E.g.: targets = ["owner/repo1", "owner/repo2"]
    SAMPLE_SIZE = 10
    rng = np.random.default_rng(42)  # Fixed seed, ensures consistent selection

    if len(common_repos) > SAMPLE_SIZE:
        targets = rng.choice(common_repos, size=SAMPLE_SIZE, replace=False).tolist()
    else:
        targets = common_repos

    print(f"\n>>> Selected the following {len(targets)} repos for comparison:")
    for r in targets:
        print(f"  - {r}")

    # 3. Construct Arguments
    # --repo r1 --repo r2 ...
    cmd_args = []
    for r in targets:
        cmd_args.append("--repo")
        cmd_args.append(r)

    # 4. Run V1 Plotting
    print(f"\n>>> [1/2] Running V1 Plotting ({SCRIPT_V1})...")
    if not SCRIPT_V1.exists():
        print(f"[Error] V1 script not found: {SCRIPT_V1}")
    else:
        # V1 is already set up, no extra args needed
        try:
            subprocess.run([sys.executable, str(SCRIPT_V1)] + cmd_args, check=True)
        except Exception as e:
            print(f"V1 Execution Failed: {e}")

    # 5. Run V2 Plotting
    print(f"\n>>> [2/2] Running V2 Plotting ({SCRIPT_V2})...")
    if not SCRIPT_V2.exists():
        print(f"[Error] V2 script not found: {SCRIPT_V2}")
    else:
        # V2 default smooth=3 is set in code, no args needed here
        try:
            subprocess.run([sys.executable, str(SCRIPT_V2)] + cmd_args, check=True)
        except Exception as e:
            print(f"V2 Execution Failed: {e}")

    print("\n" + "=" * 50)
    print("Comparison generation complete! Please compare the following two folders:")
    print(f"V1 (8w): v1_data/05_c_viz_8w/")
    print(f"V2 (8w): v2_data/05_c_viz/")
    print("=" * 50)


if __name__ == "__main__":
    main()