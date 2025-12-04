# -*- coding: utf-8 -*-
"""
v2/extra02_05d_compare_run.py

This is an extra code, when I'm using to check 8w or weekly.

Purpose: Randomly select 10 repositories from the data, then run BOTH the "8-week" and
"Weekly" plotting scripts on the SAME set of repositories.
This ensures both output folders contain plots for the identical repo set.
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# --- Config: confirm your script filenames ---

SCRIPT_8W = "05c_plot_v2_8w.py"
SCRIPT_WEEKLY = "extra01_05c_test_weekly.py"

# --- Paths ---
THIS = Path(__file__).resolve()
DIR_V2 = THIS.parent
DATA_DIR = DIR_V2.parent / "v2_data" / "05_b_apply"


def get_latest_data():
    """Return the newest 05b assignment file."""
    candidates = sorted(DATA_DIR.glob("05b_assignments_with_stage_AND_DEAD_v2_*.csv"))
    if not candidates:
        print(f"[ERROR] No 05b data found in {DATA_DIR}. Please run steps 03 and 05b first.")
        sys.exit(1)
    return candidates[-1]


def main():
    # 1) Read repository list
    csv_file = get_latest_data()
    print(f"[1/4] Loading data: {csv_file.name}")
    df = pd.read_csv(csv_file)
    all_repos = sorted(df["repo"].unique())

    # 2) Randomly sample 10 repos
    rng = np.random.default_rng(42)  # Fixed seed for reproducible selection
    targets = rng.choice(all_repos, size=10, replace=False)
    print(f"[2/4] Selected repositories (n=10):")
    for r in targets:
        print(f"  - {r}")

    # 3) Build command-line args for the plotting scripts
    # Format: python script.py --repo r1 --repo r2 ...
    cmd_args = []
    for r in targets:
        cmd_args.append("--repo")
        cmd_args.append(r)

    # 4) Run the 8-week plotter (05c_plot_v2_8w.py)
    print(f"\n[3/4] Running 8-week plotting script ({SCRIPT_8W})...")
    try:
        subprocess.run([sys.executable, str(DIR_V2 / SCRIPT_8W)] + cmd_args, check=True)
    except Exception as e:
        print(f"[ERROR] 8-week plotting failed: {e}")

    # 5) Run the Weekly plotter (extra01_05c_test_weekly.py)
    print(f"\n[4/4] Running Weekly plotting script ({SCRIPT_WEEKLY})...")
    try:
        subprocess.run([sys.executable, str(DIR_V2 / SCRIPT_WEEKLY)] + cmd_args, check=True)
    except Exception as e:
        print(f"[ERROR] Weekly plotting failed: {e}")

    print("\n" + "=" * 40)
    print("Comparison completed! Please check the following folders:")
    print("1) 8-week plots:      v2_data/05_c_viz/")
    print("2) Weekly plots:      v2_data/05_c_weekly_test/")
    print("=" * 40)


if __name__ == "__main__":
    main()
