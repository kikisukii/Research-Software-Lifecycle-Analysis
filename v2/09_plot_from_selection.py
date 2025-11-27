# -*- coding: utf-8 -*-
"""
09_plot_from_selection.py
- NO changes to v2/05c*.py are required.
- Dynamically locates and imports the 05c plotter module (must define: plot_repo,
  infer_latest_assignments, infer_stamp_from_name).
- Reads latest (or specified) survey_selection_v2_*.csv.
- Saves PNGs directly to v2_data/09_plot_from_selection/survey_images_<STAMP>_K<K>/.
- Writes final_survey_list_<STAMP>_K<K>.csv for Qualtrics import.

Run:
  python v2/09_plot_from_selection.py
  # or specify selection & smoothing:
  python v2/09_plot_from_selection.py --selection-csv v2_data/08_survey_candidates/survey_selection_v2_20251114_004622_K6.csv --smooth 3
"""

from __future__ import annotations
import argparse
import importlib.util
import re
from pathlib import Path
import pandas as pd

# -------- project paths (robust to cwd) --------------------------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
V2_DIR = ROOT / "v2"
S08_DIR = ROOT / "v2_data" / "08_survey_candidates"
OUT_ROOT = ROOT / "v2_data" / "09_plot_from_selection"

SEL_PAT = re.compile(r"survey_selection_v2_(\d{8}(?:_\d{6})?)_K(\d+)\.csv")

PREFERRED_05C_NAMES = ["05c_plot_v2_8w.py"]


# ---------- helpers -----------------------------------------------------------
def pick_latest_selection() -> Path:
    files = sorted(S08_DIR.glob("survey_selection_v2_*_K*.csv"))
    if not files:
        raise FileNotFoundError("No survey_selection_v2_*_K*.csv under v2_data/08_survey_candidates/")
    return files[-1]

def parse_stamp_k(selection_path: Path) -> tuple[str, str]:
    m = SEL_PAT.search(selection_path.name)
    if not m:
        return "unknown", ""
    return m.group(1), m.group(2)

def _import_module_from(path: Path):
    spec = importlib.util.spec_from_file_location("p05c_mod", str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def find_05c_module():
    # 1) try preferred names
    for name in PREFERRED_05C_NAMES:
        p = V2_DIR / name
        if p.exists():
            return _import_module_from(p), p
    # 2) scan any 05c*.py containing 'def plot_repo'
    for p in sorted(V2_DIR.glob("05c*.py")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "def plot_repo" in txt:
            return _import_module_from(p), p
    raise FileNotFoundError(
        "Could not locate a 05c plotter in v2/. Expected one of: "
        + ", ".join(PREFERRED_05C_NAMES)
        + " or any 05c*.py that defines def plot_repo(...)."
    )

def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("\\", "__")

# ---------- main --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot images from survey selection directly into 09_plot_from_selection.")
    ap.add_argument("--selection-csv", type=str, default=None, help="Path to survey_selection_v2_*.csv")
    ap.add_argument("--smooth", type=int, default=3, help="Smoothing window passed to 05c plotter (default: 3)")
    args = ap.parse_args()

    # Selection CSV
    sel_csv = Path(args.selection_csv) if args.selection_csv else pick_latest_selection()
    if not sel_csv.exists():
        raise FileNotFoundError(f"Selection CSV not found: {sel_csv}")
    stamp, k = parse_stamp_k(sel_csv)
    print(f"[INFO] Selection: {sel_csv.name} (STAMP={stamp}, K={k})")

    df_sel = pd.read_csv(sel_csv)
    if "repo" not in df_sel.columns:
        raise ValueError("Selection CSV must contain a 'repo' column.")
    repos = [r for r in df_sel["repo"].astype(str).tolist() if r.strip()]
    if not repos:
        raise ValueError("No repos found in selection CSV.")

    # Locate & load 05c module
    p05c, path_05c = find_05c_module()
    print(f"[INFO] Using 05c module: {path_05c.name}")

    # Check required callables
    for fn in ("plot_repo", "infer_latest_assignments", "infer_stamp_from_name"):
        if not hasattr(p05c, fn):
            raise AttributeError(f"{path_05c.name} must define {fn}(...).")

    # Load assignments like 05c does
    f_assign = p05c.infer_latest_assignments()
    print(f"[INFO] Assignments: {f_assign.name}")
    stamp_assign = p05c.infer_stamp_from_name(f_assign)

    df = pd.read_csv(f_assign, parse_dates=["week_dt"])

    # Stage label normalization if needed
    uniq = set(df["stage_name"].astype(str).unique())
    if "Baseline" not in uniq and "Baseline (Solo)" in uniq:
        print("[WARN] Found 'Baseline (Solo)' — remapping to 'Baseline' for plotting consistency.")
        df["stage_name"] = df["stage_name"].replace({"Baseline (Solo)": "Baseline"})

    # Keep only repos in assignments
    repos_all = set(df["repo"].astype(str).unique())
    missing = [r for r in repos if r not in repos_all]
    if missing:
        print("[WARN] Repos in selection but not in assignments:")
        for r in missing:
            print("  -", r)
    target = [r for r in repos if r in repos_all]
    if not target:
        raise ValueError("No overlapping repos between selection and assignments.")

    # Output dir under 09_plot_from_selection
    out_dir = OUT_ROOT  # no timestamped subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving images to: {out_dir}")

    # Plot
    for i, r in enumerate(target, 1):
        print(f"[{i}/{len(target)}] {r} ...")
        p05c.plot_repo(df[df["repo"] == r], r, stamp_assign, out_dir, smooth=args.smooth)

    # Build import list for Qualtrics
    rows = [{
        "repo": r,
        "display_name": r,
        "image_path": str((out_dir / f"repo_{safe_name(r)}.png").relative_to(ROOT)),
        "quota": 5
    } for r in target]

    out_csv = OUT_ROOT / f"final_survey_list_{stamp}_K{k}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote import list: {out_csv}")

if __name__ == "__main__":
    main()
