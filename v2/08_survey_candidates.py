# -*- coding: utf-8 -*-
"""
08_survey_candidates.py
- Detect the latest v2 05b STAMP (supports YYYYMMDD or YYYYMMDD_HHMMSS) and K
- Build per-repo candidate metrics from 05b assignments (READ-ONLY)
- Write a sorted candidate table (+ convenience bin labels)
- ALWAYS select N repositories via STRATIFIED RANDOM SAMPLING within bins,
  under explicit coverage constraints:
    * HARD per-stage caps (with a small overflow distributed randomly)
    * dormant archetypes
    * release/activity high/low balance
    * long-span >= 4 target (best effort)
- Record strategy + seed in meta JSON for reproducibility.

Run (from project root or anywhere):
    python v2/08_survey_candidates.py
    # or pick another seed:
    python v2/08_survey_candidates.py --rng-seed 42
"""

import json
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Project roots (robust to current working dir) --------------------
THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent.parent          # one level up from /v2/
V2_05B_DIR = PROJ_ROOT / "v2_data" / "05_b_apply"
OUT_DIR    = PROJ_ROOT / "v2_data" / "08_survey_candidates"

ASSIGN_PREFIX = "05b_assignments_with_stage_AND_DEAD_v2_"
SUMMARY_PREFIX = "05b_current_stage_summary_v2_"
ASSIGN_GLOB = f"{ASSIGN_PREFIX}*_K*.csv"
SUMMARY_GLOB = f"{SUMMARY_PREFIX}*_K*.csv"

FOUR_FEATS = [
    "commits_8w_sum",
    "contributors_8w_unique",
    "issues_closed_8w_count",
    "releases_8w_count",
]

# Support both: ..._v2_20251114_K6.csv  OR  ..._v2_20251114_004622_K6.csv
STAMPK_RE = re.compile(r"_v2_(\d{8})(?:_(\d{6}))?_K(\d+)", re.IGNORECASE)


def parse_stamp_k(name: str):
    """Returns (stamp_str, sort_key_int, k)."""
    m = STAMPK_RE.search(name)
    if not m:
        return None, None, None
    date = m.group(1)
    time = m.group(2) or ""          # optional HHMMSS
    k = int(m.group(3))
    stamp_str = f"{date}_{time}" if time else date
    sort_key = int(date + (time if time else "000000"))
    return stamp_str, sort_key, k


def pick_latest_files():
    """Pick the latest assignment; then the best-matching summary."""
    assigns = list(V2_05B_DIR.glob(ASSIGN_GLOB))
    if not assigns:
        raise FileNotFoundError(f"No assignment files like {ASSIGN_GLOB} in {V2_05B_DIR}")

    assigns = sorted(assigns, key=lambda p: (parse_stamp_k(p.name)[1] or 0, p.stat().st_mtime))
    latest_assign = assigns[-1]
    a_stamp, a_sort, a_k = parse_stamp_k(latest_assign.name)

    summaries = list(V2_05B_DIR.glob(SUMMARY_GLOB))
    if not summaries:
        raise FileNotFoundError(f"No summary files like {SUMMARY_GLOB} in {V2_05B_DIR}")

    exact = [p for p in summaries if parse_stamp_k(p.name)[:1] == (a_stamp,) and parse_stamp_k(p.name)[2] == a_k]
    if exact:
        summary_pick = exact[0]
    else:
        same_k = sorted(
            [p for p in summaries if parse_stamp_k(p.name)[2] == a_k],
            key=lambda p: abs((parse_stamp_k(p.name)[1] or 0) - a_sort),
        )
        summary_pick = same_k[0] if same_k else sorted(
            summaries, key=lambda p: abs((parse_stamp_k(p.name)[1] or 0) - a_sort)
        )[0]

    return latest_assign, summary_pick, a_stamp, a_k


# --------------------------- Selection helpers --------------------------------

def build_candidates(assign_csv: Path, summary_csv: Path) -> pd.DataFrame:
    """Aggregate per-repo metrics from 05b assignments; merge current stage."""
    df = pd.read_csv(assign_csv)
    summary = pd.read_csv(summary_csv)[["repo", "stage_name"]].rename(
        columns={"stage_name": "current_stage"}
    )

    required = {"repo", "week_unix", "stage_name", *FOUR_FEATS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in assignments CSV: {missing}")

    # Row-level helpers (no recomputation of features)
    df["valid_row"]      = df[FOUR_FEATS].notna().all(axis=1)
    df["all_zero_8w"]    = (df[FOUR_FEATS].fillna(0).eq(0)).all(axis=1)
    df["has_release_8w"] = (df["releases_8w_count"].fillna(0) > 0).astype(int)

    # Current stage: prefer summary; fallback to latest row per repo
    latest_idx = df.groupby("repo")["week_unix"].idxmax()
    latest_stage = df.loc[latest_idx, ["repo", "stage_name"]].rename(
        columns={"stage_name": "current_stage_fallback"}
    )
    current_stage = latest_stage.merge(summary, on="repo", how="left")
    current_stage["current_stage"] = current_stage["current_stage"].fillna(
        current_stage["current_stage_fallback"]
    )
    current_stage = current_stage[["repo", "current_stage"]]

    # Per-repo aggregation
    g = df.groupby("repo", sort=False)
    agg = pd.DataFrame({
        "total_weeks": g.size(),
        "valid_8w": g["valid_row"].sum(),
        "na_ratio": 1 - g["valid_row"].mean(),
        "activity_med": g["commits_8w_sum"].median(),
        "activity_p90": g["commits_8w_sum"].quantile(0.9),
        "release_freq": g["has_release_8w"].mean(),
        "zero_8w_ratio": g["all_zero_8w"].mean(),
        "span_years": (g["week_unix"].max() - g["week_unix"].min()) / (365 * 24 * 3600),
    }).reset_index()

    cand = agg.merge(current_stage, on="repo", how="left")
    return cand


def readable_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Readability-first ordering (for human-friendly viewing only)."""
    return df.sort_values(
        by=["na_ratio", "valid_8w", "zero_8w_ratio", "activity_p90", "activity_med", "release_freq", "repo"],
        ascending=[True,      False,   True,            False,          False,          False,            True]
    ).reset_index(drop=True)


# ---- RANDOM utilities ---------------------------------------------------------
def _rng(seed: int | None):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()

def _choice_df(df: pd.DataFrame, k: int, rng: np.random.Generator) -> list[int]:
    if df.empty or k <= 0:
        return []
    k = min(k, len(df))
    idx = rng.choice(df.index.values, size=k, replace=False)
    return list(idx)


def auto_select(
    cand: pd.DataFrame,
    n_select: int = 15,
    per_stage_cap: int = 2,
    min_valid_weeks: int = 26,
    max_na_ratio: float = 0.10,
    release_high_thr: float = 0.15,
    span_long_years: float = 3.0,
    dormant_target_lo: int = 2,
    dormant_target_hi: int = 3,
    min_bin_cover: int = 6,
    rng_seed: int | None = 20251127,  # reproducible random (default)
):
    """
    ALWAYS-RANDOM selection under explicit constraints with HARD per-stage caps.
    If n_select > per_stage_cap * #stages, the overflow slots are assigned
    by giving +1 capacity to randomly drawn stages (distinct).
    """

    rng = _rng(rng_seed)

    # 1) Hard filter ----------------------------------------------------------
    pool = cand.query("valid_8w >= @min_valid_weeks and na_ratio <= @max_na_ratio").copy()
    if pool.empty:
        raise ValueError("No candidates after hard filter.")

    # 2) Bins -----------------------------------------------------------------
    med_thr = pool["activity_med"].median(skipna=True)
    pool["activity_bin"] = np.where(pool["activity_med"] >= med_thr, "high", "low")
    pool["release_bin"]  = np.where(pool["release_freq"] >= release_high_thr, "high", "low")
    pool["span_bin"]     = np.where(pool["span_years"] >= span_long_years, "long", "short-mid")

    # 3) Per-stage HARD caps (with random overflow distribution) --------------
    stages = sorted(pool["current_stage"].dropna().unique().tolist())
    n_stages = len(stages)
    stage_cap = {s: per_stage_cap for s in stages}

    overflow = max(0, n_select - per_stage_cap * n_stages)
    if overflow > 0:
        give = rng.choice(stages, size=min(overflow, n_stages), replace=False).tolist()
        for s in give:
            stage_cap[s] += 1

    stage_count = {s: 0 for s in stages}

    # 4) Helpers enforcing stage caps ----------------------------------------
    chosen: list[str] = []
    reasons: dict[str, str] = {}

    def _add_by_row(row: pd.Series, reason: str) -> bool:
        """Add a single repo if its stage still has capacity."""
        rid = row["repo"]; stg = row["current_stage"]
        if rid in reasons:
            return False
        if stage_count.get(stg, 0) >= stage_cap.get(stg, 0):
            return False
        reasons[rid] = reason
        chosen.append(rid)
        stage_count[stg] = stage_count.get(stg, 0) + 1
        return True

    def _pick_and_add(df_pick: pd.DataFrame, reason: str, k: int = 1):
        """Pick up to k repos randomly that still fit stage caps."""
        if df_pick.empty or k <= 0:
            return
        df_pick = df_pick[~df_pick["repo"].isin(chosen)]
        if df_pick.empty:
            return
        df_pick = df_pick.sample(frac=1.0, random_state=int(rng.integers(1, 1_000_000)))
        added = 0
        for _, r in df_pick.iterrows():
            if added >= k or len(chosen) >= n_select:
                break
            if _add_by_row(r, reason):
                added += 1

    def counts(ids):
        sub = pool[pool["repo"].isin(ids)]
        return {
            "n": len(sub),
            "release_high": int((sub["release_bin"] == "high").sum()),
            "release_low":  int((sub["release_bin"] == "low").sum()),
            "activity_high": int((sub["activity_bin"] == "high").sum()),
            "activity_low":  int((sub["activity_bin"] == "low").sum()),
            "span_long": int((sub["span_bin"] == "long").sum()),
            "span_short": int((sub["span_bin"] == "short-mid").sum()),
        }

    # 5) Stage coverage (respect hard caps) -----------------------------------
    pool_shuf = pool.sample(frac=1.0, random_state=int(rng.integers(1, 1_000_000)))
    for stg, grp in pool_shuf.groupby("current_stage"):
        if len(chosen) >= n_select:
            break
        need = stage_cap.get(stg, 0) - stage_count.get(stg, 0)
        if need <= 0:
            continue
        # prefer 1 long + 1 short-mid
        _pick_and_add(grp[grp["span_bin"] == "long"],      f"stage_pick:{stg}", k=min(1, need))
        need = stage_cap.get(stg, 0) - stage_count.get(stg, 0)
        _pick_and_add(grp[grp["span_bin"] == "short-mid"], f"stage_pick:{stg}", k=min(1, need))
        # fill within stage up to its cap
        need = stage_cap.get(stg, 0) - stage_count.get(stg, 0)
        if need > 0:
            _pick_and_add(grp, f"stage_pick:{stg}", k=need)

    # 6) Dormant archetypes (respect caps) ------------------------------------
    remain_slots = max(0, n_select - len(chosen))
    dorm_target = min(dormant_target_hi, max(dormant_target_lo, remain_slots))
    dorm = pool[(pool["zero_8w_ratio"] >= 0.50) & (pool["release_bin"] == "low")]
    if dorm_target > 0:
        _pick_and_add(dorm, "dormant_archetype", k=dorm_target)

    # 7) Bin balancing (respect caps; best effort) ----------------------------
    def add_for_bin(count_key_base: str, col_name: str, target_min: int, reason_tag: str):
        c = counts(chosen)
        need_high = max(0, target_min - c[f"{count_key_base}_high"])
        need_low  = max(0, target_min - c[f"{count_key_base}_low"])
        for label, need in [("high", need_high), ("low", need_low)]:
            if need <= 0 or len(chosen) >= n_select:
                continue
            df_need = pool[(pool[col_name] == label) & (~pool["repo"].isin(chosen))]
            _pick_and_add(df_need, f"{reason_tag}:{label}", k=min(need, n_select - len(chosen)))

    add_for_bin("release",  "release_bin",  min_bin_cover, "release_balance")
    add_for_bin("activity", "activity_bin", min_bin_cover, "activity_balance")

    # 8) Span target: add longs if below 4 (best effort) ----------------------
    c_now = counts(chosen)
    long_need = max(0, 4 - c_now["span_long"])
    if long_need > 0 and len(chosen) < n_select:
        df_long = pool[(pool["span_bin"] == "long") & (~pool["repo"].isin(chosen))]
        _pick_and_add(df_long, "span_balance:long", k=min(long_need, n_select - len(chosen)))

    # 9) Fill remaining (respect caps) ----------------------------------------
    if len(chosen) < n_select:
        rest = pool[~pool["repo"].isin(chosen)]
        _pick_and_add(rest, "fill", k=n_select - len(chosen))

    # 10) Build output ---------------------------------------------------------
    sel = pool[pool["repo"].isin(chosen)].copy()
    sel["selection_reason"] = sel["repo"].map(reasons).fillna("fill")
    # For reporting: order by readability
    sel = readable_sort(sel)

    meta = {
        "params": {
            "n_select": n_select,
            "per_stage_cap": per_stage_cap,
            "min_valid_weeks": min_valid_weeks,
            "max_na_ratio": max_na_ratio,
            "release_high_thr": release_high_thr,
            "span_long_years": span_long_years,
            "dormant_target_lo": dormant_target_lo,
            "dormant_target_hi": dormant_target_hi,
            "min_bin_cover": min_bin_cover,
            "strategy": "random_bins_with_hard_stage_caps",
            "rng_seed": rng_seed,
            "stage_cap": stage_cap,
        },
        "counts": counts(chosen),
        "stage_counts": sel["current_stage"].value_counts(dropna=False).to_dict(),
    }
    return sel, pool, meta


# ----------------------------------- Main ------------------------------------

def main(
    n_select: int,
    per_stage_cap: int,
    min_valid_weeks: int,
    max_na_ratio: float,
    release_high_thr: float,
    span_long_years: float,
    dormant_target_lo: int,
    dormant_target_hi: int,
    min_bin_cover: int,
    rng_seed: int | None,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assign_csv, summary_csv, stamp, k = pick_latest_files()
    print(f"[INFO] Using STAMP={stamp}, K={k}")
    print(f"[INFO] Assignments: {assign_csv}")
    print(f"[INFO] Summary    : {summary_csv}")

    cand = build_candidates(assign_csv, summary_csv)

    # Candidate table (sorted for readability) + bins for convenience
    cand_bins = cand.copy()
    med_thr = cand_bins["activity_med"].median(skipna=True)
    cand_bins["activity_bin"] = np.where(cand_bins["activity_med"] >= med_thr, "high", "low")
    cand_bins["release_bin"]  = np.where(cand_bins["release_freq"] >= release_high_thr, "high", "low")
    cand_bins["span_bin"]     = np.where(cand_bins["span_years"] >= span_long_years, "long", "short-mid")
    cand_sorted = readable_sort(cand_bins)

    cand_out = OUT_DIR / f"survey_candidates_v2_{stamp}_K{k}.csv"
    cand_sorted.to_csv(cand_out, index=False)
    print(f"[OK] Wrote candidates: {cand_out} (rows={len(cand_sorted)})")

    # Random selection under constraints (with hard stage caps)
    sel, pool, meta = auto_select(
        cand_sorted,
        n_select=n_select,
        per_stage_cap=per_stage_cap,
        min_valid_weeks=min_valid_weeks,
        max_na_ratio=max_na_ratio,
        release_high_thr=release_high_thr,
        span_long_years=span_long_years,
        dormant_target_lo=dormant_target_lo,
        dormant_target_hi=dormant_target_hi,
        min_bin_cover=min_bin_cover,
        rng_seed=rng_seed,
    )

    sel_out = OUT_DIR / f"survey_selection_v2_{stamp}_K{k}.csv"
    meta_out = OUT_DIR / f"selection_meta_v2_{stamp}_K{k}.json"

    sel.to_csv(sel_out, index=False)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Wrote selection : {sel_out} (rows={len(sel)})")
    print(f"[OK] Wrote meta      : {meta_out}")
    print("\n[SELECTION COUNTS]")
    print(pd.Series(meta["counts"]))
    print("\n[STAGE COUNTS]")
    print(sel["current_stage"].value_counts(dropna=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/Select survey repositories for v2 (latest STAMP) with stratified random sampling and hard per-stage caps.")
    parser.add_argument("--n_select", type=int, default=15, help="Number of repositories to select (default: 15).")
    parser.add_argument("--per_stage_cap", type=int, default=2, help="Max number per current_stage (default: 2).")
    parser.add_argument("--min_valid_weeks", type=int, default=26, help="Minimum weeks with all four 8w features (default: 26).")
    parser.add_argument("--max_na_ratio", type=float, default=0.10, help="Maximum NA ratio per repo (default: 0.10).")
    parser.add_argument("--release_high_thr", type=float, default=0.15, help="Threshold for 'high release frequency' (default: 0.15).")
    parser.add_argument("--span_long_years", type=float, default=3.0, help="Threshold (years) for 'long' span (default: 3.0).")
    parser.add_argument("--dormant_target_lo", type=int, default=2, help="Lower target for dormant archetypes (default: 2).")
    parser.add_argument("--dormant_target_hi", type=int, default=3, help="Upper target for dormant archetypes (default: 3).")
    parser.add_argument("--min_bin_cover", type=int, default=6, help="Min coverage for high/low of release/activity bins (best effort; default: 6).")
    parser.add_argument("--rng-seed", type=int, default=20251127, help="Random seed for reproducible random selection (default: 20251127).")
    args = parser.parse_args()
    main(**vars(args))
