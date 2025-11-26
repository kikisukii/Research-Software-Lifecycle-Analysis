#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_b_label_only_v1.py

Objective:
  Apply the frozen KMeans model and manual labeling template to generate
  FINAL "Fact Labels" for the ENTIRE dataset.

Updates (vs old version):
  - Now produces THREE outputs to match v2 structure:
    1. Labels (Full Dataset)
    2. Profiles with Stage Names (Stats per stage)
    3. Cluster Map (ID -> Stage mapping CSV)

Inputs (Auto-detected in v1_data):
  1. Features: Latest '03_features_weekly_v1_*.csv' (EXCLUDING _filtered)
  2. Model:    Latest '05_global_models_v1_*.json'
  3. Template: Latest '05_labeling_template_v1_*.json'

Outputs (in v1_data/05_b_apply/):
  - 05_labels_v1_<STAMP>_K{K}.csv
  - 05_profile_with_stage_v1_<STAMP>_K{K}.csv   <-- NEW
  - 05_cluster_map_v1_<STAMP>_K{K}.csv          <-- NEW
  - 05_current_stage_v1_<STAMP>_K{K}.csv        (Snapshot)

Usage:
  python 05_b_label_only_v1.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# [CONFIG] Path Logic
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V1_DATA_ROOT = PROJECT_ROOT / "v1_data"
FEATURE_DIR = V1_DATA_ROOT / "03_features"
MODEL_DIR = V1_DATA_ROOT / "05_a_kmeans"
OUTPUT_DIR_ROOT = V1_DATA_ROOT / "05_b_apply"

STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def infer_stamp(p: Path) -> str:
    m = STAMP_RE.search(p.name)
    if not m:
        return "00000000_000000"
    return m.group(1)


def find_latest_full_features() -> Path:
    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Feature directory not found: {FEATURE_DIR}")
    candidates = list(FEATURE_DIR.glob("03_features_weekly_v1_*.csv"))
    full_candidates = [p for p in candidates if "_filtered" not in p.name]
    if not full_candidates:
        raise FileNotFoundError("No full feature CSVs found in 03_features.")
    latest = sorted(full_candidates, key=lambda p: infer_stamp(p))[-1]
    print(f"[AUTO] Found latest features: {latest.name}")
    return latest


def find_latest_model_and_template() -> tuple[Path, Path]:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    models = list(MODEL_DIR.glob("05_global_models_v1_*.json"))
    if not models:
        raise FileNotFoundError("No model JSONs found.")
    latest_model = sorted(models, key=lambda p: infer_stamp(p))[-1]

    stamp = infer_stamp(latest_model)
    templates = list(MODEL_DIR.glob(f"05_labeling_template_v1_{stamp}_*.json"))
    if not templates:
        raise FileNotFoundError(f"No labeling template found for stamp {stamp}")

    print(f"[AUTO] Found latest model:    {latest_model.name}")
    print(f"[AUTO] Found matching tmpl:   {templates[0].name}")
    return latest_model, templates[0]


def load_features_full(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    required = {"repo", "week_unix", "c8_total", "M8_24", "dead_flag", "age_weeks"}
    if not required.issubset(df.columns):
        raise ValueError(f"Features missing columns: {required - set(df.columns)}")
    df["week_date_utc"] = pd.to_datetime(df["week_unix"], unit="s", utc=True).dt.date.astype(str)
    return df


def load_model_json(fp: Path, expect_K: int):
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    key = f"K{expect_K}"
    if key not in obj:
        if len(obj) == 1:
            key = list(obj.keys())[0]
        else:
            raise ValueError(f"Key {key} not found in model.")
    m = obj[key]
    return {
        "centers_std": np.array(m["centers_std"], dtype=np.float32),
        "scaler_center": np.array(m["scaler_center"], dtype=np.float32),
        "scaler_scale": np.array(m["scaler_scale"], dtype=np.float32),
        "w_c": float(m["w_c"]),
        "w_m": float(m["w_m"]),
        "K": int(m["K"]),
    }


def load_stage_map(template_fp: Path):
    with open(template_fp, "r", encoding="utf-8") as f:
        t = json.load(f)
    K = int(t.get("K", -1))
    mp = {}
    for c in t["clusters"]:
        mp[int(c["cluster_id"])] = (c.get("stage") or "").strip()
    return K, mp


def predict_clusters(df: pd.DataFrame, model: dict):
    mask = (df["dead_flag"] == 0) & (df["age_weeks"] >= 24)
    cid = np.full(len(df), -1, dtype=int)
    if not mask.any(): return cid

    X0 = np.log1p(df.loc[mask, "c8_total"].values.astype(np.float32))
    X1 = df.loc[mask, "M8_24"].values.astype(np.float32)
    X = np.stack([X0, X1], axis=1)

    Xz = (X - model["scaler_center"]) / np.maximum(model["scaler_scale"], 1e-12)
    Xw = Xz.copy()
    Xw[:, 0] *= model["w_c"]
    Xw[:, 1] *= model["w_m"]
    Cw = model["centers_std"].copy()
    Cw[:, 0] *= model["w_c"]
    Cw[:, 1] *= model["w_m"]

    d2 = ((Xw[:, None, :] - Cw[None, :, :]) ** 2).sum(axis=2)
    cid[mask.values] = np.argmin(d2, axis=1)
    return cid


def to_current_stage(df: pd.DataFrame):
    idx = df.sort_values(["repo", "week_unix"]).groupby("repo").tail(1).index
    return df.loc[idx, ["repo", "week_unix", "week_date_utc", "stage"]].reset_index(drop=True)


def generate_profiles(df: pd.DataFrame, stage_map: dict):
    """Generates statistics per stage (including cluster ID info)."""
    # Filter to analyzed rows (valid cluster ID)
    valid = df[df["cluster_id"] != -1].copy()

    # Calculate stats
    # We group by ['cluster_id', 'stage_core'] to keep the mapping clear
    prof = (valid.groupby(["cluster_id", "stage_core"])
            .agg(
        n_weeks=("repo", "size"),
        n_repos=("repo", "nunique"),
        c8_med=("c8_total", "median"),
        c8_q25=("c8_total", lambda x: np.quantile(x, 0.25)),
        c8_q75=("c8_total", lambda x: np.quantile(x, 0.75)),
        M_med=("M8_24", "median"),
        M_q25=("M8_24", lambda x: np.quantile(x, 0.25)),
        M_q75=("M8_24", lambda x: np.quantile(x, 0.75))
    )
            .reset_index()
            .rename(columns={"stage_core": "stage"})
            .sort_values("cluster_id")
            )
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv")
    ap.add_argument("--models_json")
    ap.add_argument("--template_json")
    ap.add_argument("--outdir")
    ap.add_argument("--emit_current", action="store_true", default=True)
    args = ap.parse_args()

    # 1. Discovery
    if args.features_csv:
        feat_path = Path(args.features_csv)
    else:
        feat_path = find_latest_full_features()

    if args.models_json and args.template_json:
        model_path = Path(args.models_json)
        tmpl_path = Path(args.template_json)
    else:
        model_path, tmpl_path = find_latest_model_and_template()

    stamp = infer_stamp(feat_path)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = OUTPUT_DIR_ROOT
    outdir.mkdir(parents=True, exist_ok=True)

    # 2. Load & Predict
    K_tmpl, stage_map = load_stage_map(tmpl_path)
    model = load_model_json(model_path, expect_K=K_tmpl)

    print(f"[INFO] Features: {feat_path.name}")
    print(f"[INFO] Model K={model['K']} ({model_path.name})")

    df = load_features_full(feat_path)
    df["cluster_id"] = predict_clusters(df, model)

    # 3. Labeling
    df["stage_core"] = df["cluster_id"].map(stage_map)

    conds = [(df["dead_flag"] == 1), (df["age_weeks"] < 24)]
    choices = ["Dead", "Initial"]
    df["stage"] = np.select(conds, choices, default=df["stage_core"])
    df["stage"] = df["stage"].fillna("Unknown")

    # 4. Generate Extra Outputs

    # A. Profiles with Stage
    df_prof = generate_profiles(df, stage_map)

    # B. Cluster Map
    df_map = pd.DataFrame(list(stage_map.items()), columns=["cluster_id", "stage"])

    # 5. Save All
    base_name = f"v1_{stamp}_K{model['K']}"

    fp_labels = outdir / f"05_labels_{base_name}.csv"
    fp_prof = outdir / f"05_profile_with_stage_{base_name}.csv"
    fp_map = outdir / f"05_cluster_map_{base_name}.csv"

    cols = ["repo", "week_unix", "week_date_utc", "c8_total", "M8_24", "age_weeks", "dead_flag", "cluster_id", "stage"]
    df[cols].to_csv(fp_labels, index=False)

    df_prof.to_csv(fp_prof, index=False)
    df_map.to_csv(fp_map, index=False)

    print(f"[OK] Labels:   {fp_labels}")
    print(f"[OK] Profiles: {fp_prof}")
    print(f"[OK] Map:      {fp_map}")

    if args.emit_current:
        fp_curr = outdir / f"05_current_stage_{base_name}.csv"
        to_current_stage(df).to_csv(fp_curr, index=False)
        print(f"[OK] Snapshot: {fp_curr}")


if __name__ == "__main__":
    main()