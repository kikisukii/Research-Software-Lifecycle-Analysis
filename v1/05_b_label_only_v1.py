#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_b_label_only_v1.py

目标：仅用 05a 的 production KMeans + 模板映射，为每周样本生成“事实标签”表，不做监督评估。
- 分簇对象：dead_flag==0 且 age_weeks>=24（live & mature）
- stage_core：仅 {Rising, Peak, Cooling}；Dead/Initial 置空（NaN）
- stage_display：展示用（Initial/Dead/R/P/C）
- 时间口径：UTC；提供 week_unix 与 week_date_utc（YYYY-MM-DD）

输入：
  --features_csv   data/03_features_weekly_v1_<STAMP>_filtered.csv
  --models_json    data/05_a_kmeans/<STAMP>/05_global_models_v1_<STAMP>.json
  --template_json  data/05_a_kmeans/<STAMP>/05_labeling_template_v1_<STAMP>_K{K}.json（你已填好 stage）
  --outdir         （可选）默认 data/05_b_apply/<STAMP>/
  --emit_current   （可选开关）若给出，则额外导出每仓“最新一周”的 stage_display

输出：
  data/05_b_apply/<STAMP>/05_labels_v1_<STAMP>_K{K}.csv
  （可选）data/05_b_apply/<STAMP>/05_current_stage_v1_<STAMP>_K{K}.csv

用法（Run > Edit Configurations…）参数示例：
  --features_csv data/03_features_weekly_v1_20251029_084949_filtered.csv
  --models_json  data/05_a_kmeans/20251029_084949/05_global_models_v1_20251029_084949.json
  --template_json data/05_a_kmeans/20251029_084949/05_labeling_template_v1_20251029_084949_K3.json
  --outdir data/05_b_apply/20251029_084949/
  --emit_current
"""
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd

STAMP_RE = re.compile(r"(\d{8}_\d{6})")

def infer_stamp(p: str) -> str:
    m = STAMP_RE.search(Path(p).name)
    if not m:
        raise ValueError(f"无法从文件名推断 <STAMP>：{p}")
    return m.group(1)

def load_features(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    need = {"repo","week_unix","c8_total","M8_24","dead_flag","age_weeks"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"features_csv 缺少必要列：{miss}")
    # UTC 绝对日期（不做时区转换）
    df["week_date_utc"] = pd.to_datetime(df["week_unix"], unit="s", utc=True).dt.date.astype(str)
    return df

def load_model_json(fp: str, expect_K: int):
    obj = json.load(open(fp, "r", encoding="utf-8"))
    key = f"K{expect_K}"
    if key not in obj:
        if len(obj) == 1:
            key = list(obj.keys())[0]
        else:
            raise ValueError(f"{fp} 中未找到键 {key}")
    m = obj[key]
    feats = m.get("features", [])
    if feats not in (["log_c8","M8_24"], ["c8_total","M8_24"]):
        raise ValueError(f"不支持的模型特征：{feats}")
    return {
        "features": feats,
        "centers_std": np.array(m["centers_std"], dtype=np.float32),
        "scaler_center": np.array(m["scaler_center"], dtype=np.float32),
        "scaler_scale":  np.array(m["scaler_scale"], dtype=np.float32),
        "K": int(m["K"]),
    }

def load_stage_map(template_fp: str):
    t = json.load(open(template_fp, "r", encoding="utf-8"))
    K = int(t.get("K", -1))
    mp = {}
    for c in t["clusters"]:
        cid = int(c["cluster_id"])
        stage = (c.get("stage") or "").strip()
        if stage not in ("Rising","Peak","Cooling"):
            raise ValueError(f"模板 cluster_id={cid} 的 stage 不是 R/P/C，请修正（当前='{stage}'）")
        mp[cid] = stage
    return K, mp

def assign_clusters(df: pd.DataFrame, model: dict):
    """
    只给 live & age>=24 的样本分簇；其它（Dead 或 Initial）cluster_id=-1
    """
    cid = np.full(len(df), -1, dtype=int)
    mask_live_mature = (df["dead_flag"] == 0) & (df["age_weeks"] >= 24)

    if model["features"] == ["log_c8","M8_24"]:
        X0 = np.log1p(df.loc[mask_live_mature, "c8_total"].values.astype(np.float32))
        X1 = df.loc[mask_live_mature, "M8_24"].values.astype(np.float32)
    else:  # ["c8_total","M8_24"]
        X0 = df.loc[mask_live_mature, "c8_total"].values.astype(np.float32)
        X1 = df.loc[mask_live_mature, "M8_24"].values.astype(np.float32)

    X = np.stack([X0, X1], axis=1)
    # 标准化到与模型相同的“未加权的标准化”空间
    Xz = (X - model["scaler_center"]) / np.maximum(model["scaler_scale"], 1e-12)
    d2 = ((Xz[:, None, :] - model["centers_std"][None, :, :]) ** 2).sum(axis=2)
    cid_live = np.argmin(d2, axis=1).astype(int)
    cid[mask_live_mature.values] = cid_live
    return cid

def to_current_stage(df_labels: pd.DataFrame) -> pd.DataFrame:
    idx = (df_labels.sort_values(["repo","week_unix"])
                    .groupby("repo", as_index=False).tail(1).index)
    return (df_labels.loc[idx, ["repo","week_unix","week_date_utc","stage_display"]]
                     .rename(columns={"stage_display":"stage"})
                     .reset_index(drop=True))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--models_json", required=True)
    ap.add_argument("--template_json", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--emit_current", action="store_true")
    args = ap.parse_args()

    stamp = infer_stamp(args.features_csv)
    outdir = Path(args.outdir or f"data/05_b_apply/{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取模板 & 模型
    K_tmpl, stage_map = load_stage_map(args.template_json)
    model = load_model_json(args.models_json, expect_K=K_tmpl)
    if model["K"] != K_tmpl:
        raise ValueError(f"模型K={model['K']} 与 模板K={K_tmpl} 不一致")

    # 读取特征并分簇
    df = load_features(args.features_csv)
    cid = assign_clusters(df, model)

    # 生成标签两列
    df_labels = df.copy()
    df_labels["cluster_id"] = cid

    # stage_core：仅 R/P/C；Dead 与 Initial 均为空
    is_live_mature = (df_labels["dead_flag"] == 0) & (df_labels["age_weeks"] >= 24)
    core = np.full(len(df_labels), None, dtype=object)
    core[is_live_mature.values] = [stage_map.get(int(k), None)
                                   for k in df_labels.loc[is_live_mature, "cluster_id"].values]
    df_labels["stage_core"] = core  # R/P/C/NaN

    # stage_display：展示用（Initial / Dead / R/P/C）
    disp = np.where(df_labels["age_weeks"] < 24, "Initial",
             np.where(df_labels["dead_flag"] == 1, "Dead", df_labels["stage_core"]))
    df_labels["stage_display"] = disp

    # 落盘：labels（保留两列；UTC 日期）
    fp_labels = outdir / f"05_labels_v1_{stamp}_K{model['K']}.csv"
    cols_out = ["repo","week_unix","week_date_utc","c8_total","M8_24","age_weeks",
                "dead_flag","cluster_id","stage_core","stage_display"]
    df_labels[cols_out].to_csv(fp_labels, index=False, encoding="utf-8")

    # （可选）导出“当前阶段清单”（用 display）
    if args.emit_current:
        fp_curr = outdir / f"05_current_stage_v1_{stamp}_K{model['K']}.csv"
        to_current_stage(df_labels).to_csv(fp_curr, index=False, encoding="utf-8")
        print(f"[OK] current -> {fp_curr}")

    print(f"[OK] labels  -> {fp_labels}")
    print(f"[INFO] K={model['K']}, samples={len(df_labels)}, repos={df_labels['repo'].nunique()}")

if __name__ == "__main__":
    main()
