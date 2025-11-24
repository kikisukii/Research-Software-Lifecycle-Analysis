#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_a_fit_kmeans_v1.py
目的：用 (K*, α*) 在合规全量样本上训练一次 KMeans（无监督），固化模型与簇画像。
- 合规样本：dead_flag==0 且 age_weeks>=24（v1 规则）
- 特征空间：x=log1p(c8_total), y=M8_24
- 预处理：RobustScaler（鲁棒标准化） -> α 加权（w_c=2-α, w_m=α） -> KMeans
输出目录：data/05_a_kmeans/<STAMP>/
  - 05_global_models_v1_<STAMP>.json
  - 05_labeling_template_v1_<STAMP>_K{K}.json
用法示例：
  # 手动给 K、α
  python 05_a_fit_kmeans_v1.py \
    --features_csv data/03_features_weekly_v1_20251029_084949_filtered.csv \
    --k 3 --alpha 1.75

  # 或从细扫结果自动取 top-1（按 silhouette 再 CH 排序）
  python 05_a_fit_kmeans_v1.py \
    --features_csv data/03_features_weekly_v1_20251029_084949_filtered.csv \
    --grid_csv data/04_b_alpha_k_grid_v1_20251029_084949_fine.csv
"""
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

STAMP_RE = re.compile(r"(\d{8}_\d{6})")

def infer_stamp(p: str) -> str:
    m = STAMP_RE.search(Path(p).name)
    if not m:
        raise ValueError(f"无法从文件名推断 <STAMP>：{p}")
    return m.group(1)

def alpha_weights(alpha: float):
    # w_c 作用于 log_c8（量级），w_m 作用于 M8_24（动量）
    return float(2.0 - alpha), float(alpha)

def pick_from_grid(grid_csv: str):
    """
    从 04 的细扫评分表里挑 top-1。
    【待确认】列名应包含：K, alpha, sil_mean, ch_mean。
    若你的列名不同，请在此函数内调整。
    """
    df = pd.read_csv(grid_csv)
    need = {"K","alpha","sil_mean","ch_mean"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"grid_csv 缺少列：{miss}（请在 pick_from_grid() 调整列名映射）")
    df = df.sort_values(["sil_mean","ch_mean"], ascending=[False, False]).reset_index(drop=True)
    row = df.iloc[0]
    return int(row["K"]), float(row["alpha"])

def load_features(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    # 必要列（v1）：repo, week_unix, c8_total, M8_24, dead_flag, age_weeks
    need = {"repo","week_unix","c8_total","M8_24","dead_flag","age_weeks"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"features_csv 缺少必要列：{miss}")
    # 合规样本：非 Dead 且 满 24 周
    df = df[(df["dead_flag"] == 0) & (df["age_weeks"] >= 24)].copy()
    # 构造特征坐标：log1p(c8_total) 与 M8_24
    df["log_c8"] = np.log1p(df["c8_total"]).astype(np.float32)
    df["M"]      = df["M8_24"].astype(np.float32)
    return df

def fit_kmeans(df: pd.DataFrame, K: int, alpha: float, random_state: int = 42):
    # 1) 鲁棒标准化（RobustScaler）
    X_base = df[["log_c8","M"]].to_numpy(dtype=np.float32)
    scaler = RobustScaler()
    X_std  = scaler.fit_transform(X_base).astype(np.float32)

    # 2) α 加权
    w_c, w_m = alpha_weights(alpha)
    X_w = X_std.copy()
    X_w[:, 0] *= w_c
    X_w[:, 1] *= w_m

    # 3) KMeans（一次定型；n_init 足够大，random_state 固定保证可复现）
    km = KMeans(n_clusters=K, n_init=50, max_iter=1000, random_state=random_state)
    km.fit(X_w)

    # 保存中心到“未加权的标准化空间”（便于后续直接计算距离）
    centers_w  = km.cluster_centers_.astype(np.float32)                # 加权后的标准化空间
    centers_un = centers_w / np.array([w_c, w_m], dtype=np.float32)    # 除回权重

    return {
        "scaler_center": scaler.center_.astype(float).tolist(),
        "scaler_scale":  scaler.scale_.astype(float).tolist(),
        "centers_std":   centers_un.astype(float).tolist(),
        "w_c": float(w_c), "w_m": float(w_m),
        "K": int(K), "alpha": float(alpha),
        "n_init": int(km.n_init), "max_iter": int(km.max_iter),
        "fit_samples": int(len(df)), "fit_repos": int(df["repo"].nunique()),
    }

def assign_by_centers(df: pd.DataFrame, centers_std, scaler_center, scaler_scale):
    X = df[["log_c8","M"]].to_numpy(dtype=np.float32)
    Xz = (X - np.array(scaler_center, dtype=np.float32)) / np.maximum(np.array(scaler_scale, dtype=np.float32), 1e-12)
    d2 = ((Xz[:, None, :] - np.array(centers_std, dtype=np.float32)[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)

def build_cluster_template(df: pd.DataFrame, model: dict):
    # 用保存的（未加权）中心做最近中心分配，生成每簇画像（中位/四分位）
    cid = assign_by_centers(df, model["centers_std"], model["scaler_center"], model["scaler_scale"])
    g = (df.assign(cluster_id=cid)
           .groupby("cluster_id")
           .agg(n=("repo","size"),
                n_repo=("repo","nunique"),
                log_c8_med=("log_c8","median"),
                log_c8_q25=("log_c8", lambda x: float(np.quantile(x, 0.25))),
                log_c8_q75=("log_c8", lambda x: float(np.quantile(x, 0.75))),
                M_med=("M","median"),
                M_q25=("M", lambda x: float(np.quantile(x, 0.25))),
                M_q75=("M", lambda x: float(np.quantile(x, 0.75))),
           )).reset_index().sort_values("cluster_id")
    tmpl = {
        "K": model["K"],
        "features": ["log_c8","M8_24"],
        "clusters": [
            {
                "cluster_id": int(r.cluster_id),
                "n": int(r.n), "n_repo": int(r.n_repo),
                "log_c8": {"median": float(r.log_c8_med), "q25": float(r.log_c8_q25), "q75": float(r.log_c8_q75)},
                "M8_24":  {"median": float(r.M_med),      "q25": float(r.M_q25),      "q75": float(r.M_q75)},
                "suggested_stage": "",   # 可选：我也可以另写一个 helper 自动填建议
                "stage": ""              # 由你一次性确认 Rising/Peak/Cooling
            } for _, r in g.iterrows()
        ]
    }
    return tmpl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True,
                    help="03_features_weekly_v1_<STAMP>_filtered.csv（只留 age>=24）")
    ap.add_argument("--k", type=int, default=None, help="K*（细扫确定）")
    ap.add_argument("--alpha", type=float, default=None, help="α*（细扫确定）")
    ap.add_argument("--grid_csv", type=str, default=None,
                    help="如未显式给 K/α，则从 04 的 _fine.csv 里自动挑 top-1")
    ap.add_argument("--outdir", type=str, default=None,
                    help="默认：data/05_a_kmeans/<STAMP>/")
    args = ap.parse_args()

    stamp = infer_stamp(args.features_csv)
    outdir = Path(args.outdir or f"data/05_a_kmeans/{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    if (args.k is None or args.alpha is None):
        if not args.grid_csv:
            raise ValueError("未提供 --k/--alpha，且未提供 --grid_csv 以自动挑选")
        K, alpha = pick_from_grid(args.grid_csv)
    else:
        K, alpha = args.k, args.alpha

    df = load_features(args.features_csv)
    model = fit_kmeans(df, K, alpha)

    # 组织模型 JSON（说明空间、特征等，方便 05_b 直接使用）
    model_json = {
        "features": ["log_c8","M8_24"],
        "centers_space": "std_unweighted",  # 未加权的标准化空间
        "scaler": "robust",
        "alpha": model["alpha"], "K": model["K"],
        "scaler_center": model["scaler_center"],
        "scaler_scale":  model["scaler_scale"],
        "centers_std":   model["centers_std"],
        "n_init": model["n_init"], "max_iter": model["max_iter"],
        "fit_samples": model["fit_samples"], "fit_repos": model["fit_repos"]
    }

    tmpl_json = build_cluster_template(df, model)

    # 落盘
    fp_model = outdir / f"05_global_models_v1_{stamp}.json"
    fp_tmpl  = outdir / f"05_labeling_template_v1_{stamp}_K{K}.json"
    with open(fp_model, "w", encoding="utf-8") as f:
        json.dump({"K"+str(K): model_json}, f, ensure_ascii=False, indent=2)
    with open(fp_tmpl, "w", encoding="utf-8") as f:
        json.dump(tmpl_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] model -> {fp_model}")
    print(f"[OK] tmpl  -> {fp_tmpl}")
    print(f"[INFO] K={K}, α={alpha:.2f}, samples={model['fit_samples']}, repos={model['fit_repos']}")

if __name__ == "__main__":
    main()
