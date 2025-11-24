#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_k_sweep_v1.py  (coarse-to-fine ready, zero-args friendly)
------------------------------------------------------------
做两件事（04 步到此为止，不打标签）：
  1) 生成折名册（GroupKFold=5，按 repo 分组）：
     -> 04_a_fold_assign_v1_<STAMP>.csv
  2) 在“仅训练折”上对 (K, α) 网格做 silhouette/CH 评分（含提速策略）：
     -> 04_b_alpha_k_grid_v1_<STAMP>.csv

特征与变换（v1）：
  - 过滤：dead_flag==0 且 age_weeks≥24
  - log1p(c8_total) -> RobustScaler([log_c8, M8_24]) -> α 加权（列缩放，w_c=2-α, w_m=α）
  - 先标准化、后加权 ⇒ 每折 scaler 拟合一次；不同 α 只做列乘常数（等价不损质）

加速：
  - silhouette 采样估计（--sil_sample，默认 20000；0 表示全量）
  - 网格阶段：n_init=10, max_iter=200；候选再复核可调大
  - float32 计算；尝试 algorithm='elkan'（若 sklearn 支持，否则回退）

用法（零参数右键）：
  - 自动从 data/ 找最新的 03_features_weekly_v1_*_filtered.csv
  - 自动推断 <STAMP> 并输出到 data/04_cluster/<STAMP>/

用法（可覆盖默认）：
  --features_csv PATH
  --out_dir PATH
  --stamp 20251029_084949
  --n_splits 5 --kmin 2 --kmax 5
  --alpha_range 0.6 1.6 0.1
  --sil_sample 20000
  --n_init 10 --max_iter 200
"""

import os
import re
import glob
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# ---------- utilities ----------
STAMP_RE = re.compile(r"(\d{8}_\d{6})")

def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def infer_stamp_from_name(path: str) -> str:
    m = STAMP_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"无法从文件名推断 <STAMP>：{path}")
    return m.group(1)

def find_latest_filtered_csv(base_dir: str) -> str:
    pattern = os.path.join(base_dir, "data", "03_features_weekly_v1_*_filtered.csv")
    cands = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(f"未找到匹配文件：{pattern}")
    def stamp_key(p: str):
        m = STAMP_RE.search(os.path.basename(p))
        return m.group(1) if m else "00000000_000000"
    cands.sort(key=stamp_key)
    return cands[-1]

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "week_date_utc" not in df.columns and "week_date" in df.columns:
        df = df.rename(columns={"week_date": "week_date_utc"})
    required = {"repo", "week_unix", "week_date_utc", "c8_total", "M8_24", "dead_flag", "age_weeks"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列：{missing}")
    return df

# ---------- fold assignment ----------
def make_fold_assign(df_in: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    df = df_in.copy()
    df = df[(df["dead_flag"] == 0) & (df["age_weeks"] >= 24)].sort_values(["repo", "week_unix"])
    gkf = GroupKFold(n_splits=n_splits)
    repos = df["repo"].values
    X_dummy = np.zeros((len(df), 1), dtype=np.int8)
    fold_of_repo = {}
    for fid, (_, test_idx) in enumerate(gkf.split(X_dummy, groups=repos)):
        for r in pd.Series(repos[test_idx]).unique():
            fold_of_repo[r] = fid
    fa = pd.DataFrame({"repo": list(fold_of_repo.keys()), "fold_id": list(fold_of_repo.values())})
    fa = fa.sort_values("repo").reset_index(drop=True)
    assert fa["repo"].is_unique
    return fa

# ---------- weighting ----------
def alpha_weights(alpha: float) -> Tuple[float, float]:
    return 2.0 - alpha, alpha  # w_c (log_c8), w_m (M8_24)

# ---------- scoring ----------
def kmeans_score(X: np.ndarray, k: int, n_init: int, max_iter: int,
                 random_state: int, sil_sample: int) -> Tuple[float, float]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] <= k:
        return np.nan, np.nan
    # 兼容不同 sklearn 版本的 algorithm 参数
    try:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                    random_state=random_state, algorithm="elkan")
    except TypeError:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                    random_state=random_state)
    y = km.fit_predict(X)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        if sil_sample and X.shape[0] > sil_sample:
            sil = silhouette_score(X, y, sample_size=sil_sample, random_state=random_state)
        else:
            sil = silhouette_score(X, y)
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X, y)
    except Exception:
        ch = np.nan
    return sil, ch

def run_grid(df_in: pd.DataFrame, fold_assign: pd.DataFrame,
             k_values: List[int], alphas: List[float],
             n_splits: int, n_init: int, max_iter: int, random_state: int,
             sil_sample: int) -> pd.DataFrame:
    # 仅非 Dead；基础特征
    df = df_in[df_in["dead_flag"] == 0].copy()
    df["log_c8"] = np.log1p(df["c8_total"]).astype(np.float32)
    df["M"] = df["M8_24"].astype(np.float32)

    # 合并 fold_id
    fmap = dict(zip(fold_assign["repo"], fold_assign["fold_id"]))
    df["fold_id"] = df["repo"].map(fmap)
    if df["fold_id"].isna().any():
        miss = df[df["fold_id"].isna()]["repo"].unique()
        raise RuntimeError(f"以下仓库没有 fold_id：{miss[:5]} 等；请检查折名册是否覆盖参与建模的仓库。")

    # 每折：拟合一次 RobustScaler，并缓存“已标准化”的基础特征
    fold_scaled = {}
    for f in range(n_splits):
        Xtr_base = df.loc[df["fold_id"] != f, ["log_c8", "M"]].to_numpy(dtype=np.float32)
        scaler = RobustScaler()
        Xs = scaler.fit_transform(Xtr_base).astype(np.float32)
        fold_scaled[f] = Xs  # 后续对 α 仅做列乘常数

    results = []
    total = len(k_values) * len(alphas)
    with tqdm(total=total, ncols=100, desc="[GRID] (K, α) scoring") as bar:
        for k in k_values:
            for alpha in alphas:
                w_c, w_m = alpha_weights(alpha)
                sils, chs = [], []
                for f in range(n_splits):
                    Xs = fold_scaled[f]
                    Xw = Xs.copy()
                    Xw[:, 0] *= w_c
                    Xw[:, 1] *= w_m
                    sil, ch = kmeans_score(Xw, k, n_init, max_iter, random_state, sil_sample)
                    if not (np.isnan(sil) and np.isnan(ch)):
                        sils.append(sil); chs.append(ch)
                row = {
                    "K": k, "alpha": alpha,
                    "sil_mean": float(np.nanmean(sils)) if sils else np.nan,
                    "sil_std":  float(np.nanstd(sils))  if sils else np.nan,
                    "ch_mean":  float(np.nanmean(chs))  if chs else np.nan,
                    "ch_std":   float(np.nanstd(chs))   if chs else np.nan,
                    "used_folds": len(sils),
                    "n_splits": n_splits,
                    "n_init": n_init, "max_iter": max_iter, "random_state": random_state,
                    "log1p": True, "scaler": "robust",
                    "w_c_formula": "2 - alpha", "w_m_formula": "alpha",
                    "sil_sample": sil_sample
                }
                results.append(row); bar.update(1)

    out = pd.DataFrame(results).sort_values(["sil_mean", "ch_mean"], ascending=[False, False]).reset_index(drop=True)
    return out

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="04_k_sweep_v1: 折名册 + (K,α) 网格评分（仅训练折）")
    parser.add_argument("--features_csv", type=str, default=None, help="03_features_weekly_v1_*_filtered.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录，默认 data/04_cluster/<STAMP>/")
    parser.add_argument("--stamp", type=str, default=None, help="<STAMP>（默认从 features_csv 名称推断）")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=5)
    parser.add_argument("--alpha_range", nargs=3, type=float, default=[0.6, 1.6, 0.1],
                        help="α 起点 终点 步长，例如：0.6 1.6 0.1")
    parser.add_argument("--sil_sample", type=int, default=20000, help="silhouette 采样量；0 表示全量（最慢）")
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    prj = project_root()
    features_csv = args.features_csv or find_latest_filtered_csv(prj)
    stamp = args.stamp or infer_stamp_from_name(features_csv)
    out_dir = args.out_dir or os.path.join(prj, "data", "04_cluster", stamp)
    ensure_dir(out_dir)

    print("=" * 80)
    print(f"[INFO] 04_k_sweep_v1 | STAMP={stamp}")
    print(f"[INFO] features_csv : {features_csv}")
    print(f"[INFO] out_dir      : {out_dir}")
    print(f"[INFO] k in [{args.kmin}, {args.kmax}], alpha={args.alpha_range}, "
          f"n_splits={args.n_splits}, n_init={args.n_init}, max_iter={args.max_iter}, "
          f"random_state={args.random_state}, sil_sample={args.sil_sample}")
    print("=" * 80)

    df = load_features(features_csv)
    n_full = len(df); n_repo = df["repo"].nunique()
    tmin = pd.to_datetime(df["week_unix"], unit="s", utc=True).min().date()
    tmax = pd.to_datetime(df["week_unix"], unit="s", utc=True).max().date()
    print(f"[INFO] 03_filtered 行数：{n_full:,}；仓库数：{n_repo:,}；周范围：{tmin} → {tmax} (UTC)")

    # 04_a：折名册
    print("\n[STEP] 04_a 生成折名册（GroupKFold 按 repo 分组） ...")
    fold_assign = make_fold_assign(df, n_splits=args.n_splits)
    fold_csv = os.path.join(out_dir, f"04_a_fold_assign_v1_{stamp}.csv")
    fold_assign.to_csv(fold_csv, index=False)
    print(f"[OK] 写出：{fold_csv} | 行数：{len(fold_assign):,}（每仓一个 fold_id）")

    # 04_b：(K,α) 网格评分（仅训练折）
    print("\n[STEP] 04_b (K, α) 网格评分（仅训练折） ...")
    k_values = list(range(args.kmin, args.kmax + 1))
    a0, a1, astep = args.alpha_range
    alphas = [round(x, 10) for x in np.arange(a0, a1 + 1e-9, astep)]
    print(f"[INFO] K 候选：{k_values}")
    print(f"[INFO] α 候选：{alphas}")

    grid_df = run_grid(df, fold_assign, k_values, alphas,
                       args.n_splits, args.n_init, args.max_iter, args.random_state,
                       args.sil_sample)
    grid_csv = os.path.join(out_dir, f"04_b_alpha_k_grid_v1_{stamp}.csv")
    grid_df.to_csv(grid_csv, index=False)
    print(f"[OK] 写出：{grid_csv} | 行数：{len(grid_df):,}")

    print("\n[DONE] 本轮到此结束：已生成 04_a 折名册 + 04_b 网格评分。请选择 (K*, α*) 进入下一步。")

if __name__ == "__main__":
    main()
