# -*- coding: utf-8 -*-
"""
v2/04_select_k_v2.py — K selection only (no clustering) + ONE PCA CSV + progress bar.

This parameterized version keeps all original defaults intact:
- By default it auto-picks the latest 03 STAMP, evaluates K=2..6,
  and writes outputs to v2_data/04_cluster/ with the same filenames as before.

You can also extend the experiment AFTER the 2..6 run, e.g., test K=7..10
and save into v2_data/04_cluster/tuning_k/ WITHOUT overwriting previous results.
See "CLI examples" below.

Outputs:
  1) v2_data/04_cluster[/...]/04_kselect_scores_v2_<STAMP>[<outtag>].csv
  2) v2_data/04_cluster[/...]/04_kselect_summary_v2_<STAMP>[<outtag>].json
  3) v2_data/04_cluster[/...]/04_pca_all_v2_<STAMP>[<outtag>].csv
     (ONE table with all PCA details: folds + full ref)

Design (agreed):
  - Features: commits_8w_sum, contributors_8w_unique, issues_closed_8w_count, releases_8w_count
  - Filter: dead_flag == 0 (Zero/Low windows kept)
  - Per-fold TRAIN pipeline: log1p -> StandardScaler -> PCA (>=90% cumulative EVR)
    * Fail-safe: if PC1 EVR > 0.70 AND |PC1 loading on 'issues_closed_8w_count'| >= 0.70 -> whiten=True
  - Metrics on TRAIN fold only: silhouette (euclidean), Calinski–Harabasz (CH)
  - ONE PCA CSV includes: scope('fold'/'full'), fold_id, component, feature, loading, evr, cum_evr,
      n_components_selected, whiten, pc1_var_ratio, pc1_top_feature, pc1_top_loading_abs,
      thresholds (evr_threshold, pc1_ratio_threshold, issues_loading_min)
  - Progress bar shows total steps = N_SPLITS * len(KS)

CLI examples (PyCharm Parameters or shell):
  # Default behavior (latest STAMP, K=2..6, default outdir)
  # -> no parameters needed

  # Extend AFTER 2..6: run only K=7..10 and write into a subfolder
  --kmin 7 --kmax 10 --outdir v2_data/04_cluster/tuning_k --outtag _k7_10_try1

  # (Optional) Force a specific STAMP (if you duplicated 03 features with a new timestamp)
  --stamp 20251124_170000 --kmin 7 --kmax 10 --outdir v2_data/04_cluster/tuning_k --outtag _k7_10_try1
"""

from __future__ import annotations
from pathlib import Path
import re, json, os, argparse
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# -------- progress bar (tqdm with safe fallback) --------
def _tqdm_iter(iterable=None, total=None, **kwargs):
    try:
        if os.environ.get("NO_TQDM", ""):
            raise ImportError
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, **kwargs)
    except Exception:
        if iterable is not None:
            return iterable
        class _Dummy:
            def update(self, n=1): pass
            def close(self): pass
        return _Dummy()

class _ManualBar:
    """Manual progress bar wrapper with safe fallback."""
    def __init__(self, total, **kwargs):
        try:
            if os.environ.get("NO_TQDM", ""):
                raise ImportError
            from tqdm import tqdm  # type: ignore
            self._bar = tqdm(total=total, **kwargs)
            self._dummy = False
        except Exception:
            self._bar = None
            self._dummy = True
            self._total = total
            self._count = 0
    def update(self, n=1):
        if not self._dummy:
            self._bar.update(n)
        else:
            self._count += n
    def close(self):
        if not self._dummy:
            self._bar.close()

# ---------------- paths & stamp ----------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
D03  = ROOT / "v2_data" / "03_features"
D04_DEFAULT  = ROOT / "v2_data" / "04_cluster"

def latest_stamp_from_03(d03: Path) -> str:
    cands = sorted(d03.glob("03_features_weekly_v2_*.csv"))
    if not cands:
        raise FileNotFoundError("No 03_features_weekly_v2_*.csv under v2_data/03_features/")
    m = re.search(r"_(\d{8}_\d{6})\.csv$", cands[-1].name)
    if not m:
        raise RuntimeError(f"Cannot parse STAMP from filename: {cands[-1].name}")
    return m.group(1)

# ---------------- argparse (NEW) ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--stamp", type=str, default=None, help="Override STAMP; by default use latest 03 STAMP.")
ap.add_argument("--kmin", type=int, default=None, help="If set with --kmax, override KS to range(kmin..kmax).")
ap.add_argument("--kmax", type=int, default=None, help="If set with --kmin, override KS to range(kmin..kmax).")
ap.add_argument("--outdir", type=str, default=None, help="Output directory; default v2_data/04_cluster/")
ap.add_argument("--outtag", type=str, default="", help="Optional tag appended to filenames to avoid overwrite.")
ap.add_argument("--folds", type=int, default=None, help="Override number of GroupKFold splits (default 5).")
ap.add_argument("--seed", type=int, default=None, help="Override RANDOM_STATE for KMeans/PCA (default 123).")
args, _ = ap.parse_known_args()

STAMP = args.stamp or latest_stamp_from_03(D03)
F_FEAT = D03 / f"03_features_weekly_v2_{STAMP}.csv"

# pick output dir
if args.outdir:
    D04 = (ROOT / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
else:
    D04 = D04_DEFAULT
D04.mkdir(parents=True, exist_ok=True)
TAG = args.outtag if args.outtag else ""  # prepend underscore in caller if desired (e.g., _k7_10)

# ---------------- config (defaults preserved) ----------------
RANDOM_STATE = 123 if args.seed is None else int(args.seed)
KS = [2,3,4,5,6]
if args.kmin is not None and args.kmax is not None:
    KS = list(range(int(args.kmin), int(args.kmax)+1))

N_SPLITS = 5 if args.folds is None else int(args.folds)

EXPLAINED_VAR_THRESHOLD = 0.90   # PCA cumulative explained variance threshold
PC1_DOMINANCE_THRESHOLD = 0.70   # if PC1 EVR > 70%
PC1_ISSUES_LOADING_MIN  = 0.70   # and |PC1 loading on 'issues_closed'| >= 0.70 => whiten=True

FEATURES = [
    "commits_8w_sum",
    "contributors_8w_unique",
    "issues_closed_8w_count",
    "releases_8w_count",
]
ISSUES_COL = "issues_closed_8w_count"

# ---------------- helpers ----------------
def log1p_df(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log1p transform."""
    return np.log1p(df)

def _n_comp(pca_obj) -> int:
    """Robustly get n_components used by PCA object."""
    return int(getattr(pca_obj, "n_components_", getattr(pca_obj, "n_components", 0)))

def choose_pca(Xz_train: np.ndarray, feature_names: List[str]) -> Tuple[PCA, int, bool, np.ndarray, np.ndarray]:
    """
    Fit PCA on standardized TRAIN; pick #components; decide whitening.
    Returns: (pca, n_components_used, whiten_flag, loadings, evr)
    """
    pca_raw = PCA(n_components=min(Xz_train.shape[1], len(feature_names)),
                  whiten=False, random_state=RANDOM_STATE)
    pca_raw.fit(Xz_train)
    evr = pca_raw.explained_variance_ratio_
    cum = np.cumsum(evr)
    m = int(np.searchsorted(cum, EXPLAINED_VAR_THRESHOLD) + 1)
    m = max(1, min(m, Xz_train.shape[1]))

    loadings = pca_raw.components_
    whiten_flag = False
    if evr[0] > PC1_DOMINANCE_THRESHOLD:
        try:
            idx_issues = feature_names.index(ISSUES_COL)
            if abs(loadings[0, idx_issues]) >= PC1_ISSUES_LOADING_MIN:
                whiten_flag = True
        except ValueError:
            pass

    if whiten_flag:
        pca = PCA(n_components=m, whiten=True, random_state=RANDOM_STATE).fit(Xz_train)
        return pca, _n_comp(pca), True, pca.components_, pca.explained_variance_ratio_
    else:
        pca = PCA(n_components=m, whiten=False, random_state=RANDOM_STATE).fit(Xz_train)
        return pca, _n_comp(pca), False, pca.components_, pca.explained_variance_ratio_

# ---------------- load & filter ----------------
df = pd.read_csv(F_FEAT)
required = {"repo","week_unix","dead_flag", *FEATURES}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns in {F_FEAT.name}: {sorted(missing)}")

df = df[df["dead_flag"] == 0].copy().reset_index(drop=True)
groups = df["repo"].astype(str).values
X_all = df[FEATURES].copy()

print(f"[INFO] STAMP={STAMP} | windows(dead=0)={len(df)} | repos={df['repo'].nunique()} | K={KS} | outdir={D04}")

# ---------------- cross-validated K selection + ONE PCA table ----------------
scores_rows = []
pca_rows    = []

gkf = GroupKFold(n_splits=N_SPLITS)
total_steps = N_SPLITS * len(KS)
pbar = _ManualBar(total=total_steps, desc="K eval (fold x K)", unit="step")

fold_id = 0
for tr_idx, va_idx in gkf.split(X_all, groups=groups):
    fold_id += 1
    Xtr_raw = X_all.iloc[tr_idx].copy()

    # TRAIN preprocess
    Xtr_log = log1p_df(Xtr_raw)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr_log)
    Xtr_z  = scaler.transform(Xtr_log)

    # PCA & fail-safe
    pca, m, whiten_flag, loadings, evr = choose_pca(Xtr_z, FEATURES)
    Xtr_p = pca.transform(Xtr_z)

    # ONE PCA table rows (scope=fold)
    cum = np.cumsum(evr)
    pc1_var_ratio = float(evr[0]) if len(evr)>0 else np.nan
    pc1_top_idx   = int(np.argmax(np.abs(loadings[0,:]))) if loadings.shape[0] >= 1 else 0
    pc1_top_feat  = FEATURES[pc1_top_idx]
    pc1_top_abs   = float(abs(loadings[0, pc1_top_idx])) if loadings.shape[0] >= 1 else np.nan

    for ic in range(_n_comp(pca)):
        for j, feat in enumerate(FEATURES):
            pca_rows.append({
                "scope": "fold",
                "fold": fold_id,
                "component": ic+1,
                "feature": feat,
                "loading": float(loadings[ic, j]),
                "explained_variance_ratio": float(evr[ic] if ic < len(evr) else np.nan),
                "cumulative_explained_variance": float(cum[ic] if ic < len(cum) else np.nan),
                "n_components_selected": int(_n_comp(pca)),
                "whiten": bool(whiten_flag),
                "pc1_var_ratio": pc1_var_ratio,
                "pc1_top_feature": pc1_top_feat,
                "pc1_top_loading_abs": pc1_top_abs,
                "evr_threshold": EXPLAINED_VAR_THRESHOLD,
                "pc1_ratio_threshold": PC1_DOMINANCE_THRESHOLD,
                "issues_loading_min": PC1_ISSUES_LOADING_MIN,
            })

    # Evaluate K on TRAIN fold only (advance progress per K)
    n_tr = Xtr_p.shape[0]
    for k in KS:
        if n_tr <= k:
            pbar.update(1)
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_tr = km.fit_predict(Xtr_p)
        sil = float(silhouette_score(Xtr_p, labels_tr, metric="euclidean"))
        ch  = float(calinski_harabasz_score(Xtr_p, labels_tr))
        scores_rows.append({
            "fold": fold_id, "k": k,
            "silhouette": sil, "calinski_harabasz": ch,
            "n_components": int(_n_comp(pca)),
            "whiten": bool(whiten_flag),
        })
        pbar.update(1)

pbar.close()

# Save per-fold K metrics
scores = pd.DataFrame(scores_rows).sort_values(["k","fold"])
scores_csv = D04 / f"04_kselect_scores_v2_{STAMP}{TAG}.csv"
scores.to_csv(scores_csv, index=False)

# Aggregate (median across folds) & suggest K (you decide finally)
agg = scores.groupby("k").agg(
    silhouette_median=("silhouette","median"),
    silhouette_p25=("silhouette", lambda s: s.quantile(0.25)),
    silhouette_p75=("silhouette", lambda s: s.quantile(0.75)),
    ch_median=("calinski_harabasz","median"),
    ncomp_median=("n_components","median"),
    whiten_ratio=("whiten","mean"),
).reset_index()

suggested = agg.sort_values(["silhouette_median","ch_median"], ascending=[False, False]).iloc[0]["k"]
summary = {
    "stamp": STAMP,
    "folds": N_SPLITS,
    "k_candidates": KS,
    "explained_variance_threshold": EXPLAINED_VAR_THRESHOLD,
    "pc1_dominance_threshold": PC1_DOMINANCE_THRESHOLD,
    "pc1_issues_loading_min": PC1_ISSUES_LOADING_MIN,
    "suggested_k_by_metrics": int(suggested),
    "table": agg.to_dict(orient="records"),
    "note": "Manual selection required. This is only a data-driven suggestion. "
            "This run may extend the initial 2..6 experiment with K=7..10.",
}
with open(D04 / f"04_kselect_summary_v2_{STAMP}{TAG}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------------- FULL-data PCA reference (diagnostic only) ----------------
X_full_log = log1p_df(X_all)
scaler_full = StandardScaler(with_mean=True, with_std=True).fit(X_full_log)
X_full_z = scaler_full.transform(X_full_log)

pca_full_raw = PCA(n_components=min(X_full_z.shape[1], len(FEATURES)),
                   whiten=False, random_state=RANDOM_STATE).fit(X_full_z)
evr_full = pca_full_raw.explained_variance_ratio_
cum_full = np.cumsum(evr_full)
m_full = int(np.searchsorted(cum_full, EXPLAINED_VAR_THRESHOLD) + 1)
m_full = max(1, min(m_full, X_full_z.shape[1]))
whiten_full = False
if evr_full[0] > PC1_DOMINANCE_THRESHOLD:
    if abs(pca_full_raw.components_[0, FEATURES.index(ISSUES_COL)]) >= PC1_ISSUES_LOADING_MIN:
        whiten_full = True

pca_full = PCA(n_components=m_full, whiten=whiten_full, random_state=RANDOM_STATE).fit(X_full_z)
evr_f  = pca_full.explained_variance_ratio_
cum_f  = np.cumsum(evr_f)
loads_f = pca_full.components_

# Append FULL rows into the SAME ONE PCA table
pca_rows_full = []
for ic in range(_n_comp(pca_full)):
    for j, feat in enumerate(FEATURES):
        pca_rows_full.append({
            "scope": "full",
            "fold": np.nan,
            "component": ic+1,
            "feature": feat,
            "loading": float(loads_f[ic, j]),
            "explained_variance_ratio": float(evr_f[ic] if ic < len(evr_f) else np.nan),
            "cumulative_explained_variance": float(cum_f[ic] if ic < len(cum_f) else np.nan),
            "n_components_selected": int(_n_comp(pca_full)),
            "whiten": bool(whiten_full),
            "pc1_var_ratio": float(evr_f[0] if len(evr_f)>0 else np.nan),
            "pc1_top_feature": FEATURES[int(np.argmax(np.abs(loads_f[0,:])))] if loads_f.shape[0] >= 1 else None,
            "pc1_top_loading_abs": float(abs(loads_f[0, int(np.argmax(np.abs(loads_f[0,:])))])) if loads_f.shape[0] >= 1 else np.nan,
            "evr_threshold": EXPLAINED_VAR_THRESHOLD,
            "pc1_ratio_threshold": PC1_DOMINANCE_THRESHOLD,
            "issues_loading_min": PC1_ISSUES_LOADING_MIN,
        })

# Save ONE PCA CSV
pca_df = pd.DataFrame([*pca_rows, *pca_rows_full]).sort_values(["scope","fold","component","feature"])
pca_csv = D04 / f"04_pca_all_v2_{STAMP}{TAG}.csv"
pca_df.to_csv(pca_csv, index=False)

print(f"[OK] Saved K-selection scores  -> {scores_csv}")
print(f"[OK] Saved K-selection summary -> {D04 / f'04_kselect_summary_v2_{STAMP}{TAG}.json'}")
print(f"[OK] Saved ONE PCA table       -> {pca_csv}")
print("[OK] No clustering performed in this step.")
