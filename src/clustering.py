"""Clustering of the department x year panel.

  python src/clustering.py evaluate            compare feature sets, algorithms and k
  python src/clustering.py fit --set nucleo --k 4   fit, profile and export the chosen model
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score,
    davies_bouldin_score, silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "depto_anio.parquet"
MODELS = ROOT / "models"

# Only indicators whose definition is stable across survey years (see README / etl.py).
FEATURE_SETS = {
    # living conditions only: no variable with a mechanical national time trend
    "nucleo": ["tasa_pobreza", "tasa_pobreza_extrema", "ymophg_median_rel",
               "tasa_ocupacion", "tasa_desempleo", "tasa_dependencia"],
    # adds household demographics, which drift nationally year after year
    "amplio": ["tasa_pobreza", "tasa_pobreza_extrema", "ymophg_median_rel",
               "tasa_ocupacion", "tasa_desempleo", "tasa_dependencia",
               "pct_jefatura_femenina", "totper_mean", "edad_mean"],
}
SEED = 42


def load(features):
    df = pd.read_parquet(PANEL).dropna(subset=features).reset_index(drop=True)
    return df, StandardScaler().fit_transform(df[features])


def labels_for(algo, k, X):
    if algo == "kmeans":
        return KMeans(k, n_init=25, random_state=SEED).fit_predict(X)
    if algo == "ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    return GaussianMixture(k, n_init=5, random_state=SEED).fit_predict(X)


def stability(k, X, runs=30):
    """Mean ARI between KMeans on the full panel and on bootstrap resamples."""
    rng = np.random.default_rng(SEED)
    base = KMeans(k, n_init=10, random_state=SEED).fit(X)
    scores = []
    for _ in range(runs):
        idx = rng.choice(len(X), len(X), replace=True)
        km = KMeans(k, n_init=10, random_state=SEED).fit(X[idx])
        scores.append(adjusted_rand_score(base.labels_, km.predict(X)))
    return float(np.mean(scores))


def evaluate():
    for name, feats in FEATURE_SETS.items():
        df, X = load(feats)
        print(f"\n=== set '{name}': {len(feats)} variables, {len(df)} filas, "
              f"{df['anio'].nunique()} cortes")
        rows = []
        for k in range(2, 8):
            for algo in ["kmeans", "ward", "gmm"]:
                lab = labels_for(algo, k, X)
                rows.append({
                    "k": k, "algo": algo,
                    "silhouette": silhouette_score(X, lab),
                    "davies_bouldin": davies_bouldin_score(X, lab),
                    "calinski": calinski_harabasz_score(X, lab),
                    "min_size": int(np.bincount(lab).min()),
                    # how much the clusters just reproduce the survey year (0 = not at all)
                    "ami_anio": adjusted_mutual_info_score(df["anio"], lab),
                    "ami_depto": adjusted_mutual_info_score(df["depto"], lab),
                    "estabilidad": stability(k, X) if algo == "kmeans" else np.nan,
                })
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def fit(set_name, k):
    feats = FEATURE_SETS[set_name]
    df, _ = load(feats)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("kmeans", KMeans(k, n_init=25, random_state=SEED))]).fit(df[feats])
    raw = pipe.predict(df[feats])
    # Renumber clusters from best (0) to worst by mean poverty, so labels carry meaning.
    order = df.assign(c=raw).groupby("c")["tasa_pobreza"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    km = pipe.named_steps["kmeans"]
    km.cluster_centers_ = km.cluster_centers_[list(order)]
    km.labels_ = np.vectorize(remap.get)(km.labels_)
    df["cluster"] = pipe.predict(df[feats])
    assert (df["cluster"] == np.vectorize(remap.get)(raw)).all()

    profile = df.groupby("cluster")[feats].mean().assign(n=df.groupby("cluster").size())
    print(profile.round(3).to_string())
    print("\nCluster por departamento y año:")
    print(df.pivot(index="depto_nombre", columns="anio", values="cluster")
            .astype("Int64").to_string(na_rep="-"))

    out = ROOT / "data" / "processed"
    df[["anio", "depto", "depto_nombre", "cluster"] + feats].to_csv(
        out / "depto_anio_clusters.csv", index=False, encoding="utf-8")

    from skl2onnx import to_onnx  # optional dependency, only needed to export
    onx = to_onnx(pipe, df[feats].to_numpy(np.float32)[:1], target_opset=17)
    MODELS.mkdir(exist_ok=True)
    (MODELS / "clustering_panel.onnx").write_bytes(onx.SerializeToString())
    (MODELS / "clustering_panel.json").write_text(json.dumps({
        "features": feats, "k": k, "set": set_name,
        "profile": profile.round(4).reset_index().to_dict("records"),
        "silhouette": round(float(silhouette_score(
            pipe.named_steps["scaler"].transform(df[feats]), df["cluster"])), 4),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nexportado models/clustering_panel.onnx y .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    f = sub.add_parser("fit")
    f.add_argument("--set", default="nucleo", choices=FEATURE_SETS)
    f.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    evaluate() if args.cmd == "evaluate" else fit(args.set, args.k)
