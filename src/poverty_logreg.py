"""Logistic regression for household poverty: tune, pick a threshold, test, export.

Same survey years as poverty_model.py, validated forward in time:
  each validation year (2023, 2024) is predicted by a model fitted on the years before it
  -> choose C, the recency half-life and the decision threshold on those predictions
  refit on train + valid -> report on 2025
The exported model is refit on every year with a poverty label.

Rows are weighted so that every survey year counts the same before a recency decay.
Without it 2007-2009 are 64% of the rows and the model predicts the poverty level of
those years: it fell 8 points short in 2025.

Usage: PYTHONPATH=src python src/poverty_logreg.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from poverty_model import (
    BINARY, CATEGORICAL, NUMERIC, TARGET, TEST_YEARS, TRAIN_YEARS, VALID_YEARS, load,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# Ratios read better than raw counts for a linear model.
EXTRA_NUMERIC = ["menores_por_persona", "mayores_por_persona"]
NUM = NUMERIC + EXTRA_NUMERIC
FEATURES = NUM + BINARY + CATEGORICAL

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 1)
# Years for a survey's weight to halve; None = every row weighs the same (the old behaviour).
HALF_LIFE_GRID = (None, 1, 1.5, 2, 3, 4, 6)


def year_weights(anio: pd.Series, target_year: int, half_life) -> np.ndarray:
    """Row weights (mean 1) for a model meant to predict `target_year`."""
    if half_life is None:
        return np.ones(len(anio))
    w = 0.5 ** ((target_year - anio) / half_life) / anio.map(anio.value_counts())
    return (w / w.mean()).to_numpy()


def fit(data: pd.DataFrame, target_year: int, C: float, half_life) -> Pipeline:
    return build(C, None).fit(data[FEATURES], data[TARGET],
                              clf__sample_weight=year_weights(data["anio"], target_year, half_life))


def add_features(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["menores_por_persona"] = h["n_menores"] / h["n_personas"]
    h["mayores_por_persona"] = h["n_mayores"] / h["n_personas"]
    return h


def build(C: float, class_weight) -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        # A missing education band becomes its own category (-1) instead of being guessed.
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=3000))])


def metrics(y, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, proba),
        "auprc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "recall_pobres": float(tp / (tp + fn)),
        "recall_no_pobres": float(tn / (tn + fp)),
        "pct_predicho_pobre": float(pred.mean()),
        # The model's estimate of the poverty rate: the mean probability, not the share
        # above the threshold (the threshold is tuned for balanced accuracy, not for that).
        "prob_media": float(np.mean(proba)),
    }


def main():
    h = add_features(load())
    train, valid, test = (h[h["anio"].isin(y)] for y in (TRAIN_YEARS, VALID_YEARS, TEST_YEARS))
    fit_tv = pd.concat([train, valid])

    # 1) Regularisation and recency half-life. Each validation year is predicted from the
    #    years before it. Judged by Brier score: the dashboard publishes the predicted share
    #    of poor households, so the probabilities have to be right, not just the ranking.
    rows, oof = [], {}
    for half_life in HALF_LIFE_GRID:
        for C in C_GRID:
            p = pd.Series(index=valid.index, dtype=float)
            for year in VALID_YEARS:
                past, now = fit_tv[fit_tv["anio"] < year], valid[valid["anio"] == year]
                p[now.index] = fit(past, year, C, half_life).predict_proba(now[FEATURES])[:, 1]
            oof[(half_life, C)] = p.to_numpy()
            by_year = [(roc_auc_score(g[TARGET], p[g.index]), brier_score_loss(g[TARGET], p[g.index]))
                       for _, g in valid.groupby("anio")]
            rows.append({"half_life": half_life, "C": C,
                         "roc_auc": np.mean([a for a, _ in by_year]),
                         "brier": np.mean([b for _, b in by_year]),
                         "pred_pobre": p.mean()})
    grid = pd.DataFrame(rows)
    print(grid.sort_values("brier").round(4).to_string(index=False))
    best = grid.sort_values(["brier", "roc_auc"], ascending=[True, False]).iloc[0]
    C = float(best["C"])
    half_life = None if pd.isna(best["half_life"]) else float(best["half_life"])
    print(f"\nelegido: C={C}, vida media={half_life} años "
          f"(pobreza real en validación: {valid[TARGET].mean():.3f})")

    # 2) Decision threshold: balanced accuracy, because with 60% of households poor
    #    F1 rewards calling everyone poor.
    p_valid = oof[(half_life, C)]
    grid_t = np.round(np.arange(0.30, 0.801, 0.01), 2)
    threshold = float(grid_t[np.argmax(
        [balanced_accuracy_score(valid[TARGET], p_valid >= t) for t in grid_t])])
    print(f"umbral elegido en validación: {threshold:.2f}")
    valid_metrics = metrics(valid[TARGET], p_valid, threshold)
    print(f"validación @{threshold:.2f}:", {k: round(v, 3) for k, v in valid_metrics.items()})

    # 3) Refit on train + valid and report on 2025.
    model = fit(fit_tv, TEST_YEARS[0], C, half_life)
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = metrics(test[TARGET], p_test, threshold)
    print(f"\nTEST {TEST_YEARS} @{threshold:.2f}:", {k: round(v, 3) for k, v in test_metrics.items()})
    print(f"pobreza real en test: {test[TARGET].mean():.3f}")

    # Calibration on test: observed poverty within each tenth of predicted probability.
    cal = pd.DataFrame({"p": p_test, "y": test[TARGET].to_numpy()})
    cal["bin"] = pd.qcut(cal["p"], 10, labels=False)
    print("\ncalibración en test (deciles de probabilidad):")
    print(cal.groupby("bin").agg(predicho=("p", "mean"), observado=("y", "mean"), n=("y", "size")).round(3).to_string())

    # 4) Final model on every labelled year, weighted towards the next survey.
    final = fit(h, int(h["anio"].max()) + 1, C, half_life)
    names = [n.split("__", 1)[1] for n in final.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(final.named_steps["clf"].coef_[0], index=names)
    odds = np.exp(coef).sort_values()
    print("\nrazón de momios (numéricas: por 1 desviación estándar):")
    print(pd.concat([odds.head(8), odds.tail(8)]).round(2).to_string())

    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    # The ColumnTransformer selects columns by name, so the graph takes one input per feature.
    onx = to_onnx(final, initial_types=[(f, FloatTensorType([None, 1])) for f in FEATURES],
                  options={id(final): {"zipmap": False}}, target_opset=17)
    MODELS.mkdir(exist_ok=True)
    (MODELS / "poverty_logreg.onnx").write_bytes(onx.SerializeToString())

    import onnxruntime as rt
    sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    feed = {f: h[[f]].to_numpy(np.float32) for f in FEATURES}
    p_onnx = sess.run(None, feed)[1][:, 1]
    p_skl = final.predict_proba(h[FEATURES])[:, 1]
    print(f"\nONNX vs sklearn: dif. máxima de probabilidad {np.abs(p_onnx - p_skl).max():.2e}")

    (MODELS / "poverty_logreg.json").write_text(json.dumps({
        "features": FEATURES, "C": C, "half_life_years": half_life, "threshold": threshold,
        "train_years": TRAIN_YEARS, "valid_years": VALID_YEARS, "test_years": TEST_YEARS,
        "valid": {k: round(float(v), 4) for k, v in valid_metrics.items()},
        "test": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "test_poverty_rate": round(float(test[TARGET].mean()), 4),
        "odds_ratios": {k: round(float(v), 4) for k, v in odds.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("exportado models/poverty_logreg.onnx y .json")


if __name__ == "__main__":
    main()
