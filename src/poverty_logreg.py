"""Logistic regression for household poverty: tune, pick a threshold, test once, export.

Same split by survey year as poverty_model.py:
  train 2007-2019 -> choose C, class weights and the decision threshold on 2023-2024
  refit on train + valid -> report once on 2025
The exported model is refit on every year with a poverty label.

Usage: python src/poverty_logreg.py
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
        "recall_pobres": tp / (tp + fn),
        "recall_no_pobres": tn / (tn + fp),
        "pct_predicho_pobre": float(pred.mean()),
    }


def main():
    h = add_features(load())
    train, valid, test = (h[h["anio"].isin(y)] for y in (TRAIN_YEARS, VALID_YEARS, TEST_YEARS))

    # 1) Regularisation strength and class weights, judged on the validation years.
    rows = []
    for cw in (None, "balanced"):
        for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10):
            m = build(C, cw).fit(train[FEATURES], train[TARGET])
            p = m.predict_proba(valid[FEATURES])[:, 1]
            rows.append({"class_weight": str(cw), "C": C,
                         "roc_auc": roc_auc_score(valid[TARGET], p),
                         "auprc": average_precision_score(valid[TARGET], p),
                         "brier": brier_score_loss(valid[TARGET], p)})
    grid = pd.DataFrame(rows)
    print(grid.round(4).to_string(index=False))
    best = grid.sort_values(["roc_auc", "brier"], ascending=[False, True]).iloc[0]
    C, cw = float(best["C"]), (None if best["class_weight"] == "None" else "balanced")
    print(f"\nelegido: C={C}, class_weight={cw}")

    # 2) Decision threshold: balanced accuracy, because with 60% of households poor
    #    F1 rewards calling everyone poor.
    model = build(C, cw).fit(train[FEATURES], train[TARGET])
    p_valid = model.predict_proba(valid[FEATURES])[:, 1]
    grid_t = np.round(np.arange(0.30, 0.801, 0.01), 2)
    threshold = float(grid_t[np.argmax(
        [balanced_accuracy_score(valid[TARGET], p_valid >= t) for t in grid_t])])
    print(f"umbral elegido en validación: {threshold:.2f}")
    print("validación @0.50:", {k: round(v, 3) for k, v in metrics(valid[TARGET], p_valid, 0.5).items()})
    print(f"validación @{threshold:.2f}:", {k: round(v, 3) for k, v in metrics(valid[TARGET], p_valid, threshold).items()})

    # 3) Refit on train + valid and look at 2025 once.
    fit_tv = pd.concat([train, valid])
    model = build(C, cw).fit(fit_tv[FEATURES], fit_tv[TARGET])
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = metrics(test[TARGET], p_test, threshold)
    print(f"\nTEST {TEST_YEARS} @{threshold:.2f}:", {k: round(v, 3) for k, v in test_metrics.items()})
    print(f"pobreza real en test: {test[TARGET].mean():.3f}")

    # Calibration on test: observed poverty within each tenth of predicted probability.
    cal = pd.DataFrame({"p": p_test, "y": test[TARGET].to_numpy()})
    cal["bin"] = pd.qcut(cal["p"], 10, labels=False)
    print("\ncalibración en test (deciles de probabilidad):")
    print(cal.groupby("bin").agg(predicho=("p", "mean"), observado=("y", "mean"), n=("y", "size")).round(3).to_string())

    # 4) Final model on every labelled year, with readable coefficients.
    final = build(C, cw).fit(h[FEATURES], h[TARGET])
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
        "features": FEATURES, "C": C, "class_weight": cw, "threshold": threshold,
        "train_years": TRAIN_YEARS, "valid_years": VALID_YEARS, "test_years": TEST_YEARS,
        "test": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "odds_ratios": {k: round(float(v), 4) for k, v in odds.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("exportado models/poverty_logreg.onnx y .json")


if __name__ == "__main__":
    main()
