"""Baseline: predict whether a household is poor from non-income characteristics.

Split by survey year, never at random, so the score reflects use on a future survey:
  train  2007-2011, 2018, 2019
  valid  2023, 2024   (model comparison)
  test   2025         (reported once, for the model chosen on validation)

Income variables are excluded on purpose: poverty is defined from income, so using
them would leak the answer.

Usage: python src/poverty_model.py
"""
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HOGARES = ROOT / "data" / "processed" / "hogares.parquet"
OUT = ROOT / "data" / "processed" / "poverty_baseline.json"

NUMERIC = ["totper", "n_ocupados", "n_menores", "n_mayores", "jefe_edad", "edad_mean",
           "ocupados_por_persona"]
BINARY = ["rural", "jefa_mujer", "jefe_ocupado", "jefe_agricultura"]
CATEGORICAL = ["depto", "jefe_educ", "max_educ"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
TARGET = "pobre"

TRAIN_YEARS = [2007, 2008, 2009, 2010, 2011, 2018, 2019]
VALID_YEARS = [2023, 2024]
TEST_YEARS = [2025]
SEED = 42


def load() -> pd.DataFrame:
    h = pd.read_parquet(HOGARES)
    h = h[h[TARGET].notna() & h["depto"].notna()].copy()
    h["rural"] = (h["ur"] == 2).astype(float)
    h["ocupados_por_persona"] = h["n_ocupados"] / h["n_personas"]
    return h


def models() -> dict:
    linear_prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": Pipeline([("prep", linear_prep), ("clf", LogisticRegression(max_iter=2000))]),
        # Trees need no scaling and handle missing values natively.
        "lightgbm": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=50, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=SEED, verbose=-1),
    }


def score(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba) if len(set(proba)) > 1 else 0.5,
        "auprc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def main():
    h = load()
    parts = {name: h[h["anio"].isin(years)] for name, years in
             [("train", TRAIN_YEARS), ("valid", VALID_YEARS), ("test", TEST_YEARS)]}
    for name, part in parts.items():
        print(f"{name:5s} {len(part):6,d} hogares, {part[TARGET].mean():.1%} pobres")

    rows, fitted = [], {}
    for name, model in models().items():
        X = parts["train"][FEATURES]
        if name == "lightgbm":
            X = X.astype({c: "category" for c in CATEGORICAL})
        model.fit(X, parts["train"][TARGET])
        fitted[name] = model
        for split in ("train", "valid"):
            Xs = parts[split][FEATURES]
            if name == "lightgbm":
                Xs = Xs.astype({c: "category" for c in CATEGORICAL})
            rows.append({"model": name, "split": split, **score(model, Xs, parts[split][TARGET])})
    table = pd.DataFrame(rows)
    print("\n" + table.round(3).to_string(index=False))

    valid = table[table["split"] == "valid"].set_index("model")
    best = valid["roc_auc"].idxmax()
    Xt = parts["test"][FEATURES]
    if best == "lightgbm":
        Xt = Xt.astype({c: "category" for c in CATEGORICAL})
    test = score(fitted[best], Xt, parts["test"][TARGET])
    print(f"\nmejor en validación: {best}  |  test {TEST_YEARS}: "
          + ", ".join(f"{k}={v:.3f}" for k, v in test.items()))

    by_year = []
    for year, part in h[h["anio"].isin(VALID_YEARS + TEST_YEARS)].groupby("anio"):
        Xy = part[FEATURES]
        if best == "lightgbm":
            Xy = Xy.astype({c: "category" for c in CATEGORICAL})
        by_year.append({"anio": int(year), **score(fitted[best], Xy, part[TARGET])})
    print(pd.DataFrame(by_year).round(3).to_string(index=False))

    if best == "lightgbm":
        imp = pd.Series(fitted[best].booster_.feature_importance("gain"), index=FEATURES)
        print("\nimportancia (ganancia, %):")
        print((imp / imp.sum() * 100).sort_values(ascending=False).round(1).to_string())
    else:
        names = fitted[best].named_steps["prep"].get_feature_names_out()
        coef = pd.Series(fitted[best].named_steps["clf"].coef_[0], index=names)
        print("\ncoeficientes más fuertes:")
        print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(12).round(3).to_string())

    OUT.write_text(json.dumps({
        "features": FEATURES, "train_years": TRAIN_YEARS, "valid_years": VALID_YEARS,
        "test_years": TEST_YEARS, "best": best, "test": test,
        "results": table.round(4).to_dict("records"),
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
