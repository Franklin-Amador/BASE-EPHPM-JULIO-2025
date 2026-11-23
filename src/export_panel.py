"""Bundle the processed panel into one JSON consumed by the API and the frontend.

Usage: python src/export_panel.py [--front "../EPHPM Front/frontend-EPHPM-2025"]
Writes api_data/panel.json and, with --front, <front>/lib/panel.json.
"""
import argparse
import json
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# Only indicators that are comparable across survey years are published.
# Left out on purpose: tasa_nbi, anosest_mean, pct_media_completa, pct_cuenta_propia
# (their definition changes between questionnaires).
INDICATORS = [
    {"key": "tasa_pobreza", "label": "Pobreza", "format": "pct", "worse": "up",
     "description": "Hogares bajo la línea de pobreza (extrema + relativa)."},
    {"key": "tasa_pobreza_extrema", "label": "Pobreza extrema", "format": "pct", "worse": "up",
     "description": "Hogares cuyo ingreso no cubre la canasta básica de alimentos."},
    {"key": "ymophg_median", "label": "Ingreso mediano del hogar", "format": "lps", "worse": "down",
     "description": "Mediana del ingreso monetario mensual por ocupación principal, en lempiras corrientes.",
     "note": "Lempiras corrientes, sin ajustar por inflación."},
    {"key": "ymophg_median_rel", "label": "Ingreso relativo al país", "format": "ratio", "worse": "down",
     "description": "Ingreso mediano del departamento dividido entre el nacional del mismo año (1.0 = igual al país)."},
    {"key": "tasa_ocupacion", "label": "Ocupación", "format": "pct", "worse": "none",
     "description": "Personas ocupadas entre la población de 15 años o más."},
    {"key": "tasa_desempleo", "label": "Desempleo", "format": "pct", "worse": "up",
     "description": "Desocupados entre la fuerza de trabajo (ocupados + desocupados)."},
    {"key": "tasa_dependencia", "label": "Dependencia demográfica", "format": "ratio", "worse": "up",
     "description": "Menores de 15 y mayores de 64 por cada persona de 15 a 64 años."},
    {"key": "pct_jefatura_femenina", "label": "Jefatura femenina", "format": "pct", "worse": "none",
     "description": "Hogares cuya jefatura la ejerce una mujer."},
    {"key": "totper_mean", "label": "Personas por hogar", "format": "num", "worse": "none",
     "description": "Tamaño promedio del hogar."},
    {"key": "edad_mean", "label": "Edad promedio", "format": "years", "worse": "none",
     "description": "Edad promedio de la población."},
    {"key": "pct_recibe_remesas", "label": "Hogares con remesas", "format": "pct", "worse": "none",
     "description": "Hogares que reciben remesas del exterior."},
    {"key": "pct_agricultura", "label": "Empleo agrícola", "format": "pct", "worse": "none",
     "description": "Ocupados cuya rama de actividad principal es la agricultura.",
     "note": "La encuesta cambió de método en 2021; el salto entre 2019 y 2021 no es solo evolución real."},
    {"key": "pct_rural", "label": "Población rural", "format": "pct", "worse": "none",
     "description": "Población que vive en área rural.",
     "note": "El marco muestral cambió entre 2011 y 2018; los dos tramos no son comparables entre sí."},
]

CLUSTER_NAMES = [
    ("Condiciones más favorables", "Menor pobreza e ingreso por encima del país."),
    ("Condiciones intermedias", "Pobreza e ingreso cercanos al promedio nacional."),
    ("Condiciones más críticas", "Mayor pobreza, ingreso muy por debajo del país y más dependientes por hogar."),
]


# Household groups whose observed poverty rate the dashboard shows.
# These are weighted survey rates, not model coefficients: the model's odds ratios are
# conditional on correlated features (e.g. "rural" flips sign once agriculture is in),
# so they mislead when read one at a time.
POVERTY_FACTORS = [
    ("Nadie en el hogar tiene trabajo", lambda h: h["n_ocupados"] == 0),
    ("La mitad o más del hogar son menores de 15 años", lambda h: h["n_menores"] / h["n_personas"] >= 0.5),
    ("En el hogar viven 6 personas o más", lambda h: h["totper"] >= 6),
    ("Nadie en el hogar pasó de 5 años de estudio", lambda h: h["max_educ"] == 0),
    ("La persona a cargo del hogar trabaja en agricultura", lambda h: h["jefe_agricultura"] == 1),
    ("La persona a cargo estudió menos de 6 años", lambda h: h["jefe_educ"] == 0),
    ("El hogar lo dirige una mujer", lambda h: h["jefa_mujer"] == 1),
    ("El hogar está en zona rural", lambda h: h["ur"] == 2),
    ("Trabaja la mitad o más del hogar", lambda h: h["n_ocupados"] / h["n_personas"] >= 0.5),
    ("Alguien en el hogar tiene estudios superiores", lambda h: h["max_educ"] == 2),
    ("La persona a cargo tiene estudios superiores", lambda h: h["jefe_educ"] == 2),
]


def poverty_factors(hogares: pd.DataFrame, year: int) -> dict:
    """Weighted poverty rate of each household group in one survey year."""
    h = hogares[(hogares["anio"] == year) & hogares["factor"].notna()]
    rate = lambda d: round(float((d["pobre"] * d["factor"]).sum() / d["factor"].sum()), 4)
    factors = []
    for label, in_group in POVERTY_FACTORS:
        group = h[in_group(h).fillna(False)]
        factors.append({"label": label, "rate": rate(group),
                        "share": round(float(group["factor"].sum() / h["factor"].sum()), 4),
                        "n": int(len(group))})
    return {"factors_year": year, "factors_national": rate(h), "factors": factors}


def poverty_block() -> dict:
    """Summary of the household poverty model for the dashboard."""
    meta = json.loads((ROOT / "models" / "poverty_logreg.json").read_text(encoding="utf-8"))
    hogares = pd.read_parquet(PROC / "hogares.parquet")
    hogares = hogares[hogares["pobre"].notna() & hogares["depto"].notna()]
    count = lambda years: int(hogares["anio"].isin(years).sum())
    return {
        "algorithm": "Regresión logística",
        "n_features": len(meta["features"]),
        "train_years": meta["train_years"], "valid_years": meta["valid_years"], "test_years": meta["test_years"],
        "n_train": count(meta["train_years"]), "n_valid": count(meta["valid_years"]), "n_test": count(meta["test_years"]),
        "test": meta["test"],
        **poverty_factors(hogares, int(hogares["anio"].max())),
    }


def geo_key(name: str) -> str:
    """Accent-free, space-free lowercase key; matches GADM NAME_1 after the same transform."""
    s = unicodedata.normalize("NFD", name)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace(" ", "").lower()


def records(df: pd.DataFrame, cols: list) -> list:
    out = []
    for rec in df[cols].to_dict("records"):
        out.append({k: (None if pd.isna(v) else (round(float(v), 4) if isinstance(v, float) else v))
                    for k, v in rec.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "variables.yml").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models" / "clustering_panel.json").read_text(encoding="utf-8"))
    depto = pd.read_parquet(PROC / "depto_anio.parquet")
    nacional = pd.read_csv(PROC / "nacional_anio.csv")
    clusters = pd.read_csv(PROC / "depto_anio_clusters.csv")[["anio", "depto", "cluster"]]
    depto = depto.merge(clusters, on=["anio", "depto"], how="left")
    depto["cluster"] = depto["cluster"].astype("Int64").astype(object).where(depto["cluster"].notna(), None)
    nacional["ymophg_median_rel"] = 1.0

    keys = [i["key"] for i in INDICATORS]
    for ind in INDICATORS:
        ind["years"] = [int(y) for y in nacional.loc[nacional[ind["key"]].notna(), "anio"]]

    panel = {
        "generated": date.today().isoformat(),
        "source": "INE Honduras, Encuesta Permanente de Hogares de Propósitos Múltiples (EPHPM)",
        "years": [int(y) for y in sorted(depto["anio"].unique())],
        "cluster_years": [int(y) for y in sorted(clusters["anio"].unique())],
        "missing_years_note": "El INE no publica microdatos descargables de 2012 a 2017 ni de 2020.",
        "indicators": INDICATORS,
        "model": {
            "algorithm": "KMeans",
            "k": meta["k"],
            "features": meta["features"],
            "silhouette": meta["silhouette"],
            "rows": int(len(clusters)),
        },
        "poverty_model": poverty_block(),
        "clusters": [
            {"id": i, "name": CLUSTER_NAMES[i][0], "description": CLUSTER_NAMES[i][1],
             "profile": {k: v for k, v in prof.items() if k != "cluster"}}
            for i, prof in enumerate(meta["profile"])
        ],
        "departments": [{"code": c, "name": n, "geo": geo_key(n)} for c, n in cfg["depto_names"].items()],
        "national": records(nacional, ["anio"] + keys + ["n_hogares"]),
        "rows": records(depto, ["anio", "depto", "cluster"] + keys + ["n_hogares"]),
    }

    text = json.dumps(panel, ensure_ascii=False, separators=(",", ":"))
    targets = [ROOT / "api_data" / "panel.json"]
    if args.front:
        targets.append(Path(args.front) / "lib" / "panel.json")
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(text, encoding="utf-8")
        print(f"written {t} ({len(text) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
