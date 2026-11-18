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
