"""
FastAPI server - EPHPM Honduras: indicator panel by department and year,
plus the KMeans model (ONNX) that assigns a living-conditions level.

The three artifacts are loaded into memory from Vercel Blob at startup:
  MODEL_BLOB_URL       clustering_panel.onnx  StandardScaler + KMeans (input is NOT scaled by hand)
  MODEL_META_BLOB_URL  clustering_panel.json  feature order and cluster profiles
  PANEL_BLOB_URL       panel.json             indicators per year and department

For local development, when those variables are not all set, the same files are read
from models/ and api_data/ (both produced by src/clustering.py and src/export_panel.py
and kept out of git).
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import numpy as np
import onnxruntime as rt
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

dotenv.load_dotenv()

BASE = Path(__file__).resolve().parent

ARTIFACTS = {
    "MODEL_BLOB_URL": BASE / "models" / "clustering_panel.onnx",
    "MODEL_META_BLOB_URL": BASE / "models" / "clustering_panel.json",
    "PANEL_BLOB_URL": BASE / "api_data" / "panel.json",
}


def load_artifacts() -> Dict[str, bytes]:
    """Fetch the three artifacts from Blob, or from local files when Blob is not configured."""
    urls = {name: os.getenv(name) for name in ARTIFACTS}
    if all(urls.values()):
        out = {}
        for name, url in urls.items():
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            out[name] = response.content
            print(f"{name}: {len(response.content)} bytes desde Blob")
        return out
    missing = [name for name, url in urls.items() if not url]
    print(f"AVISO: faltan {missing}; usando archivos locales (solo para desarrollo)")
    try:
        return {name: path.read_bytes() for name, path in ARTIFACTS.items()}
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Sin Blob configurado ({missing}) y sin archivo local {exc.filename}. "
            "Define las tres variables *_BLOB_URL."
        ) from exc


artifacts = load_artifacts()

app = FastAPI(
    title="EPHPM Honduras API",
    description="Indicadores socioeconómicos por departamento y año, y nivel de condiciones de vida (KMeans, ONNX).",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PANEL = json.loads(artifacts["PANEL_BLOB_URL"].decode("utf-8"))
MODEL_META = json.loads(artifacts["MODEL_META_BLOB_URL"].decode("utf-8"))
FEATURES: List[str] = MODEL_META["features"]

sess = rt.InferenceSession(artifacts["MODEL_BLOB_URL"], providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_output = sess.get_outputs()[0].name

# Guards against pointing MODEL_BLOB_URL at a model from another version (e.g. the old 8-feature one).
n_inputs = sess.get_inputs()[0].shape[-1]
if n_inputs != len(FEATURES):
    raise RuntimeError(f"El modelo espera {n_inputs} variables y el metadato declara {len(FEATURES)}.")

CLUSTERS = {c["id"]: c for c in PANEL["clusters"]}
DEPARTMENTS = {d["code"]: d for d in PANEL["departments"]}


class Indicators(BaseModel):
    """The six indicators the model uses. Rates are fractions between 0 and 1."""
    tasa_pobreza: float = Field(ge=0, le=1)
    tasa_pobreza_extrema: float = Field(ge=0, le=1)
    ymophg_median_rel: float = Field(gt=0, le=5, description="Ingreso mediano / mediana nacional del año")
    tasa_ocupacion: float = Field(ge=0, le=1)
    tasa_desempleo: float = Field(ge=0, le=1)
    tasa_dependencia: float = Field(ge=0, le=3)


class ClusterResponse(BaseModel):
    cluster: int
    cluster_name: str
    description: str


class BatchResponse(BaseModel):
    total: int
    clusters: List[Dict]
    summary: Dict[str, int]


def predict(items: List[Indicators]) -> List[int]:
    X = np.array([[getattr(d, f) for f in FEATURES] for d in items], dtype=np.float32)
    return [int(c) for c in sess.run([label_output], {input_name: X})[0].flatten()]


@app.get("/")
def home():
    return {
        "api": "EPHPM Honduras",
        "version": app.version,
        "fuente": PANEL["source"],
        "años": PANEL["years"],
        "endpoints": ["/years", "/indicators", "/clusters", "/national",
                      "/departments?year=", "/departments/{code}/series", "/predict", "/info"],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "ready": True}


@app.get("/years")
def years():
    """Survey years in the panel, and the subset that has a cluster."""
    return {"years": PANEL["years"], "cluster_years": PANEL["cluster_years"],
            "note": PANEL["missing_years_note"]}


@app.get("/indicators")
def indicators():
    return PANEL["indicators"]


@app.get("/clusters")
def clusters():
    return PANEL["clusters"]


@app.get("/national")
def national():
    """National series, one record per survey year."""
    return PANEL["national"]


@app.get("/departments")
def departments(year: Optional[int] = Query(default=None)):
    """Indicators and cluster of every department for one year (latest by default)."""
    year = year or PANEL["years"][-1]
    if year not in PANEL["years"]:
        raise HTTPException(status_code=404, detail=f"Sin datos para {year}. Años: {PANEL['years']}")
    return [{**r, "name": DEPARTMENTS[r["depto"]]["name"]} for r in PANEL["rows"] if r["anio"] == year]


@app.get("/departments/{code}/series")
def department_series(code: int):
    if code not in DEPARTMENTS:
        raise HTTPException(status_code=404, detail="Código de departamento desconocido (1-18)")
    return {"department": DEPARTMENTS[code],
            "series": [r for r in PANEL["rows"] if r["depto"] == code]}


@app.post("/predict", response_model=ClusterResponse)
def predict_cluster(data: Indicators):
    cluster = predict([data])[0]
    return ClusterResponse(cluster=cluster, cluster_name=CLUSTERS[cluster]["name"],
                           description=CLUSTERS[cluster]["description"])


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(payload: List[Indicators]):
    if not payload or len(payload) > 200:
        raise HTTPException(status_code=400, detail="Entre 1 y 200 registros")
    labels = predict(payload)
    summary = {str(c): labels.count(c) for c in CLUSTERS}
    return BatchResponse(total=len(labels),
                         clusters=[{"index": i, "cluster": c} for i, c in enumerate(labels)],
                         summary=summary)


@app.get("/info")
def model_info():
    return {**PANEL["model"], "framework": "onnxruntime", "generated": PANEL["generated"]}
