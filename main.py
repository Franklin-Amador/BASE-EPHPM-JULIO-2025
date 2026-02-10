
"""
FastAPI Server - Clustering Model para Datos Socioeconómicos Honduras
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict
import os

app = FastAPI(
    title="🗺️ Honduras Socioeconomic Clustering API",
    description="API para clasificar departamentos hondureños en clusters de desarrollo",
    version="1.0.0"
)

# Cargar modelo y scaler al iniciar
MODEL_PATH = "models/clustering_pipeline.pkl"
SCALER_PATH = "models/scaler.pkl"
KMEANS_PATH = "models/kmeans.pkl"
FEATURES_PATH = "models/feature_names.txt"

# Validar que los archivos existen
if not all(os.path.exists(p) for p in [SCALER_PATH, KMEANS_PATH, FEATURES_PATH]):
    raise FileNotFoundError("❌ Modelos no encontrados. Ejecuta la celda anterior primero.")

# Cargar modelos
scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)

# Cargar nombres de características
with open(FEATURES_PATH, 'r') as f:
    feature_names = f.read().strip().split(',')

print(f"✅ Modelo cargado con {kmeans.n_clusters} clusters")
print(f"✅ Características esperadas: {feature_names}")

# ═══════════════════════════════════════════════════════════════════════
# MODELOS DE DATOS (Schemas)
# ═══════════════════════════════════════════════════════════════════════

class DepartmentData(BaseModel):
    """Datos de un departamento para clasificación"""
    ymophg_mean: float
    ymophg_median: float
    anosest_mean: float
    edad_mean: float
    totper_mean: float
    tasa_ocupacion: float
    tasa_pobreza: float
    tasa_nbi: float

    class Config:
        json_schema_extra = {
            "example": {
                "ymophg_mean": 8500.5,
                "ymophg_median": 7200.0,
                "anosest_mean": 6.5,
                "edad_mean": 35.2,
                "totper_mean": 4.1,
                "tasa_ocupacion": 0.65,
                "tasa_pobreza": 0.45,
                "tasa_nbi": 0.38
            }
        }

class ClusterResponse(BaseModel):
    """Respuesta con asignación de cluster"""
    cluster: int
    cluster_name: str
    confidence: float
    description: str

class BatchResponse(BaseModel):
    """Respuesta para predicción en lote"""
    total: int
    clusters: List[Dict]
    summary: Dict[str, int]

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    """Endpoint raíz - información de la API"""
    return {
        "mensaje": "🗺️ API de Clustering Socioeconómico de Honduras",
        "versión": "1.0.0",
        "modelo": "KMeans (4 clusters)",
        "características": len(feature_names),
        "documentación": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check del servidor"""
    return {"status": "healthy", "ready": True}

@app.post("/predict", response_model=ClusterResponse)
def predict_cluster(data: DepartmentData):
    """
    Predice el cluster de desarrollo para datos departamentales.

    - **ymophg_mean**: Ingreso promedio (Lempiras)
    - **ymophg_median**: Ingreso mediano (Lempiras)
    - **anosest_mean**: Años de educación promedio
    - **edad_mean**: Edad promedio
    - **totper_mean**: Personas por hogar promedio
    - **tasa_ocupacion**: Tasa de ocupación (0-1)
    - **tasa_pobreza**: Tasa de pobreza (0-1)
    - **tasa_nbi**: Tasa de NBI (0-1)

    Retorna:
    - **cluster**: Número de cluster (0, 1, 2, 3)
    - **cluster_name**: Nombre descriptivo del cluster
    - **confidence**: Distancia al centroide (menor = más confiable)
    """
    try:
        # Preparar datos en el orden correcto
        X_input = np.array([[
            data.ymophg_mean,
            data.ymophg_median,
            data.anosest_mean,
            data.edad_mean,
            data.totper_mean,
            data.tasa_ocupacion,
            data.tasa_pobreza,
            data.tasa_nbi
        ]])

        # Escalar (convertir a DataFrame para preservar nombres de características)
        X_df = pd.DataFrame(X_input, columns=feature_names)
        X_scaled = scaler.transform(X_df)

        # Predecir cluster
        cluster = int(kmeans.predict(X_scaled)[0])

        # Calcular distancia al centroide (confianza)
        distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_, axis=1)
        distance = float(distances[0])
        confidence = 1 / (1 + distance)  # Convertir a confianza (0-1)

        # Nombres de clusters
        cluster_names = {
            0: "Desarrollo Alto 🟢",
            1: "Desarrollo Medio-Alto 🔵",
            2: "Desarrollo Medio-Bajo 🟠",
            3: "Desarrollo Bajo 🔴"
        }

        descriptions = {
            0: "Departamento con indicadores socioeconómicos altos",
            1: "Departamento con indicadores socioeconómicos medio-altos",
            2: "Departamento con indicadores socioeconómicos medio-bajos",
            3: "Departamento con indicadores socioeconómicos bajos"
        }

        return ClusterResponse(
            cluster=cluster,
            cluster_name=cluster_names[cluster],
            confidence=confidence,
            description=descriptions[cluster]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(data_list: List[DepartmentData]):
    """
    Predice clusters para múltiples departamentos en lote.
    """
    if not data_list or len(data_list) > 100:
        raise HTTPException(status_code=400, detail="Proporcionar entre 1 y 100 registros")

    try:
        # Preparar matriz de entrada
        X_input = np.array([[
            d.ymophg_mean, d.ymophg_median, d.anosest_mean, d.edad_mean,
            d.totper_mean, d.tasa_ocupacion, d.tasa_pobreza, d.tasa_nbi
        ] for d in data_list])

        # Escalar (convertir a DataFrame para preservar nombres de características)
        X_df = pd.DataFrame(X_input, columns=feature_names)
        X_scaled = scaler.transform(X_df)

        # Predecir clusters
        clusters = kmeans.predict(X_scaled)

        # Preparar respuesta
        results = []
        cluster_count = {0: 0, 1: 0, 2: 0, 3: 0}

        for i, cluster in enumerate(clusters):
            cluster = int(cluster)
            cluster_count[cluster] += 1
            results.append({"index": i, "cluster": cluster})

        return BatchResponse(
            total=len(data_list),
            clusters=results,
            summary=cluster_count
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción por lote: {str(e)}")

@app.get("/info")
def model_info():
    """Información del modelo"""
    return {
        "modelo": "KMeans Clustering",
        "n_clusters": kmeans.n_clusters,
        "features": feature_names,
        "n_features": len(feature_names),
        "iter": kmeans.n_iter_,
        "inertia": float(kmeans.inertia_)
    }

# Para ejecutar: uvicorn app:app --reload
