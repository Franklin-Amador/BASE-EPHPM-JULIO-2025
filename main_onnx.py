"""
FastAPI Server - Clustering Model ONNX (Versión Ligera)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import onnxruntime as rt
import numpy as np
from typing import List, Dict, Union
import os
import dotenv
import requests
import json

#Cargar variables de entorno
dotenv.load_dotenv()

app = FastAPI(
    title="🗺️ Honduras Clustering API (ONNX)",
    description="API ligera para clasificar departamentos - Solo 50MB",
    version="2.0.0 - ONNX + Vercel Blob"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════
# CARGAR MODELOS DIRECTAMENTE DESDE VERCEL BLOB (EN RAM)
# ═══════════════════════════════════════════════════════════════════════

def load_from_url(url: str, description: str = "archivo"):
    """Carga un archivo directamente desde URL a memoria"""
    try:
        print(f"📥 Cargando {description} desde blob...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        print(f"✅ {description} cargado en RAM ({len(response.content)} bytes)")
        return response.content
    except Exception as e:
        raise Exception(f"❌ Error cargando {description} desde {url}: {str(e)}")

# Obtener URLs desde .env
MODEL_BLOB_URL = os.getenv("MODEL_BLOB_URL")
SCALER_PARAMS_BLOB_URL = os.getenv("SCALER_PARAMS_BLOB_URL")
FEATURE_NAMES_BLOB_URL = os.getenv("FEATURE_NAMES_BLOB_URL")

if not all([MODEL_BLOB_URL, SCALER_PARAMS_BLOB_URL, FEATURE_NAMES_BLOB_URL]):
    raise ValueError("❌ Faltan variables de entorno para Vercel Blob (MODEL_BLOB_URL, SCALER_PARAMS_BLOB_URL, FEATURE_NAMES_BLOB_URL)")

print("🔄 Cargando modelos desde Vercel Blob Storage...")

# Cargar modelo ONNX directamente en RAM
model_bytes = load_from_url(MODEL_BLOB_URL, "Modelo ONNX")
sess = rt.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Cargar parámetros del scaler (JSON, sin necesidad de sklearn)
scaler_params_bytes = load_from_url(SCALER_PARAMS_BLOB_URL, "Scaler Params")
scaler_params = json.loads(scaler_params_bytes.decode('utf-8'))
scaler_mean = np.array(scaler_params['mean'])
scaler_scale = np.array(scaler_params['scale'])

# Cargar feature names directamente en RAM
features_bytes = load_from_url(FEATURE_NAMES_BLOB_URL, "Feature Names")
feature_names = features_bytes.decode('utf-8').strip().split(',')

print(f"✅ Modelo ONNX cargado en RAM")
print(f"✅ Scaler params cargado en RAM (sin sklearn)")
print(f"✅ Características: {len(feature_names)}")

# ═══════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class DepartmentData(BaseModel):
    """Datos de un departamento"""
    ymophg_mean: float
    ymophg_median: float
    anosest_mean: float
    edad_mean: float
    totper_mean: float
    tasa_ocupacion: float
    tasa_pobreza: float
    tasa_nbi: float

class ClusterResponse(BaseModel):
    """Respuesta con cluster"""
    cluster: int
    cluster_name: str
    description: str

class ClusterDefinition(BaseModel):
    """Definición de cluster"""
    value: int
    name: str
    description: str

class BatchResponse(BaseModel):
    """Respuesta para lote"""
    total: int
    clusters: List[Dict]
    summary: Dict[str, int]

class BatchRequest(BaseModel):
    """Solicitud para predicción por lote"""
    data: List[DepartmentData]
    n_results: int = Field(default=10, ge=1, le=100)

# Catálogo de clusters para reutilizar en endpoints
CLUSTER_DEFINITIONS = {
    0: {
        "name": "Desarrollo Alto 🟢",
        "description": "Indicadores socioeconómicos altos"
    },
    1: {
        "name": "Desarrollo Medio-Alto 🔵",
        "description": "Indicadores socioeconómicos medio-altos"
    },
    2: {
        "name": "Desarrollo Medio-Bajo 🟠",
        "description": "Indicadores socioeconómicos medio-bajos"
    },
    3: {
        "name": "Desarrollo Bajo 🔴",
        "description": "Indicadores socioeconómicos bajos"
    }
}

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    """Información de la API"""
    return {
        "mensaje": "🗺️ API de Clustering Honduras (ONNX Lightweight)",
        "versión": "2.0.0",
        "modelo": "ONNX + onnxruntime",
        "storage": "Vercel Blob (carga directa en RAM)",
        "tamaño": "~50 MB (95% más ligero)"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "ready": True}

@app.get("/clusters", response_model=List[ClusterDefinition])
def get_clusters():
    """Retorna catálogo de clusters con valor, nombre y definición"""
    return [
        ClusterDefinition(
            value=value,
            name=data["name"],
            description=data["description"]
        )
        for value, data in CLUSTER_DEFINITIONS.items()
    ]

@app.post("/predict", response_model=ClusterResponse)
def predict_cluster(data: DepartmentData):
    """Predice el cluster"""
    try:
        # Preparar datos
        X_input = np.array([[
            data.ymophg_mean, data.ymophg_median, data.anosest_mean, data.edad_mean,
            data.totper_mean, data.tasa_ocupacion, data.tasa_pobreza, data.tasa_nbi
        ]], dtype=np.float32)

        # Normalizar manualmente (sin sklearn)
        X_scaled = ((X_input - scaler_mean) / scaler_scale).astype(np.float32)

        # Predicción con ONNX
        pred_onnx = sess.run([output_name], {input_name: X_scaled})
        cluster = int(pred_onnx[0][0])

        return ClusterResponse(
            cluster=cluster,
            cluster_name=CLUSTER_DEFINITIONS[cluster]["name"],
            description=CLUSTER_DEFINITIONS[cluster]["description"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(payload: Union[BatchRequest, List[DepartmentData]]):
    """Predice para múltiples departamentos"""
    if isinstance(payload, list):
        data_list = payload
        n_results = len(data_list)
    else:
        data_list = payload.data
        n_results = payload.n_results

    if not data_list or len(data_list) > 100:
        raise HTTPException(status_code=400, detail="Entre 1 y 100 registros")

    if n_results > len(data_list):
        raise HTTPException(
            status_code=400,
            detail=(
                f"n_results ({n_results}) no puede ser mayor al número de registros enviados ({len(data_list)})."
            )
        )

    try:
        # Solo se procesan los n_results solicitados por el usuario
        selected_data = data_list[:n_results]

        X_input = np.array([[
            d.ymophg_mean, d.ymophg_median, d.anosest_mean, d.edad_mean,
            d.totper_mean, d.tasa_ocupacion, d.tasa_pobreza, d.tasa_nbi
        ] for d in selected_data], dtype=np.float32)

        # Normalizar manualmente (sin sklearn)
        X_scaled = ((X_input - scaler_mean) / scaler_scale).astype(np.float32)
        pred_onnx = sess.run([output_name], {input_name: X_scaled})
        clusters = pred_onnx[0].flatten().astype(int)

        results = []
        cluster_count = {"0": 0, "1": 0, "2": 0, "3": 0}

        for i, cluster in enumerate(clusters):
            cluster = int(cluster)
            cluster_count[str(cluster)] += 1
            results.append({"index": i, "cluster": cluster})

        return BatchResponse(
            total=n_results,
            clusters=results,
            summary=cluster_count
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/info")
def model_info():
    """Info del modelo"""
    return {
        "modelo": "KMeans (ONNX)",
        "n_clusters": 4,
        "features": feature_names,
        "n_features": len(feature_names),
        "framework": "onnxruntime",
        "storage": "Vercel Blob (RAM only)",
        "memoria": "~50 MB"
    }

