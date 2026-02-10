# 🗺️ API de Clustering Socioeconómico - Honduras

Modelo de machine learning desplegado en FastAPI para clasificar departamentos hondureños según sus indicadores socioeconómicos.

## 📋 Contenido

### Modelos Entrenados (carpeta `models/`)

- `clustering_pipeline.pkl` - Pipeline completo (StandardScaler + KMeans)
- `kmeans.pkl` - Modelo KMeans con 4 clusters
- `scaler.pkl` - StandardScaler para normalizar datos
- `feature_names.txt` - Listado de características esperadas

### Código

- `fastapi_app.py` - Servidor FastAPI (API)
- `clustering_client.py` - Cliente Python para consumir la API
- `etl_hogar.ipynb` - Notebook con análisis completo

## 🚀 Instalación y Ejecución

### Paso 1: Instalar dependencias

```bash
pip install fastapi uvicorn joblib numpy pandas scikit-learn
```

### Paso 2: Ejecutar el servidor

```bash
# Opción A: Con auto-reload (desarrollo)
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

# Opción B: Sin auto-reload (producción)
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

### Paso 3: Acceder a la API

**Documentación interactiva:**

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

**Health check:**

- <http://localhost:8000/health>

## 📊 Endpoints

### 1. GET `/`

Información general de la API
```bash
#!/bin/bash
curl http://localhost:8000/
```

### 2. GET `/health`

Verifica que el servidor está operativo
```bash
#!/bin/bash
curl http://localhost:8000/health
```

### 3. GET `/info`

Obtiene información del modelo
```bash
curl http://localhost:8000/info
```

### 4. POST `/predict`

Predice el cluster para un departamento

**Request:**
```json
{
  "ymophg_mean": 8500.5,
  "ymophg_median": 7200.0,
  "anosest_mean": 6.5,
  "edad_mean": 35.2,
  "totper_mean": 4.1,
  "tasa_ocupacion": 0.65,
  "tasa_pobreza": 0.45,
  "tasa_nbi": 0.38
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ymophg_mean": 8500.5,
    "ymophg_median": 7200.0,
    "anosest_mean": 6.5,
    "edad_mean": 35.2,
    "totper_mean": 4.1,
    "tasa_ocupacion": 0.65,
    "tasa_pobreza": 0.45,
    "tasa_nbi": 0.38
  }'
```

**Response:**
```json
{
  "cluster": 1,
  "cluster_name": "Desarrollo Medio-Alto 🔵",
  "confidence": 0.87,
  "description": "Departamento con indicadores socioeconómicos medio-altos"
}
```

### 5. POST `/predict-batch`

Predice clusters para múltiples departamentos (máximo 100)

**Request:**
```json
[
  {
    "ymophg_mean": 9000,
    "ymophg_median": 8000,
    "anosest_mean": 7,
    "edad_mean": 34,
    "totper_mean": 3.8,
    "tasa_ocupacion": 0.7,
    "tasa_pobreza": 0.4,
    "tasa_nbi": 0.3
  },
  {
    "ymophg_mean": 5000,
    "ymophg_median": 4000,
    "anosest_mean": 4,
    "edad_mean": 37,
    "totper_mean": 4.5,
    "tasa_ocupacion": 0.5,
    "tasa_pobreza": 0.65,
    "tasa_nbi": 0.55
  }
]
```

**Response:**
``` json
{
  "total": 2,
  "clusters": [
    {"index": 0, "cluster": 0},
    {"index": 1, "cluster": 3}
  ],
  "summary": {
    "0": 1,
    "1": 0,
    "2": 0,
    "3": 1
  }
}
```

## 🐍 Cliente Python

Usar el cliente `clustering_client.py`:

```bash
python clustering_client.py
```

O en tu código Python:

```python
from clustering_client import ClusteringAPIClient

# Crear cliente
client = ClusteringAPIClient("http://localhost:8000")

# Verificar conexión
if client.health_check():
    print("✅ Conectado")
    
    # Predicción simple
    resultado = client.predict(
        ymophg_mean=8500,
        ymophg_median=7500,
        anosest_mean=6.5,
        edad_mean=35,
        totper_mean=4,
        tasa_ocupacion=0.65,
        tasa_pobreza=0.45,
        tasa_nbi=0.35
    )
    print(f"Cluster: {resultado['cluster_name']}")
else:
    print("❌ No se puede conectar")
```

## 🎯 Clusters de Desarrollo

| Cluster | Nombre | Descripción |
| ------- | ------ | ----------- |
| 0 | Desarrollo Alto 🟢 | Indicadores socioeconómicos altos |
| 1 | Desarrollo Medio-Alto 🔵 | Indicadores socioeconómicos medio-altos |
| 2 | Desarrollo Medio-Bajo 🟠 | Indicadores socioeconómicos medio-bajos |
| 3 | Desarrollo Bajo 🔴 | Indicadores socioeconómicos bajos |

## 📈 Características del Modelo

**Tipo:** Unsupervised Learning (K-Means Clustering)

- **Clusters:** 4 grupos
- **Normalización:** StandardScaler

**Características (8):**

1. `ymophg_mean` - Ingreso promedio (Lempiras)
2. `ymophg_median` - Ingreso mediano (Lempiras)
3. `anosest_mean` - Años de educación promedio
4. `edad_mean` - Edad promedio
5. `totper_mean` - Personas por hogar promedio
6. `tasa_ocupacion` - Tasa de ocupación (0-1)
7. `tasa_pobreza` - Tasa de pobreza (0-1)
8. `tasa_nbi` - Tasa de Necesidades Básicas Insatisfechas (0-1)

## 🔧 Deployar en Producción

### Opción 1: Gunicorn + Uvicorn (Recomendado)

```bash
pip install gunicorn
gunicorn fastapi_app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Opción 2: Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t clustering-api .
docker run -p 8000:8000 clustering-api
```

### Opción 3: Railway, Heroku, etc

El archivo `fastapi_app.py` es compatible con plataformas serverless.

## 📝 Notas

- El modelo espera exactamente 8 características en el orden definido
- Los valores de tasa deben estar entre 0 y 1
- La confianza es una métrica inversa de la distancia al centroide del cluster
- El servidor se auto-documenta en `/docs` y `/redoc`

## 📞 Soporte

Para más información, revisar:

- `etl_hogar.ipynb` - Análisis completo del modelo
- `fastapi_app.py` - Documentación del código
- <http://localhost:8000/docs> - Documentación de la API
