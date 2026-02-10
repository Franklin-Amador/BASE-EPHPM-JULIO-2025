# 🚀 Deployment ONNX - Guía Completa

## 📊 Comparación: PKL vs ONNX

``` text
┌─────────────────────┬──────────────┬──────────────┐
│ Componente          │ PKL Version  │ ONNX Version │
├─────────────────────┼──────────────┼──────────────┤
│ scikit-learn        │ ~150 MB      │ ❌ No need   │
│ onnxruntime         │ ❌ No need   │ ~45 MB       │
│ joblib              │ ~5 MB        │ ~5 MB        │
│ modelo              │ ~50 KB       │ ~50 KB       │
├─────────────────────┼──────────────┼──────────────┤
│ TOTAL               │ ~155 MB      │ ~50 MB       │
│ Reducción           │ -----        │ 68% menor    │
│ Free tier?          │ ❌ NO        │ ✅ SÍ        │
└─────────────────────┴──────────────┴──────────────┘
```

---

## 🔧 Instalación Local ONNX

### Opción 1: Solo dependencias ONNX (recomendado)

```bash
pip install fastapi uvicorn onnxruntime joblib numpy pandas
```

### Opción 2: Desde requirements.txt

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar API ONNX

### Terminal 1: Iniciar servidor

```bash
cd "BASE EPHPM JULIO 2025"
uvicorn fastapi_app_onnx:app --reload --port 8000
```

### Terminal 2: Probar (PowerShell)

```powershell
# Health check
curl http://localhost:8000/health

# Predicción
$body = @{
    ymophg_mean = 8500
    ymophg_median = 7500
    anosest_mean = 6.5
    edad_mean = 35
    totper_mean = 4
    tasa_ocupacion = 0.65
    tasa_pobreza = 0.45
    tasa_nbi = 0.35
} | ConvertTo-Json

curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -Body $body
```

### Terminal 3: Documentación

Abre en navegador: **<http://localhost:8000/docs>**

---

## 🌐 Deployment a Render.com (Free Tier)

### Archivos necesarios

**1. requirements.txt** (ONNX version)

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
onnxruntime==1.16.3
joblib==1.3.2
numpy==1.24.3
pandas==2.1.3
```

### 2. Procfile

``` text
web: uvicorn fastapi_app_onnx:app --host 0.0.0.0 --port $PORT
```

### 3. runtime.txt (opcional)

``` text
python-3.9.18
```

### 4. .gitignore

``` text
venv/
__pycache__/
*.pyc
.DS_Store
.env
```

### Deploy en Render

1. **GitHub**: Push tu código a GitHub

   ``` text

  git init
  git add .
  git commit -m "ONNX deployment ready"
  git branch -M main
  git remote add origin <https://github.com/tu-usuario/tu-repo.git>
  git push -u origin main

   ```

2. **Render.com**:
   - Ir a <https://render.com>
   - Sign up (gratis)
   - New → Web Service
   - Conectar GitHub repo
   - Settings:
     - **Name**: honduras-clustering-api
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn fastapi_app_onnx:app --host 0.0.0.0 --port $PORT`

3. **Deploy**:
   - Click "Create Web Service"
   - Espera ~3 minutos
   - Tu API estará en: `https://honduras-clustering-api.onrender.com`

---

## ✅ Test Remoto

```bash
# Health check
curl https://honduras-clustering-api.onrender.com/health

# Docs
https://honduras-clustering-api.onrender.com/docs

# Predicción
curl -X POST https://honduras-clustering-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ymophg_mean": 8500,
    "ymophg_median": 7500,
    "anosest_mean": 6.5,
    "edad_mean": 35,
    "totper_mean": 4,
    "tasa_ocupacion": 0.65,
    "tasa_pobreza": 0.45,
    "tasa_nbi": 0.35
  }'
```

---

## 🐳 Alternativa: Docker

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copiar archivos
COPY requirements.txt .
COPY fastapi_app_onnx.py .
COPY models/ ./models/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando
CMD ["uvicorn", "fastapi_app_onnx:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Construir y ejecutar

```bash
# Construir
docker build -t clustering-api-onnx .

# Ejecutar localmente
docker run -p 8000:8000 clustering-api-onnx

# Push a Docker Hub
docker tag clustering-api-onnx:latest tu-usuario/clustering-api-onnx:latest
docker push tu-usuario/clustering-api-onnx:latest
```

### Usar en Render (con Docker)

- Conectar repo GitHub
- Render detecta Dockerfile
- Deploy automático

---

## 📦 Archivos del Proyecto

``` text
BASE EPHPM JULIO 2025/
├── fastapi_app_onnx.py          # ⭐ Servidor ONNX (ligero)
├── fastapi_app.py               # Servidor PKL (pesado)
├── clustering_client.py          # Cliente Python
├── requirements.txt              # Dependencias ONNX
├── models/
│   ├── clustering_model.onnx     # ⭐ Modelo ONNX (50 KB)
│   ├── clustering_pipeline.pkl   # Modelo PKL full
│   ├── scaler.pkl                # Normalizador
│   └── feature_names.txt         # Columnas esperadas
├── data/
│   ├── gadm41_HND_1.json         # Mapa Honduras
│   └── ...
├── etl_hogar.ipynb               # Notebook análisis
└── README.md                     # Este archivo
```

---

## 🚨 Troubleshooting

### Error: "No module named 'onnxruntime'"

```bash
pip install onnxruntime
```

### Error: "ONNX model is corrupted"

Regenerar el ONNX desde notebook:

```python
# En etl_hogar.ipynb, ejecutar celda de conversión ONNX
```

### API lenta en free tier

- Render sleep después de 15 min inactividad
- Primera llamada reactiva el servidor (~10 seg)
- Llamadas posteriores son rápidas

### Modelo no encuentra "feature_names.txt"

```bash
# Verificar que existe
ls models/

# Si falta, regenerar desde notebook
```

---

## 🎯 Performance Esperado

| Métrica | ONNX | PKL |
| --------- | ------ | ------- |
| Inferencia | ~5 ms | ~10 ms |
| Tamaño | 50 MB | 155 MB |
| Memoria RAM | ~80 MB | ~200 MB |
| Startup | ~2 seg | ~5 seg |
| Free tier? | ✅ SÍ | ❌ NO |

---

## 📞 Próximos Pasos

1. ✅ Probar ONNX localmente
2. ✅ Crear cuenta en Render.com
3. ✅ Hacer push a GitHub
4. ✅ Conectar repo en Render
5. ✅ Deploy automático
6. ✅ Compartir URL pública

---

## 💡 Referencias

- [ONNX Runtime Quickstart](https://onnxruntime.ai/docs/get-started/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Render Docs](https://render.com/docs)
- [GitHub Actions CI/CD](https://github.com/features/actions)
