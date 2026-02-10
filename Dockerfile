FROM python:3.13-slim

WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY fastapi_app_onnx.py .
COPY models/ ./models/

# Instalar dependencias (sin cache para reducir tamaño)
RUN pip install --no-cache-dir -r requirements.txt

# Expose puerto
EXPOSE 8000

# Comando por defecto
CMD ["uvicorn", "fastapi_app_onnx:app", "--host", "0.0.0.0", "--port", "8000"]
