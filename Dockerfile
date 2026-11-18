FROM python:3.13-slim

WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY main_onnx.py .

# Instalar dependencias (sin cache para reducir tamaño)
RUN pip install --no-cache-dir -r requirements.txt

# Expose puerto
EXPOSE 8000

# Comando por defecto
CMD ["uvicorn", "main_onnx:app", "--host", "0.0.0.0", "--port", "8000"]
