FROM python:3.12-slim

# Evitar generación de archivos .pyc y asegurar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de Python con caché optimizado
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Exponer el puerto estándar de FastAPI
EXPOSE 8000

# Comando de arranque con python -m uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]