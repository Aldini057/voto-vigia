FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código, plantillas, modelo y datos procesados
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8000

# Usar el puerto que asigne Railway ($PORT) o 8000 por defecto
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
