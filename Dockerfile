FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crea dir + user
RUN mkdir -p /app/output /app/qdrant_storage && \
    useradd -m appuser && \
    chown -R appuser:appuser /app /tmp

USER appuser

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]