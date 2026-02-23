FROM python:3.14-slim

# Metadata
LABEL maintainer="https://github.com/marcomaggiotti"

WORKDIR /app

# 1️⃣ Cache pip + NO wheels problematici
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# 2️⃣ JIT + Free-threaded GIL (3.14!)
ENV PYTHON_JIT_ENABLED=1
ENV PYTHON_GIL=0
# Free-threaded mode!

# Copy source
COPY . .

# Healthcheck (Railway loves)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Non-root user (security)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]