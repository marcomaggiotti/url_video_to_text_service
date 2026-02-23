FROM python:3.12-slim

WORKDIR /app

# Solo ESSENZIALI (no torch pesante!)
COPY requirements.txt .
RUN pip install --no-cache-dir --quiet \
    fastapi \
    uvicorn[standard] \
    pydantic \
    slowapi \
    numpy \
    && pip cache purge

# Copy app
COPY main.py .
COPY tools/ ./tools/
COPY downloads/ ./downloads/
# Lazy models → NO download build time
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]