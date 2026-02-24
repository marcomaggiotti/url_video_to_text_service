FROM python:3.12-slim

WORKDIR /app

# ✅ FFmpeg per Whisper audio
RUN apt-get update && apt-get install -y **ffmpeg** && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p downloads

# Copy app
COPY main.py .
COPY tools/ ./tools/
#COPY downloads/ ./downloads/
# Lazy models → NO download build time
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]