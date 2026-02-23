FROM python:3.14-slim

# Metadata
LABEL maintainer="https://github.com/marcomaggiotti"

WORKDIR /app

# 1️⃣ Cache pip + NO wheels problematici
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Runtime (leggero)
FROM python:3.14-alpine
# 5x più piccolo!
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# Cleanup
RUN apk del --no-cache gcc musl-dev && \
    find /root/.local -type f -name '*.pyc' -delete

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#