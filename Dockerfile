FROM python:3.12-alpine AS builder

WORKDIR /app
COPY requirements.txt .
# NO torch full → CPU minimal
RUN apk add --no-cache gcc musl-dev linux-headers && \
    pip install --no-cache-dir --compile --user \
    fastapi uvicorn[standard] pydantic slowapi && \
    apk del gcc musl-dev linux-headers

FROM python:3.12-alpine
WORKDIR /app

# Copy SOLO runtime
COPY --from=builder /root/.local /root/.local
COPY main.py .

# NO models download in build → lazy load runtime
RUN apk add --no-cache curl tini && \
    adduser -D appuser && chown -R appuser /app
USER appuser

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]