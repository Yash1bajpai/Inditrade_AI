# Use official Python lightweight image
FROM python:3.12-slim

# Install system dependencies needed for downloading and extracting data
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-backend.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-backend.txt

# Download data and models from GitHub Release.
# curl -f fails on HTTP errors and `set -eu` aborts the build, so a broken
# release can no longer produce a "successful" image with no data inside.
# Per-type mv stays tolerant (the zip may legitimately lack some extensions),
# but the image must contain at least one data or model artifact to pass.
ARG DATA_RELEASE_URL=https://github.com/Yash1bajpai/Inditrade_AI/releases/download/data-v1/inditrade-data.zip
RUN set -eu; \
    curl -fL --retry 3 "${DATA_RELEASE_URL}" -o /tmp/data.zip; \
    unzip -oq /tmp/data.zip -d /tmp/data_extracted; \
    rm /tmp/data.zip; \
    mkdir -p /app/data/processed /app/models; \
    mv /tmp/data_extracted/*.parquet /app/data/processed/ 2>/dev/null || true; \
    mv /tmp/data_extracted/*.jsonl /app/data/processed/ 2>/dev/null || true; \
    mv /tmp/data_extracted/*.pkl /app/models/ 2>/dev/null || true; \
    mv /tmp/data_extracted/*.onnx /app/models/ 2>/dev/null || true; \
    mv /tmp/data_extracted/*.json /app/models/ 2>/dev/null || true; \
    if [ -z "$(ls -A /app/data/processed)" ] && [ -z "$(ls -A /app/models)" ]; then \
        echo "FATAL: release archive contained no data or model artifacts" >&2; \
        exit 1; \
    fi

# Copy application source code and tracked data files
COPY src/ src/
COPY config/ config/
COPY data/processed/flagged_trade_anomalies.csv /app/data/processed/
COPY data/processed/node2vec_trade_embeddings.parquet /app/data/processed/
# Compressed BM25 policy index: powers offline sparse retrieval when the
# Qdrant cluster is unreachable (see src/rag/sparse_index.py).
COPY data/cache/bm25_index.pkl.lzma /app/data/cache/

# Ensure necessary directories exist
RUN mkdir -p data/raw logs && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
