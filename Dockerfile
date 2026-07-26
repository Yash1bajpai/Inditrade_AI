# Use official Python lightweight image
FROM python:3.10-slim

# Install system dependencies needed for downloading and extracting data
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-backend.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-backend.txt

# Download data and models from GitHub Release
ARG DATA_RELEASE_URL=https://github.com/Yash1bajpai/Inditrade_AI/releases/download/data-v1/inditrade-data.zip
RUN curl -L ${DATA_RELEASE_URL} -o /tmp/data.zip && \
    unzip -o /tmp/data.zip -d /tmp/data_extracted && \
    rm /tmp/data.zip && \
    mkdir -p /app/data/processed /app/models && \
    mv /tmp/data_extracted/*.parquet /app/data/processed/ 2>/dev/null || true && \
    mv /tmp/data_extracted/*.jsonl /app/data/processed/ 2>/dev/null || true && \
    mv /tmp/data_extracted/*.pkl /app/models/ 2>/dev/null || true && \
    mv /tmp/data_extracted/*.onnx /app/models/ 2>/dev/null || true && \
    mv /tmp/data_extracted/*.json /app/models/ 2>/dev/null || true

# Copy application source code
COPY src/ src/
COPY config/ config/

# Ensure necessary directories exist
RUN mkdir -p data/raw logs

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
