"""
IndiTrade AI — India's Trade Intelligence Platform API.

FastAPI application entrypoint integrating all 4 core intelligence modules:
- Module A: TradeFlow Forecast (/forecast)
- Module B: PolicyGPT RAG (/query)
- Module C: TradeNet Graph (/network)
- Module D: AnomalyGuard (/anomaly)

Includes UptimeRobot /health ping endpoint to prevent Render free tier sleep (512MB RAM limit).
"""

import os
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.backend.api.forecast import router as forecast_router
from src.backend.api.query import router as query_router
from src.backend.api.network import router as network_router
from src.backend.api.anomaly import router as anomaly_router
from src.backend.database import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("IndiTradeAPI")

# Initialize FastAPI App
app = FastAPI(
    title="IndiTrade AI API",
    description="End-to-end India Trade Intelligence Platform processing 20+ years of trade flows, macro indicators, and DGFT policy regulations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Vercel domain e.g. ["https://inditrade-ai.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Module Routers
app.include_router(forecast_router, tags=["Module A: TradeFlow Forecast"])
app.include_router(query_router, tags=["Module B: PolicyGPT RAG"])
app.include_router(network_router, tags=["Module C: TradeNet Graph"])
app.include_router(anomaly_router, tags=["Module D: AnomalyGuard"])


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures API latency and logs requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.get("/", tags=["System"])
async def root():
    """Root welcome endpoint with sitemap."""
    return {
        "platform": "IndiTrade AI",
        "tagline": "India's Trade Intelligence Platform ($0 Infrastructure)",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "forecast": "POST /forecast",
            "query": "POST /query",
            "network": "GET /network",
            "anomaly": "GET /anomaly"
        }
    }


@app.get("/health", tags=["System"])
async def health_check():
    """
    UptimeRobot Keep-Alive Endpoint.
    Ping this endpoint every 10 minutes via UptimeRobot (free tier) to prevent Render 512MB instance from sleeping!
    """
    return {
        "status": "ONLINE",
        "timestamp": time.time(),
        "memory_optimization": "ACTIVE (ONNX + Lazy Loading)",
        "database_connections": {
            "qdrant_vector_db": "ONLINE" if db_manager.qdrant else "FALLBACK_LOCAL",
            "supabase_relational_db": "ONLINE" if db_manager.supabase else "FALLBACK_LOCAL"
        }
    }


if __name__ == "__main__":
    import uvicorn
    # Run locally on port 8000
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
