from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.backend.api import forecast, query, anomaly, network

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="IndiTrade AI Backend API",
    description="Backend services for Trade Forecasting, Qualitative RAG, and Anomaly Detection",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(query.router, prefix="/api/query", tags=["Query (LLM)"])
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["Anomaly"])
app.include_router(network.router, prefix="/api/network", tags=["Network"])

@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint for UptimeRobot monitoring"""
    return {"status": "ok", "service": "IndiTrade AI Backend"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting IndiTrade AI Backend...")
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
