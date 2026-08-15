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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://inditrade.vercel.app", "https://inditrade-ai.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(query.router, prefix="/api/query", tags=["Query (LLM)"])
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["Anomaly"])
app.include_router(network.router, prefix="/api/network", tags=["Network"])

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Health"])
def health_check():
    """Endpoint for UptimeRobot monitoring and readiness probe"""
    import os
    from src.backend.database import qdrant
    
    status = "ok"
    details = []
    
    if qdrant is None:
        status = "degraded"
        details.append("Qdrant not initialized")
        
    if not os.path.exists("data/processed/trade_features.parquet"):
        status = "degraded"
        details.append("Trade features data missing")
        
    if not os.path.exists("models/xgboost_trade_forecast.pkl"):
        status = "degraded"
        details.append("Forecast model missing")
        
    return {"status": status, "service": "IndiTrade AI Backend", "details": details}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting IndiTrade AI Backend...")
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)

