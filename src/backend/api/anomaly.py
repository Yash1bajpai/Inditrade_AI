from fastapi import APIRouter
from pydantic import BaseModel
import logging

logger = logging.getLogger("api.anomaly")
router = APIRouter()

class AnomalyRequest(BaseModel):
    usd_inr: float
    crude_price: float

anomaly_model = None

def load_model():
    global anomaly_model
    if anomaly_model is None:
        try:
            import joblib
            logger.info("Lazy-loading Isolation Forest anomaly model...")
            anomaly_model = joblib.load("models/isolation_forest_anomalies.pkl")
            logger.info("Anomaly model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Anomaly model: {e}")
            anomaly_model = "FAILED"

@router.post("/")
async def detect_anomaly(req: AnomalyRequest):
    load_model()
    if anomaly_model == "FAILED":
        return {"error": "Anomaly model is unavailable."}
        
    try:
        import pandas as pd
        df = pd.DataFrame([{
            "INR=X": req.usd_inr,
            "CL=F": req.crude_price,
            "lag_1y_export": 0,
            "lag_1y_import": 0,
            "rolling_3y_mean_export": 0,
            "policy_event_flag": 0
        }])
        
        # Isolation Forest returns -1 for anomaly, 1 for normal
        prediction = anomaly_model['model'].predict(df)[0]
        is_anomaly = bool(prediction == -1)
        
        return {
            "is_anomaly": is_anomaly,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return {"error": str(e)}
