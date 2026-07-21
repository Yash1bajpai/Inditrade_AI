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

@router.get("/historical")
async def get_historical_anomalies():
    try:
        import pandas as pd
        import os
        filepath = "data/processed/flagged_trade_anomalies.csv"
        if not os.path.exists(filepath):
            return {"data": []}
            
        df = pd.read_csv(filepath)
        # Select top 50 anomalies for plotting to avoid overwhelming the UI
        df = df.head(50)
        # Sort by period so the X-axis is chronological
        if "period" in df.columns:
            df = df.sort_values(by="period")
        
        import math
        data = []
        for _, row in df.iterrows():
            val = float(row.get("primaryValue", 0))
            mean_val = float(row.get("primaryValue_rolling_3y_mean", 1))
            
            # Fix Zero-Division and NaN Bug
            if pd.isna(mean_val) or math.isnan(mean_val) or mean_val == 0:
                mean_val = 1e-9
                
            deviation_pct = ((val - mean_val) / mean_val) * 100
            
            cmd_desc = row.get("cmdDesc", "Unknown")
            if pd.isna(cmd_desc) or str(cmd_desc).lower() == "nan":
                cmd_desc = "Unknown"
                
            data.append({
                "date": str(row.get("period", "Unknown")),
                "value": val,
                "partner": str(row.get("partnerDesc", "Unknown")),
                "commodity": str(cmd_desc),
                "reason": f"Value deviated {deviation_pct:.1f}% from 3yr mean"
            })
        return {"data": data}
    except Exception as e:
        logger.error(f"Error fetching historical anomalies: {e}")
        return {"error": str(e)}
