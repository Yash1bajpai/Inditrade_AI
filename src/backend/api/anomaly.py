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

        if "anomaly_score" in df.columns:
            df = df.sort_values(by="anomaly_score", ascending=False)

        df = df.head(50)

        if "period" in df.columns:
            df = df.sort_values(by="period")

        import math
        data = []
        for _, row in df.iterrows():
            val = float(row.get("primaryValue", 0))
            mean_val = row.get("primaryValue_rolling_3y_mean", 1)
            score = float(row.get("anomaly_score", -1))

            if pd.isna(mean_val) or math.isnan(float(mean_val)) or float(mean_val) == 0:
                deviation_pct = 0
                reason_str = "Value deviated (No historical 3yr data)"
            else:
                mean_val = float(mean_val)
                deviation_pct = ((val - mean_val) / mean_val) * 100
                reason_str = f"Value deviated {deviation_pct:+.1f}% from 3yr mean"

            cmd_desc = row.get("cmdDesc")
            cmd_code = row.get("cmdCode", "XX")
            if pd.isna(cmd_desc) or str(cmd_desc).lower() == "nan" or str(cmd_desc).strip() == "" or str(cmd_desc) == "None":

                hs_map = {
                    "84": "Machinery & Mechanical Appliances",
                    "85": "Electrical Machinery & Electronics",
                    "90": "Optical, Photographic & Medical Instruments",
                    "39": "Plastics & Articles Thereof",
                    "73": "Articles of Iron or Steel",
                    "27": "Mineral Fuels & Oils",
                    "29": "Organic Chemicals",
                    "30": "Pharmaceutical Products",
                    "87": "Vehicles & Parts",
                    "71": "Precious Metals & Stones"
                }
                cmd_desc = hs_map.get(str(cmd_code).zfill(2), f"HS Code {cmd_code}")

            data.append({
                "date": str(row.get("period", "Unknown")),
                "value": val,
                "partner": str(row.get("partnerDesc", "Unknown")),
                "commodity": str(cmd_desc),
                "reason": reason_str,
                "anomaly_score": score
            })
        return {"data": data}
    except Exception as e:
        logger.error(f"Error fetching historical anomalies: {e}")
        return {"error": str(e)}

