from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from src.utils.data_cache import load_csv


logger = logging.getLogger("api.anomaly")
router = APIRouter()

class AnomalyRequest(BaseModel):
    usd_inr: float
    crude_price: float

anomaly_model = None

# The six features the Isolation Forest was trained on, in order. Declared here
# so the loaded pickle can be validated at load time rather than blowing up
# inside predict() with an opaque feature-count error.
ANOMALY_FEATURES = [
    "brent_crude_yoy_pct",
    "primaryValue_yoy_growth_rate",
    "unit_value",
    "value_vs_3y_mean",
    "wgt_vs_3y_mean",
    "policy_event_flag",
]

def load_model():
    global anomaly_model
    if anomaly_model is None:
        try:
            import joblib
            logger.info("Lazy-loading Isolation Forest anomaly model...")
            loaded = joblib.load("models/isolation_forest_anomalies.pkl")

            # Validate the schema up front. The request handler indexes
            # loaded['model'], so a bare estimator pickle would raise TypeError
            # on every request instead of failing once, loudly, here.
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"expected a dict-wrapped pickle with a 'model' key, got {type(loaded).__name__}"
                )
            if "model" not in loaded:
                raise KeyError("pickle dict is missing the 'model' key")
            if not hasattr(loaded["model"], "predict"):
                raise TypeError("loaded['model'] has no predict() method")

            n_expected = getattr(loaded["model"], "n_features_in_", None)
            if n_expected is not None and n_expected != len(ANOMALY_FEATURES):
                raise ValueError(
                    f"model expects {n_expected} features but the API supplies "
                    f"{len(ANOMALY_FEATURES)}: {ANOMALY_FEATURES}"
                )

            anomaly_model = loaded
            logger.info("Anomaly model loaded and schema-validated successfully.")
        except Exception as e:
            logger.error(f"Failed to load Anomaly model: {e}")
            anomaly_model = "FAILED"

@router.post("/")
def detect_anomaly(req: AnomalyRequest):
    """Hypothetical scenario tester (USD/INR + crude only)."""
    load_model()
    if anomaly_model == "FAILED":
        raise HTTPException(status_code=503, detail="Anomaly model is unavailable.")

    try:
        import pandas as pd
        # Use the exact features the Isolation Forest model was trained on
        # Since we only get usd_inr and crude_price from frontend, we use dummy/derived values
        # Compute dynamic heuristics based on the two inputs
        base_crude = 80.0
        crude_yoy = ((req.crude_price - base_crude) / base_crude) * 100
        val_3y = -0.2 if req.crude_price > 100 else 0.05
        policy_flag = 1 if req.usd_inr > 85 else 0

        feature_values = {
            "brent_crude_yoy_pct": crude_yoy,
            "primaryValue_yoy_growth_rate": val_3y * 100,
            "unit_value": req.crude_price * 10,
            "value_vs_3y_mean": val_3y,
            "wgt_vs_3y_mean": val_3y,
            "policy_event_flag": policy_flag,
        }

        # The pickle records the training-time feature order in its 'features'
        # key; sklearn requires predict() frames to match that exact order.
        # The hardcoded ANOMALY_FEATURES list is only a fallback.
        feature_order = list(anomaly_model.get("features") or ANOMALY_FEATURES)
        df = pd.DataFrame(
            [[feature_values.get(f, 0.0) for f in feature_order]],
            columns=feature_order,
        )

        prediction = anomaly_model['model'].predict(df)[0]
        is_anomaly = bool(prediction == -1)

        return {
            "is_anomaly": is_anomaly,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during anomaly detection.")

@router.get("/historical")
def get_historical_anomalies():
    try:
        import pandas as pd
        import os
        filepath = "data/processed/flagged_trade_anomalies.csv"
        if not os.path.exists(filepath):
            return {"data": []}

        df = load_csv(filepath)

        if "anomaly_score" in df.columns:
            df = df.sort_values(by="anomaly_score", ascending=False)

        df = df.head(50)

        if "period" in df.columns:
            # Removed period sort here to preserve severity order for table
            pass

        import math
        data = []
        for _, row in df.iterrows():
            val = float(row.get("primaryValue", 0))
            mean_val = row.get("primaryValue_rolling_3y_mean", 1)
            score = float(row.get("anomaly_score", -1))

            if pd.isna(mean_val) or math.isnan(float(mean_val)) or float(mean_val) == 0:
                deviation_pct = 0
                reason_str = "Value deviated (No historical 3yr data)"
                reason_code = "no_baseline"
            else:
                mean_val = float(mean_val)
                deviation_pct = ((val - mean_val) / mean_val) * 100
                reason_str = f"Value deviated {deviation_pct:+.1f}% from 3yr mean"
                reason_code = "deviation"

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
                "reason_code": reason_code,
                "anomaly_score": score
            })
        
        table_data = data
        chart_data = sorted(data, key=lambda x: str(x.get("date", "")))
        
        return {"chart_data": chart_data, "table_data": table_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical anomalies: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching anomaly history.")

