from fastapi import APIRouter
from pydantic import BaseModel
import logging

logger = logging.getLogger("api.forecast")
router = APIRouter()

class ForecastRequest(BaseModel):
    usd_inr: float
    crude_price: float
    year: int

# Lazy loaded model placeholder
xgboost_model = None

def load_model():
    global xgboost_model
    if xgboost_model is None:
        try:
            import joblib
            logger.info("Lazy-loading XGBoost forecast model...")
            xgboost_model = joblib.load("models/xgboost_trade_forecast.pkl")
            logger.info("XGBoost model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            xgboost_model = "FAILED"

@router.post("/")
async def get_forecast(req: ForecastRequest):
    load_model()
    
    if xgboost_model == "FAILED":
        return {"error": "Forecast model is unavailable."}
        
    try:
        import pandas as pd
        # Construct input DataFrame matching training features
        # Note: In a real scenario, you'd calculate lags/rolling means based on historical data.
        # Here we do a simplified inference for the demo.
        df = pd.DataFrame([{
            "INR=X": req.usd_inr,
            "CL=F": req.crude_price,
            "year": req.year,
            # Fill other features with median/zeros for demo stability
            "lag_1y_export": 0,
            "lag_1y_import": 0,
            "rolling_3y_mean_export": 0,
            "policy_event_flag": 0
        }])
        
        prediction = xgboost_model['model'].predict(df)[0]
        return {
            "year": req.year,
            "forecasted_trade_value_usd": float(prediction),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
