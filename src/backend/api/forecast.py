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
        # Initialize all features to 0
        expected_features = xgboost_model['features']
        input_data = {feat: 0.0 for feat in expected_features}
        
        # Override with user inputs where applicable
        if "usdinr_mean" in input_data:
            input_data["usdinr_mean"] = float(req.usd_inr)
        if "brent_crude_mean" in input_data:
            input_data["brent_crude_mean"] = float(req.crude_price)
        if "period" in input_data:
            input_data["period"] = float(req.year)
            
        df = pd.DataFrame([input_data])
        
        prediction = xgboost_model['model'].predict(df)[0]
        
        # Extract feature importances
        try:
            importances = xgboost_model['model'].feature_importances_
            # Zip and sort top 5
            feat_imp = sorted(zip(expected_features, importances), key=lambda x: x[1], reverse=True)[:5]
            feature_importance = [{"feature": f, "importance": float(i)} for f, i in feat_imp]
        except:
            feature_importance = []
            
        return {
            "year": req.year,
            "forecasted_trade_value_usd": float(prediction),
            "feature_importance": feature_importance,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
