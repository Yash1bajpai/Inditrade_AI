from fastapi import APIRouter
from pydantic import BaseModel
import logging

logger = logging.getLogger("api.forecast")
router = APIRouter()

class ForecastRequest(BaseModel):
    usd_inr: float
    crude_price: float
    year: int
    partner_code: str
    commodity_code: str

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

from typing import Optional
import math

@router.get("/history")
async def get_global_history(partner_code: Optional[str] = None, commodity_code: Optional[str] = None):
    """
    Returns the real global trade volume history (last 3-5 years)
    for the XGBoost chart before the prediction point.
    """
    try:
        import pandas as pd
        df = pd.read_parquet("data/processed/trade_features.parquet")
        
        if partner_code is not None:
            df = df[df['partnerCode'].astype(str) == str(partner_code)]
        if commodity_code is not None:
            df = df[df['cmdCode'].astype(str) == str(commodity_code)]
            
        if df.empty:
            return {"history": []}
            
        yearly_vol = df.groupby("period")["primaryValue"].sum().reset_index()
        yearly_vol = yearly_vol.sort_values(by="period")

        recent = yearly_vol.tail(4)

        history = []
        for _, row in recent.iterrows():
            history.append({
                "year": str(int(row["period"])),
                "value": float(row["primaryValue"] / 1e9)
            })
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to fetch global history: {e}")
        return {"history": []}

PARTNER_MAP = {'36': 'Australia', '56': 'Belgium', '156': 'China', '276': 'Germany', '344': 'Hong Kong', '360': 'Indonesia', '368': 'Iraq', '392': 'Japan', '410': 'South Korea', '458': 'Malaysia', '528': 'Netherlands', '643': 'Russia', '682': 'Saudi Arabia', '702': 'Singapore', '704': 'Vietnam', '784': 'UAE', '826': 'UK', '842': 'USA'}
CMD_MAP = {'01': 'Live Animals', '02': 'Meat', '03': 'Fish', '04': 'Dairy', '05': 'Animal Products', '06': 'Trees/Plants', '07': 'Vegetables', '08': 'Fruits/Nuts', '09': 'Coffee/Tea/Spices', '10': 'Cereals', '11': 'Milling Products', '12': 'Oil Seeds', '13': 'Gums/Resins', '14': 'Vegetable Plaiting', '15': 'Fats/Oils', '16': 'Prepared Meat/Fish', '17': 'Sugars', '18': 'Cocoa', '19': 'Cereal Preps', '20': 'Vegetable/Fruit Preps', '21': 'Misc Edibles', '22': 'Beverages', '23': 'Food Waste/Fodder', '24': 'Tobacco', '25': 'Salt/Earths/Stone', '26': 'Ores/Slag/Ash', '27': 'Mineral Fuels', '28': 'Inorganic Chemicals', '29': 'Organic Chemicals', '30': 'Pharmaceuticals', '31': 'Fertilizers', '32': 'Tanning/Dyes', '33': 'Essential Oils/Cosmetics', '34': 'Soap/Waxes', '35': 'Albuminoids', '36': 'Explosives', '37': 'Photographic Goods', '38': 'Misc Chemicals', '39': 'Plastics', '40': 'Rubber', '41': 'Raw Hides/Skins', '42': 'Leather Articles', '43': 'Furskins', '44': 'Wood', '45': 'Cork', '46': 'Straw/Esparto', '47': 'Wood Pulp', '48': 'Paper/Paperboard', '49': 'Printed Books', '50': 'Silk', '51': 'Wool', '52': 'Cotton', '53': 'Vegetable Textile Fibers', '54': 'Man-made Filaments', '55': 'Man-made Staple Fibers', '56': 'Wadding/Felt/Yarn', '57': 'Carpets', '58': 'Special Woven Fabrics', '59': 'Impregnated Fabrics', '60': 'Knitted Fabrics', '61': 'Knitted Apparel', '62': 'Non-knitted Apparel', '63': 'Other Textiles', '64': 'Footwear', '65': 'Headgear', '66': 'Umbrellas', '67': 'Prepared Feathers', '68': 'Stone/Plaster Articles', '69': 'Ceramics', '70': 'Glass', '71': 'Precious Stones/Metals', '72': 'Iron/Steel', '73': 'Articles of Iron/Steel', '74': 'Copper', '75': 'Nickel', '76': 'Aluminum', '78': 'Lead', '79': 'Zinc', '80': 'Tin', '81': 'Other Base Metals', '82': 'Tools/Cutlery', '83': 'Misc Base Metal Articles', '84': 'Nuclear Reactors/Boilers/Machinery', '85': 'Electrical Machinery', '86': 'Railway/Tramway', '87': 'Vehicles', '88': 'Aircraft/Spacecraft', '89': 'Ships/Boats', '90': 'Optical/Medical Instruments', '91': 'Clocks/Watches', '92': 'Musical Instruments', '93': 'Arms/Ammunition', '94': 'Furniture', '95': 'Toys/Sports', '96': 'Misc Manufactured', '97': 'Works of Art', '99': 'Special Classification'}

@router.get("/year_breakdown")
async def get_year_breakdown(year: int, group_by: str, partner_code: Optional[str] = None, commodity_code: Optional[str] = None):
    try:
        import pandas as pd
        df = pd.read_parquet("data/processed/trade_features.parquet")
        df = df[df['period'] == year]
        
        if partner_code:
            df = df[df['partnerCode'].astype(str) == str(partner_code)]
        if commodity_code:
            df = df[df['cmdCode'].astype(str) == str(commodity_code)]
            
        if df.empty:
            return []
            
        if group_by == 'partner':
            group_col = 'partnerCode'
            map_dict = PARTNER_MAP
        elif group_by == 'commodity':
            group_col = 'cmdCode'
            map_dict = CMD_MAP
        else:
            return []
            
        grouped = df.groupby(group_col)['primaryValue'].sum().reset_index()
        top = grouped.sort_values(by='primaryValue', ascending=False).head(10)
        
        res = []
        for _, row in top.iterrows():
            code = str(row[group_col]).split('.')[0]
            # Ensure two digits for HS codes if length is 1
            if group_by == 'commodity' and len(code) == 1:
                code = "0" + code
            res.append({"code": code, "name": map_dict.get(code, code), "value_billions": float(row["primaryValue"]/1e9)})
        
        return res
    except Exception as e:
        logger.error(f"Failed to fetch year breakdown: {e}")
        return []

@router.get("/country_series")
async def get_country_series(partner_code: str):
    try:
        import pandas as pd
        df = pd.read_parquet("data/processed/trade_features.parquet")
        df = df[df['partnerCode'].astype(str) == str(partner_code)]
        if df.empty:
            return {"yearly": [], "top_commodities": []}
            
        yearly = df.groupby("period")["primaryValue"].sum().reset_index()
        yearly = yearly.sort_values(by="period")
        yearly_res = [{"year": str(int(row["period"])), "value_billions": float(row["primaryValue"]/1e9)} for _, row in yearly.iterrows()]
        
        top_cmds = df.groupby("cmdCode")["primaryValue"].sum().reset_index()
        top_cmds = top_cmds.sort_values(by="primaryValue", ascending=False).head(5)
        top_cmds_res = []
        for _, row in top_cmds.iterrows():
            code = str(row["cmdCode"]).split('.')[0]
            if len(code) == 1:
                code = "0" + code
            top_cmds_res.append({"code": code, "name": CMD_MAP.get(code, code), "value_billions": float(row["primaryValue"]/1e9)})
        
        return {"yearly": yearly_res, "top_commodities": top_cmds_res}
    except Exception as e:
        logger.error(f"Failed to fetch country series: {e}")
        return {"yearly": [], "top_commodities": []}

@router.post("/")
async def get_forecast(req: ForecastRequest):
    load_model()

    if xgboost_model == "FAILED":
        return {"error": "Forecast model is unavailable."}

    try:
        import pandas as pd
        import numpy as np

        expected_features = xgboost_model['features']
        input_data = {feat: 0.0 for feat in expected_features}

        df_hist = pd.read_parquet("data/processed/trade_features.parquet")
        df_hist['partnerCode'] = df_hist['partnerCode'].astype(str)
        df_hist['cmdCode'] = df_hist['cmdCode'].astype(str)
        
        df_filtered = df_hist[(df_hist['partnerCode'] == str(req.partner_code)) & (df_hist['cmdCode'] == str(req.commodity_code))]
        
        if df_filtered.empty:
            return {"error": "No historical data for this combination, try another"}
            
        latest_row = df_filtered.sort_values(by="period").iloc[-1]
        
        for feat in expected_features:
            if feat in latest_row:
                val = latest_row[feat]
                if pd.notna(val):
                    try:
                        input_data[feat] = float(val)
                    except (ValueError, TypeError):
                        pass

        if "usdinr_mean" in input_data:
            input_data["usdinr_mean"] = float(req.usd_inr)
        if "brent_crude_mean" in input_data:
            input_data["brent_crude_mean"] = float(req.crude_price)
        if "period" in input_data:
            input_data["period"] = float(req.year)

        df = pd.DataFrame([input_data])
        prediction_log = xgboost_model['model'].predict(df)[0]
        actual_prediction_usd = float(np.expm1(prediction_log))
        
        if not math.isfinite(actual_prediction_usd):
            return {"error": "Model returned an invalid prediction"}

        try:
            importances = xgboost_model['model'].feature_importances_
            feat_imp = sorted(zip(expected_features, importances), key=lambda x: x[1], reverse=True)[:5]
            feature_importance = [{"feature": f, "importance": float(i)} for f, i in feat_imp]
        except:
            feature_importance = []

        return {
            "year": req.year,
            "forecasted_trade_value_usd": actual_prediction_usd,
            "forecasted_trade_value_billions": actual_prediction_usd / 1e9,
            "feature_importance": feature_importance,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


