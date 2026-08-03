from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from typing import Optional
import math
import pandas as pd

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
            try:
                import json
                with open("models/xgboost_trade_forecast_meta.json", "r") as f:
                    xgboost_model["meta"] = json.load(f)
            except Exception as e:
                # Do NOT fabricate a metric here. An absent meta file means the
                # metrics are genuinely unknown; returning a hardcoded R² would
                # show the UI a score that no training run actually produced.
                logger.warning(f"Model meta JSON unavailable ({e}); metrics will be empty.")
                xgboost_model["meta"] = {"metrics": {}}
            logger.info("XGBoost model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            xgboost_model = "FAILED"

PARTNER_MAP = {'36': 'Australia', '56': 'Belgium', '156': 'China', '276': 'Germany', '344': 'Hong Kong', '360': 'Indonesia', '368': 'Iraq', '392': 'Japan', '410': 'South Korea', '458': 'Malaysia', '528': 'Netherlands', '643': 'Russia', '682': 'Saudi Arabia', '702': 'Singapore', '704': 'Vietnam', '784': 'UAE', '826': 'UK', '842': 'USA'}
CMD_MAP = {'01': 'Live Animals', '02': 'Meat', '03': 'Fish', '04': 'Dairy', '05': 'Animal Products', '06': 'Trees/Plants', '07': 'Vegetables', '08': 'Fruits/Nuts', '09': 'Coffee/Tea/Spices', '10': 'Cereals', '11': 'Milling Products', '12': 'Oil Seeds', '13': 'Gums/Resins', '14': 'Vegetable Plaiting', '15': 'Fats/Oils', '16': 'Prepared Meat/Fish', '17': 'Sugars', '18': 'Cocoa', '19': 'Cereal Preps', '20': 'Vegetable/Fruit Preps', '21': 'Misc Edibles', '22': 'Beverages', '23': 'Food Waste/Fodder', '24': 'Tobacco', '25': 'Salt/Earths/Stone', '26': 'Ores/Slag/Ash', '27': 'Mineral Fuels', '28': 'Inorganic Chemicals', '29': 'Organic Chemicals', '30': 'Pharmaceuticals', '31': 'Fertilizers', '32': 'Tanning/Dyes', '33': 'Essential Oils/Cosmetics', '34': 'Soap/Waxes', '35': 'Albuminoids', '36': 'Explosives', '37': 'Photographic Goods', '38': 'Misc Chemicals', '39': 'Plastics', '40': 'Rubber', '41': 'Raw Hides/Skins', '42': 'Leather Articles', '43': 'Furskins', '44': 'Wood', '45': 'Cork', '46': 'Straw/Esparto', '47': 'Wood Pulp', '48': 'Paper/Paperboard', '49': 'Printed Books', '50': 'Silk', '51': 'Wool', '52': 'Cotton', '53': 'Vegetable Textile Fibers', '54': 'Man-made Filaments', '55': 'Man-made Staple Fibers', '56': 'Wadding/Felt/Yarn', '57': 'Carpets', '58': 'Special Woven Fabrics', '59': 'Impregnated Fabrics', '60': 'Knitted Fabrics', '61': 'Knitted Apparel', '62': 'Non-knitted Apparel', '63': 'Other Textiles', '64': 'Footwear', '65': 'Headgear', '66': 'Umbrellas', '67': 'Prepared Feathers', '68': 'Stone/Plaster Articles', '69': 'Ceramics', '70': 'Glass', '71': 'Precious Stones/Metals', '72': 'Iron/Steel', '73': 'Articles of Iron/Steel', '74': 'Copper', '75': 'Nickel', '76': 'Aluminum', '78': 'Lead', '79': 'Zinc', '80': 'Tin', '81': 'Other Base Metals', '82': 'Tools/Cutlery', '83': 'Misc Base Metal Articles', '84': 'Nuclear Reactors/Boilers/Machinery', '85': 'Electrical Machinery', '86': 'Railway/Tramway', '87': 'Vehicles', '88': 'Aircraft/Spacecraft', '89': 'Ships/Boats', '90': 'Optical/Medical Instruments', '91': 'Clocks/Watches', '92': 'Musical Instruments', '93': 'Arms/Ammunition', '94': 'Furniture', '95': 'Toys/Sports', '96': 'Misc Manufactured', '97': 'Works of Art', '98': 'Special Classification', '99': 'Special Classification'}

combo_cache = None

@router.get("/valid_combinations")
async def get_valid_combinations():
    global combo_cache
    if combo_cache is not None:
        return combo_cache
        
    try:
        df = pd.read_parquet("data/processed/trade_features.parquet")
        MIN_COMBO_TOTAL_USD = 50_000_000
        MIN_COMBO_YEARS = 3
        
        df['pStr'] = df['partnerCode'].apply(lambda x: str(x).split('.')[0])
        df['cStr'] = df['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))
        
        # Exclude India 356/699
        df = df[~df['pStr'].isin(['356', '699'])]
        
        grouped = df.groupby(['pStr', 'cStr']).agg(
            total_usd=('primaryValue', 'sum'),
            years_count=('period', 'nunique')
        ).reset_index()
        
        valid = grouped[(grouped['total_usd'] >= MIN_COMBO_TOTAL_USD) & (grouped['years_count'] >= MIN_COMBO_YEARS)]
        valid_map = {}
        for _, row in valid.iterrows():
            valid_map.setdefault(row['pStr'], []).append(row['cStr'])
            
        partners_set = set(valid_map.keys())
        partner_totals = df[df['pStr'].isin(partners_set)].groupby('pStr')['primaryValue'].sum().reset_index()
        partner_totals = partner_totals.sort_values(by='primaryValue', ascending=False)
        
        partners = []
        for _, row in partner_totals.iterrows():
            c = row['pStr']
            partners.append({"code": c, "name": PARTNER_MAP.get(c, c)})
            
        combo_cache = {"partners": partners, "map": valid_map}
        return combo_cache
    except Exception as e:
        logger.error(f"Failed to build valid combinations: {e}")
        return {"partners": [], "map": {}}

def get_latest_trade_value(partner_code: str, commodity_code: str) -> float:
    """
    Returns the most recent recorded trade value (USD) for a (partner, commodity)
    pair, or 0.0 when the pair has no history at all.

    Used by the POST / pre-flight check to distinguish "this pair is genuinely
    untraded, refuse to forecast" from "this pair is thin but real". Returning
    0.0 on any failure is deliberate: a missing parquet or an unknown pair should
    both route into the HTTP 400 explanation path, never a 500.
    """
    try:
        df = pd.read_parquet("data/processed/trade_features.parquet")
    except Exception as e:
        logger.warning(f"get_latest_trade_value: could not read trade features: {e}")
        return 0.0

    try:
        p_code = str(partner_code).split('.')[0]
        c_code = str(commodity_code).split('.')[0].zfill(2)

        pStr = df['partnerCode'].apply(lambda x: str(x).split('.')[0])
        cStr = df['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))

        matched = df[(pStr == p_code) & (cStr == c_code)]
        if matched.empty:
            return 0.0

        latest_period = matched['period'].max()
        latest_value = matched[matched['period'] == latest_period]['primaryValue'].sum()

        if pd.isna(latest_value) or not math.isfinite(float(latest_value)):
            return 0.0
        return float(latest_value)
    except Exception as e:
        logger.warning(f"get_latest_trade_value({partner_code}, {commodity_code}) failed: {e}")
        return 0.0

def resolve_partner_code(code: str) -> str:
    c = str(code).split('.')[0]
    if not c.isdigit():
        rev_map = {v.lower(): k for k, v in PARTNER_MAP.items()}
        c = rev_map.get(str(code).lower(), c)
    return c

@router.get("/partner_signature")
async def get_partner_signature(partner_code: str):
    try:
        p_code_str = resolve_partner_code(partner_code)
        df = pd.read_parquet("data/processed/trade_features.parquet")
        df['pStr'] = df['partnerCode'].apply(lambda x: str(x).split('.')[0])
        df['cStr'] = df['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))
        
        df_partner = df[df['pStr'] == p_code_str]
        if df_partner.empty:
            return []
            
        top = df_partner.groupby('cStr')['primaryValue'].sum().reset_index()
        top = top.sort_values(by='primaryValue', ascending=False).head(5)
        
        res = []
        for _, row in top.iterrows():
            c = row['cStr']
            res.append({"code": c, "name": CMD_MAP.get(c, c), "value_billions": float(row['primaryValue']/1e9)})
        return res
    except Exception as e:
        logger.error(f"Failed to fetch partner signature: {e}")
        return []

@router.get("/history")
async def get_global_history(partner_code: Optional[str] = None, commodity_code: Optional[str] = None):
    try:
        df = pd.read_parquet("data/processed/trade_features.parquet")
    except Exception as e:
        # Distinguish "the dataset is missing/unreadable" (a server fault) from
        # "the query matched no rows" (a valid empty result).
        logger.error(f"history: trade features parquet unavailable: {e}")
        raise HTTPException(status_code=503, detail="Historical trade dataset is unavailable.")

    try:
        if partner_code:
            df = df[df['partnerCode'].astype(str).str.split('.').str[0] == str(partner_code).split('.')[0]]
        if commodity_code:
            df = df[df['cmdCode'].astype(str).str.split('.').str[0].str.zfill(2) == str(commodity_code).split('.')[0].zfill(2)]
            
        if df.empty:
            return {"history": []}
            
        df = df[df['flowCode'] == 'X']
            
        yearly_vol = df.groupby("period")["primaryValue"].sum().reset_index().sort_values(by="period")
        recent = yearly_vol.tail(4)
        return {"history": [{"year": str(int(r["period"])), "value": float(r["primaryValue"] / 1e9)} for _, r in recent.iterrows()]}
    except Exception as e:
        logger.error(f"history: failed to aggregate: {e}")
        return {"history": []}

MIN_DATA_YEAR = 1990
MAX_DATA_YEAR = 2100
VALID_GROUP_BY = ("partner", "commodity")

@router.get("/year_breakdown")
async def get_year_breakdown(year: int, group_by: str, partner_code: Optional[str] = None, commodity_code: Optional[str] = None):
    # Reject bad input explicitly instead of returning [] — an empty list is
    # indistinguishable from "valid query, no matching rows".
    if not (MIN_DATA_YEAR <= year <= MAX_DATA_YEAR):
        raise HTTPException(
            status_code=422,
            detail=f"year must be between {MIN_DATA_YEAR} and {MAX_DATA_YEAR}; got {year}."
        )
    if group_by not in VALID_GROUP_BY:
        raise HTTPException(
            status_code=422,
            detail=f"group_by must be one of {list(VALID_GROUP_BY)}; got {group_by!r}."
        )

    try:
        df = pd.read_parquet("data/processed/trade_features.parquet")
        df = df[df['period'] == year]
        if partner_code:
            p_code_str = resolve_partner_code(partner_code)
            df = df[df['partnerCode'].astype(str).str.split('.').str[0] == p_code_str]
        if commodity_code:
            df = df[df['cmdCode'].astype(str).str.split('.').str[0].str.zfill(2) == str(commodity_code).split('.')[0].zfill(2)]
            
        if df.empty: return []
        
        df['pStr'] = df['partnerCode'].apply(lambda x: str(x).split('.')[0])
        df['cStr'] = df['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))
            
        if group_by == 'partner':
            group_col, map_dict = 'pStr', PARTNER_MAP
        else:
            group_col, map_dict = 'cStr', CMD_MAP
            
        agg = df.groupby(group_col)['primaryValue'].sum().reset_index()
        agg = agg.sort_values(by='primaryValue', ascending=False).head(10)
        
        res = []
        for _, row in agg.iterrows():
            c = row[group_col]
            res.append({"code": c, "name": map_dict.get(c, c), "value_billions": float(row['primaryValue']/1e9)})
        return res
    except Exception as e:
        logger.error(f"Failed to fetch year breakdown: {e}")
        return []

@router.get("/country_series")
async def get_country_series(partner_code: str):
    try:
        p_code_str = resolve_partner_code(partner_code)
        df = pd.read_parquet("data/processed/trade_features.parquet")
        df['pStr'] = df['partnerCode'].apply(lambda x: str(x).split('.')[0])
        df['cStr'] = df['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))
        
        df = df[df['pStr'] == p_code_str]
        if df.empty:
            return {"yearly": [], "top_commodities": []}
            
        flow_col = next((c for c in df.columns if c.lower() in ['flowcode', 'tradeflow', 'flow'] or set(df[c].dropna().unique()).issubset({'M','X','Import','Export'})), None)
        
        yearly = df.groupby("period").apply(
            lambda x: pd.Series({
                "value_billions": float(x["primaryValue"].sum()/1e9),
                "import_billions": float(x[x[flow_col].isin(['M', 'Import'])]['primaryValue'].sum()/1e9) if flow_col else None,
                "export_billions": float(x[x[flow_col].isin(['X', 'Export'])]['primaryValue'].sum()/1e9) if flow_col else None
            })
        ).reset_index().sort_values(by="period")
        
        yearly_res = []
        for _, r in yearly.iterrows():
            item = {"year": str(int(r["period"])), "value_billions": r["value_billions"]}
            if flow_col and pd.notna(r["import_billions"]):
                item["import_billions"] = r["import_billions"]
                item["export_billions"] = r["export_billions"]
            yearly_res.append(item)
            
        top = df.groupby("cStr")["primaryValue"].sum().reset_index().sort_values(by="primaryValue", ascending=False).head(5)
        top_cmds = [{"code": r["cStr"], "name": CMD_MAP.get(r["cStr"], r["cStr"]), "value_billions": float(r["primaryValue"]/1e9)} for _, r in top.iterrows()]
        
        return {"yearly": yearly_res, "top_commodities": top_cmds}
    except Exception as e:
        logger.error(f"country_series err: {e}")
        return {"yearly": [], "top_commodities": []}

@router.post("/")
async def get_forecast(req: ForecastRequest):
    load_model()
    if xgboost_model == "FAILED":
        raise HTTPException(status_code=503, detail="Forecast model is unavailable.")

    # Pre-flight check G1
    global combo_cache
    if combo_cache is None:
        await get_valid_combinations()
        
    p_code = str(req.partner_code).split('.')[0]
    c_code = str(req.commodity_code).split('.')[0].zfill(2)
    
    if p_code not in combo_cache.get("map", {}) or c_code not in combo_cache["map"].get(p_code, []):
        suggested = []
        if p_code in combo_cache.get("map", {}):
            valid_cmds = combo_cache["map"][p_code]
            try:
                df = pd.read_parquet("data/processed/trade_features.parquet")
                df['pStr'] = df['partnerCode'].apply(lambda x: str(x).split('.')[0])
                df['cStr'] = df['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))
                df_filt = df[(df['pStr'] == p_code) & (df['cStr'].isin(valid_cmds))]
                top_for_partner = df_filt.groupby('cStr')['primaryValue'].sum().sort_values(ascending=False).head(3)
                suggested = [{"code": c, "name": CMD_MAP.get(c, c)} for c in top_for_partner.index]
            except Exception as e:
                logger.warning(f"Could not build commodity suggestions for partner {p_code}: {e}")
        
        latest_val = get_latest_trade_value(p_code, c_code)
        if latest_val < 10000:
            p_name = PARTNER_MAP.get(p_code, p_code)
            c_name = CMD_MAP.get(c_code, c_code)
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"India-{p_name} has no meaningful historical trade in {c_name}; forecast unreliable here.",
                    "suggested_commodities": suggested
                }
            )

    try:
        import numpy as np

        expected_features = xgboost_model['features']
        input_data = {feat: 0.0 for feat in expected_features}

        df_hist = pd.read_parquet("data/processed/trade_features.parquet")
        df_hist['pStr'] = df_hist['partnerCode'].apply(lambda x: str(x).split('.')[0])
        df_hist['cStr'] = df_hist['cmdCode'].apply(lambda x: str(x).split('.')[0].zfill(2))
        
        df_filtered = df_hist[(df_hist['pStr'] == p_code) & (df_hist['cStr'] == c_code)]
        if df_filtered.empty:
            raise HTTPException(status_code=400, detail="No historical data for this combination, try another")
            
        latest_row = df_filtered.sort_values(by="period").iloc[-1]
        
        for feat in expected_features:
            if feat in latest_row:
                val = latest_row[feat]
                if pd.notna(val):
                    try:
                        input_data[feat] = float(val)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Feature '{feat}' has non-numeric value {val!r}; defaulting to 0.0"
                        )

        if "usdinr_mean" in input_data: input_data["usdinr_mean"] = float(req.usd_inr)
        if "brent_crude_mean" in input_data: input_data["brent_crude_mean"] = float(req.crude_price)
        if "period" in input_data: input_data["period"] = float(req.year)

        df = pd.DataFrame([input_data])
        prediction_log = xgboost_model['model'].predict(df)[0]
        actual_prediction_usd = float(np.expm1(prediction_log))
        
        if not math.isfinite(actual_prediction_usd):
            raise HTTPException(status_code=500, detail="Model returned an invalid prediction")

        try:
            importances = xgboost_model['model'].feature_importances_
            feat_imp = sorted(zip(expected_features, importances), key=lambda x: x[1], reverse=True)[:5]
            feature_importance = [{"feature": f, "importance": float(i)} for f, i in feat_imp]
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            feature_importance = []

        return {
            "year": req.year,
            "forecasted_trade_value_usd": actual_prediction_usd,
            "forecasted_trade_value_billions": actual_prediction_usd / 1e9,
            "feature_importance": feature_importance,
            "metrics": xgboost_model.get("meta", {}).get("metrics", {}),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
