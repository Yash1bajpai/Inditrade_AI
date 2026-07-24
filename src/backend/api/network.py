from fastapi import APIRouter
import logging
import os

logger = logging.getLogger("api.network")
router = APIRouter()

network_data = None

def load_network_data():
    global network_data
    if network_data is None:
        try:
            import pandas as pd
            filepath = "data/processed/node2vec_trade_embeddings.parquet"
            if not os.path.exists(filepath):
                logger.warning("Node2Vec embeddings not found.")
                network_data = []
                return

            df = pd.read_parquet(filepath)

            if 'node_type' in df.columns:
                df = df[df['node_type'] == 'partner'].copy()

            try:
                anomaly_df = pd.read_csv("data/processed/flagged_trade_anomalies.csv")
                vol_map = anomaly_df.groupby("partnerDesc")["primaryValue"].sum().to_dict()
            except Exception:
                vol_map = {}

            import numpy as np
            has_embeddings = 'embedding_vector' in df.columns
            if has_embeddings:
                vectors = np.stack(df['embedding_vector'].values)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(vectors)
                df['x'] = coords[:, 0]
                df['y'] = coords[:, 1]
            else:
                df['x'] = [None] * len(df)
                df['y'] = [None] * len(df)

            topo_map = {
                "USA": "United States of America",
                "Russian Federation": "Russia",
                "UK": "United Kingdom",
                "South Korea": "South Korea",
                "Hong Kong": "Hong Kong"
            }

            nodes = []
            for _, row in df.iterrows():
                country = str(row.get('node_desc', 'Unknown'))

                mapped_country = topo_map.get(country, country)

                if country in vol_map:
                    vol = float(vol_map[country])
                elif 'trade_value_usd' in row and not pd.isna(row['trade_value_usd']):
                    vol = float(row['trade_value_usd'])
                else:
                    vol = None

                x_val = float(row['x']) if not pd.isna(row['x']) else None
                y_val = float(row['y']) if not pd.isna(row['y']) else None
                
                nodes.append({
                    "id": str(row.get('node_id', country)),
                    "country_name": mapped_country,
                    "original_country": country,
                    "trade_volume": vol,
                    "x": x_val,
                    "y": y_val,
                    "val": max(10, min(100, vol / 1000000000)) if vol is not None else None
                })

            network_data = nodes
            logger.info("Node2Vec network data loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Node2Vec data: {e}")
            network_data = []

@router.get("/")
async def get_network():
    load_network_data()
    return {"nodes": network_data}

@router.get("/history/{country}")
async def get_country_history(country: str):
    """
    Returns the real 10-year historical trade volume for a specific country
    to populate the Drill-Down Modal in the UI.
    """
    try:
        import pandas as pd

        df = pd.read_parquet("data/processed/trade_features.parquet")

        country_df = df[df['partnerDesc'] == country]
        if country_df.empty:
            return {"history": []}

        yearly_vol = country_df.groupby("period")["primaryValue"].sum().reset_index()
        yearly_vol = yearly_vol.sort_values(by="period")

        history = []
        for _, row in yearly_vol.iterrows():
            history.append({
                "period": str(int(row["period"])),
                "volume": float(row["primaryValue"] / 1e9)
            })

        latest_year = country_df['period'].max()
        latest_df = country_df[country_df['period'] == latest_year]
        top_domains = latest_df.groupby("cmdDesc")["primaryValue"].sum().reset_index()
        top_domains = top_domains.sort_values(by="primaryValue", ascending=False).head(10)
        
        domains = []
        for _, row in top_domains.iterrows():
            domains.append({
                "name": str(row["cmdDesc"]),
                "value": float(row["primaryValue"] / 1e9)
            })

        return {"history": history, "latest_year": int(latest_year), "domains": domains}
    except Exception as e:
        logger.error(f"Failed to fetch history for {country}: {e}")
        return {"error": str(e), "history": []}

