"""
Feature Engineering Module for IndiTrade AI.

Merges UN Comtrade trade flows, RBI macroeconomic indicators, and policy event flags.
Generates lag features (1y, 3y, 5y), rolling statistics, growth rates, and volatility
for XGBoost forecasting (Module A) and Isolation Forest anomaly detection (Module D).
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TradeFeatureEngineer")


class TradeFeatureEngineer:
    """Creates time-series lags, rolling means, and macroeconomic interaction features."""

    def __init__(self, 
                 raw_trade_dir: str = "data/raw/un_comtrade",
                 raw_rbi_dir: str = "data/raw/rbi",
                 processed_dir: str = "data/processed"):
        self.raw_trade_dir = Path(raw_trade_dir)
        self.raw_rbi_dir = Path(raw_rbi_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_or_generate_synthetic_trade_data(self) -> pd.DataFrame:
        """
        Loads downloaded UN Comtrade parquet files if available.
        Otherwise generates realistic synthetic historical trade flows (2005-2024)
        for India across top 10 HS 2-digit commodities and top 20 partner countries
        so feature engineering and model training can run out-of-the-box.
        """
        parquet_files = list(self.raw_trade_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading real UN Comtrade data from {parquet_files[0]}...")
            return pd.read_parquet(parquet_files[0])
            
        logger.warning("No raw Comtrade Parquet found. Generating realistic synthetic Indian trade baseline (2005-2024)...")
        years = list(range(2005, 2025))
        partners = ["USA", "UAE", "China", "Russia", "Saudi Arabia", "Singapore", "Germany", 
                    "UK", "Australia", "Japan", "South Korea", "France", "Netherlands", 
                    "Indonesia", "Malaysia", "Vietnam", "Brazil", "South Africa", "Italy", "Bangladesh"]
        commodities = [
            ("27", "Mineral fuels and oils"),
            ("71", "Precious stones and gold"),
            ("85", "Electrical machinery and electronics"),
            ("84", "Mechanical appliances and boilers"),
            ("30", "Pharmaceutical products"),
            ("29", "Organic chemicals"),
            ("10", "Cereals"),
            ("72", "Iron and steel"),
            ("87", "Vehicles and automotive parts"),
            ("39", "Plastics and articles thereof")
        ]
        
        records = []
        for year in years:
            for partner in partners:
                for cmd_code, cmd_desc in commodities:
                    # Base trade volume in USD (varies by commodity and partner)
                    base_val = np.random.lognormal(mean=19.5, sigma=1.2)  # ~300M to ~10B USD
                    
                    # Specific real-world Indian trade trends
                    if partner == "USA" and cmd_code in ["85", "30"]: base_val *= 3.5  # Tech & Pharma exports
                    if partner == "UAE" and cmd_code in ["71", "27"]: base_val *= 4.0  # Gold & Oil hub
                    if partner == "Russia" and cmd_code == "27" and year >= 2022: base_val *= 6.5  # Post-2022 Russian oil surge
                    if partner == "China" and cmd_code in ["85", "84", "29"]: base_val *= 3.8  # Electronics & API imports
                    
                    # Growth trend over time (approx 7% annual CAGR)
                    growth_factor = (1.07) ** (year - 2005)
                    trade_val = base_val * growth_factor * np.random.normal(1.0, 0.15)
                    
                    records.append({
                        "year": year,
                        "quarter": np.random.choice([1, 2, 3, 4]),
                        "reporter_code": "356",
                        "reporter_name": "India",
                        "partner_name": partner,
                        "commodity_code": cmd_code,
                        "commodity_desc": cmd_desc,
                        "trade_value_usd": max(100000.0, np.round(trade_val, 2)),
                        "flow_type": np.random.choice(["Export", "Import"], p=[0.45, 0.55])
                    })
                    
        df = pd.DataFrame(records)
        return df

    def load_or_generate_rbi_data(self) -> pd.DataFrame:
        """Loads RBI macro indicators Parquet or generates baseline."""
        parquet_files = list(self.raw_rbi_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading RBI macro data from {parquet_files[0]}...")
            return pd.read_parquet(parquet_files[0])
            
        logger.info("Generating baseline RBI macro indicators for merge...")
        years = list(range(2005, 2025))
        n = len(years)
        return pd.DataFrame({
            "year": years,
            "inr_usd_rate": np.round(np.linspace(44.0, 83.5, n) + np.random.normal(0, 1.0, n), 2),
            "india_gdp_growth_pct": np.round(np.random.normal(6.8, 1.5, n), 2),
            "crude_oil_price_usd": np.round(np.random.normal(75.0, 15.0, n), 2)
        })

    def build_features(self) -> str:
        """
        Executes feature engineering pipeline:
        1. Merges trade flows with RBI macroeconomic indicators
        2. Sorts time series by (partner, commodity, flow_type, year)
        3. Computes lags (1y, 3y, 5y) and rolling statistics (3y mean, 5y mean, volatility)
        4. Adds policy event flag (0/1) for major regulatory shifts
        5. Saves to Parquet for XGBoost and AnomalyGuard training
        """
        logger.info("Starting Trade Feature Engineering Pipeline...")
        
        trade_df = self.load_or_generate_synthetic_trade_data()
        rbi_df = self.load_or_generate_rbi_data()
        
        # Merge macro indicators on year
        df = pd.merge(trade_df, rbi_df, on="year", how="left")
        
        # Ensure correct sorting for time-series operations
        df = df.sort_values(by=["partner_name", "commodity_code", "flow_type", "year"]).reset_index(drop=True)
        
        # Group by unique trade flow time series
        group_cols = ["partner_name", "commodity_code", "flow_type"]
        
        logger.info("Computing time-series lag features (lag_1y, lag_3y, lag_5y)...")
        df["lag_1y"] = df.groupby(group_cols)["trade_value_usd"].shift(1)
        df["lag_3y"] = df.groupby(group_cols)["trade_value_usd"].shift(3)
        df["lag_5y"] = df.groupby(group_cols)["trade_value_usd"].shift(5)
        
        logger.info("Computing rolling statistics (rolling_mean_3y, rolling_mean_5y, volatility)...")
        df["rolling_mean_3y"] = df.groupby(group_cols)["trade_value_usd"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df["rolling_mean_5y"] = df.groupby(group_cols)["trade_value_usd"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df["volatility"] = df.groupby(group_cols)["trade_value_usd"].transform(lambda x: x.rolling(window=3, min_periods=1).std()).fillna(0.0)
        
        # Growth rates and GDP interactions
        df["growth_rate"] = ((df["trade_value_usd"] - df["lag_1y"]) / df["lag_1y"]).fillna(0.0)
        df["partner_gdp_growth"] = df["india_gdp_growth_pct"] + np.random.normal(0, 1.5, len(df))  # Proxy interaction
        
        # Policy event flags (0/1) for major regulatory years (e.g., 2016 Demonetization/GST prep, 2020 COVID, 2022 CEPA & Russian oil, 2023 FTP)
        policy_years = [2016, 2020, 2022, 2023]
        df["policy_event_flag"] = df["year"].apply(lambda y: 1 if y in policy_years else 0)
        
        # Fill remaining NaNs from initial lag periods with backfill/median or rolling means
        df["lag_1y"] = df["lag_1y"].fillna(df["rolling_mean_3y"])
        df["lag_3y"] = df["lag_3y"].fillna(df["rolling_mean_3y"])
        df["lag_5y"] = df["lag_5y"].fillna(df["rolling_mean_5y"])
        
        # Save processed features to Parquet
        output_file = self.processed_dir / "trade_features.parquet"
        df.to_parquet(output_file, index=False, engine="pyarrow")
        
        logger.info(f"SUCCESS: Generated 15+ time-series features across {len(df)} rows. Saved to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    engineer = TradeFeatureEngineer()
    out_path = engineer.build_features()
    print(f"\n[+] Feature Engineering completed successfully: {out_path}")
