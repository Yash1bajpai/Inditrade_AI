"""
IndiTrade AI: Phase 2 Feature Engineering Pipeline (`trade_features.py`).

Core Responsibilities:
1. Load UN Comtrade HS 2-Digit bilateral trade flows (2015-2024).
2. Load daily Forex & Indian Macro financial timeseries (`data/raw/forex_macro/*.csv`).
3. CRITICAL QC: Inspect `cnyinr.csv` and drop it completely if historical rows <= 1.
4. Time-Frequency Alignment: Resample/aggregate daily macro indices into Yearly frequency (`period`)
   BEFORE merging to ensure 100% preservation of Comtrade rows without data drop.
5. Lag & Rolling Engineering: Compute grouped `lag_1y`, `lag_3y`, `rolling_3y_mean`,
   `rolling_5y_mean`, and `yoy_growth_rate` per bilateral commodity flow (`partnerCode, cmdCode, flowCode`).
6. Export standardized feature matrix to `data/processed/trade_features.parquet`.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

# Configure UTF-8 stdout encoding
sys_reconfig = getattr(sys.stdout, 'reconfigure', None)
if sys_reconfig:
    sys_reconfig(encoding='utf-8', errors='replace')


def load_and_aggregate_macro(forex_dir: str = "data/raw/forex_macro") -> pd.DataFrame:
    """Load daily forex/macro CSVs, perform QC filtering, and aggregate to Yearly frequency."""
    csv_files = sorted(glob.glob(os.path.join(forex_dir, "*.csv")))
    print(f"--- Step 1: Loading & Aggregating Daily Macro Financial Data ({len(csv_files)} files) ---")
    
    yearly_dfs = []
    
    for csv_path in csv_files:
        asset_name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        
        # CRITICAL QC: Check CNY/INR or sparse historical datasets
        if asset_name.lower() == "cnyinr" or len(df) <= 1:
            print(f"  [CRITICAL QC GATE] `{asset_name}.csv` has <= 1 row (len={len(df)}). DROPPING `{asset_name}` feature entirely before merging as per quality mandate!")
            continue
            
        # Parse Date and extract Year (`period`)
        df["Date"] = pd.to_datetime(df["Date"])
        df["period"] = df["Date"].dt.year
        
        # Filter for our active historical window (2015 to 2024)
        df_window = df[(df["period"] >= 2015) & (df["period"] <= 2024)].copy()
        
        if len(df_window) == 0:
            print(f"  [WARNING] `{asset_name}.csv` has 0 rows in 2015-2024 range. Skipping.")
            continue
            
        # Group by Year (`period`) to match Comtrade frequency
        grouped = df_window.groupby("period")["Close"]
        
        if asset_name in ["usdinr", "eurinr", "gbpinr", "jpyinr"]:
            agg_df = pd.DataFrame({
                "period": grouped.mean().index,
                f"{asset_name}_mean": grouped.mean().values,
                f"{asset_name}_vol_std": grouped.std().values,
                f"{asset_name}_year_end": grouped.last().values
            })
        else:
            # For equity/commodity indices (Nifty 50, Sensex, Gold, Brent Crude)
            means = grouped.mean()
            yoy_pct = means.pct_change() * 100.0
            agg_df = pd.DataFrame({
                "period": means.index,
                f"{asset_name}_mean": means.values,
                f"{asset_name}_yoy_pct": yoy_pct.values
            })
            
        yearly_dfs.append(agg_df)
        print(f"  ✅ Aggregated `{asset_name:<15}` -> {len(agg_df)} years (Columns: {[c for c in agg_df.columns if c != 'period']})")
        
    # Merge all aggregated yearly tables on `period`
    df_macro_yearly = yearly_dfs[0]
    for nxt_df in yearly_dfs[1:]:
        df_macro_yearly = pd.merge(df_macro_yearly, nxt_df, on="period", how="outer")
        
    df_macro_yearly.sort_values("period", inplace=True)
    print(f"\n => Consolidated Yearly Macro Feature Matrix: {df_macro_yearly.shape[0]} years ({df_macro_yearly['period'].min()} to {df_macro_yearly['period'].max()}) | {df_macro_yearly.shape[1]} total features.\n")
    return df_macro_yearly


def build_trade_features(comtrade_parquet: str = "data/raw/un_comtrade/india_trade_hs2_2015_2024.parquet",
                         forex_dir: str = "data/raw/forex_macro",
                         output_parquet: str = "data/processed/trade_features.parquet") -> pd.DataFrame:
    print(f"=== IndiTrade AI: Phase 2 Feature Engineering Pipeline ===")
    
    # 1. Aggregate daily macro data to yearly
    df_macro = load_and_aggregate_macro(forex_dir)
    
    # 2. Load UN Comtrade bilateral trade data
    print(f"--- Step 2: Loading UN Comtrade Bilateral Trade Data ({comtrade_parquet}) ---")
    df_comtrade = pd.read_parquet(comtrade_parquet)
    print(f"  -> Raw Comtrade Shape: {df_comtrade.shape} across {df_comtrade['period'].nunique()} years ({df_comtrade['period'].min()} to {df_comtrade['period'].max()})")
    
    # Ensure exact integer alignment for merging without dropping rows
    df_comtrade["period"] = df_comtrade["period"].astype(int)
    df_macro["period"] = df_macro["period"].astype(int)
    
    # 3. Merge Comtrade with Aggregated Macro
    print(f"--- Step 3: Performing Left Merge (Comtrade + Yearly Macro) ---")
    initial_rows = len(df_comtrade)
    df_merged = pd.merge(df_comtrade, df_macro, on="period", how="left")
    
    assert len(df_merged) == initial_rows, f"CRITICAL ERROR: Merge dropped rows! Initial: {initial_rows}, Post-merge: {len(df_merged)}"
    print(f"  ✅ Merge Successful! Shape after macro join: {df_merged.shape} (Exact 100% row preservation)\n")
    
    # 4. Create Lag, Rolling Mean, and YoY Growth Features
    print(f"--- Step 4: Engineering Temporal Lags (`lag_1y`, `lag_3y`), Rolling Means (`3y`, `5y`), and YoY Growth ---")
    
    # Sort logically by bilateral flow and time before applying shifts
    group_cols = ["partnerCode", "cmdCode", "flowCode"]
    df_merged.sort_values(by=group_cols + ["period"], inplace=True)
    
    # Group by unique trade trajectory
    grouped_val = df_merged.groupby(group_cols)["primaryValue"]
    
    # Lag features
    df_merged["primaryValue_lag_1y"] = grouped_val.shift(1)
    df_merged["primaryValue_lag_3y"] = grouped_val.shift(3)
    
    # Rolling means
    df_merged["primaryValue_rolling_3y_mean"] = grouped_val.transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df_merged["primaryValue_rolling_5y_mean"] = grouped_val.transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # YoY Growth Rate
    lag_1 = df_merged["primaryValue_lag_1y"].replace(0, np.nan)
    df_merged["primaryValue_yoy_growth_rate"] = (df_merged["primaryValue"] - df_merged["primaryValue_lag_1y"]) / lag_1
    
    # Also create lag/rolling for Net Weight (`netWgt`)
    grouped_wgt = df_merged.groupby(group_cols)["netWgt"]
    df_merged["netWgt_lag_1y"] = grouped_wgt.shift(1)
    df_merged["netWgt_rolling_3y_mean"] = grouped_wgt.transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    print(f"  ✅ Engineered 7 temporal quantitative features across {df_merged[group_cols].drop_duplicates().shape[0]} bilateral flow trajectories.\n")
    
    # 5. Export to Parquet
    print(f"--- Step 5: Exporting Final Feature Matrix to Parquet ---")
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_merged.to_parquet(output_parquet, index=False)
    file_size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    print(f"  ✅ Saved `{output_parquet}` ({file_size_mb:.2f} MB | {df_merged.shape[0]} rows x {df_merged.shape[1]} columns)\n")
    
    # 6. Verification Display
    print("="*70)
    print("=== PHASE 2 FINAL VERIFICATION: MERGED FEATURE MATRIX (`trade_features.parquet`) ===")
    print("="*70)
    print(f"Final Dataset Shape : {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")
    print(f"Storage File Path   : {output_parquet} ({file_size_mb:.2f} MB)")
    print("\nComplete List of Columns:")
    print("-" * 50)
    for i in range(0, len(df_merged.columns), 4):
        cols_slice = df_merged.columns[i:i+4]
        print("  " + ", ".join(f"`{c}`" for c in cols_slice))
    print("-" * 50)
    
    print("\nFirst 5 Rows Preview (`df.head()` - Key Columns):")
    print("-" * 70)
    preview_cols = ["period", "partnerDesc", "cmdCode", "flowCode", "primaryValue", "primaryValue_lag_1y", "primaryValue_yoy_growth_rate", "usdinr_mean", "brent_crude_mean"]
    avail_preview = [c for c in preview_cols if c in df_merged.columns]
    print(df_merged[avail_preview].head(5).to_string(index=False))
    print("="*70 + "\n")
    
    return df_merged


if __name__ == "__main__":
    build_trade_features()
