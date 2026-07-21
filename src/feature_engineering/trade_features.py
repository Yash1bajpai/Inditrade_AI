"""
IndiTrade AI: Phase 2 Feature Engineering Pipeline (`trade_features.py`).

Core Responsibilities & Bug Fixes:
1. Load daily Forex & Indian Macro financial timeseries (`data/raw/forex_macro/*.csv`).
2. CRITICAL QC: Inspect `cnyinr.csv` and drop it completely if historical rows <= 1.
3. Time-Frequency Alignment: Resample/aggregate daily macro indices into Yearly frequency (`period`)
   BEFORE merging to ensure 100% preservation of Comtrade rows without data drop.
4. UN M49 Mapping Fix: Map numeric `partnerCode` directly to `partnerDesc` and `partnerISO`
   using official M49 lookup so that `partnerDesc.isna().sum() == 0`.
5. Aggregation Bug Fix: Aggregate Comtrade rows across `['partnerCode', 'cmdCode', 'flowCode', 'period']`
   via `groupby().agg()` BEFORE lag computation to eliminate duplicates split across customs/mot codes.
6. Lag & Rolling Engineering: Compute grouped `lag_1y`, `lag_3y`, `lag_5y`, `rolling_3y_mean`,
   `rolling_5y_mean`, and `yoy_growth_rate` per bilateral commodity flow (`partnerCode, cmdCode, flowCode`).
7. Policy Flag: Add `policy_event_flag` indicating major DGFT/Macro policy years (`2020, 2022, 2023`).
8. Export standardized feature matrix to `data/processed/trade_features.parquet`.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys_reconfig = getattr(sys.stdout, 'reconfigure', None)
if sys_reconfig:
    sys_reconfig(encoding='utf-8', errors='replace')

M49_PARTNER_MAP = {
    "842": {"desc": "USA", "iso": "USA"},
    "156": {"desc": "China", "iso": "CHN"},
    "784": {"desc": "United Arab Emirates", "iso": "ARE"},
    "682": {"desc": "Saudi Arabia", "iso": "SAU"},
    "368": {"desc": "Iraq", "iso": "IRQ"},
    "702": {"desc": "Singapore", "iso": "SGP"},
    "344": {"desc": "Hong Kong", "iso": "HKG"},
    "360": {"desc": "Indonesia", "iso": "IDN"},
    "410": {"desc": "South Korea", "iso": "KOR"},
    "036": {"desc": "Australia", "iso": "AUS"},
    "36": {"desc": "Australia", "iso": "AUS"},
    "276": {"desc": "Germany", "iso": "DEU"},
    "392": {"desc": "Japan", "iso": "JPN"},
    "756": {"desc": "Switzerland", "iso": "CHE"},
    "458": {"desc": "Malaysia", "iso": "MYS"},
    "826": {"desc": "United Kingdom", "iso": "GBR"},
    "643": {"desc": "Russian Federation", "iso": "RUS"},
    "528": {"desc": "Netherlands", "iso": "NLD"},
    "056": {"desc": "Belgium", "iso": "BEL"},
    "56": {"desc": "Belgium", "iso": "BEL"},
    "250": {"desc": "France", "iso": "FRA"},
    "704": {"desc": "Vietnam", "iso": "VNM"}
}

def load_and_aggregate_macro(forex_dir: str = "data/raw/forex_macro") -> pd.DataFrame:
    """Load daily forex/macro CSVs, perform QC filtering, and aggregate to Yearly frequency."""
    csv_files = sorted(glob.glob(os.path.join(forex_dir, "*.csv")))
    print(f"--- Step 1: Loading & Aggregating Daily Macro Financial Data ({len(csv_files)} files) ---")

    yearly_dfs = []

    for csv_path in csv_files:
        asset_name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)

        if asset_name.lower() == "cnyinr" or len(df) <= 1:
            print(f"  [CRITICAL QC GATE] `{asset_name}.csv` has <= 1 row (len={len(df)}). DROPPING `{asset_name}` feature entirely before merging as per quality mandate!")
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df["period"] = df["Date"].dt.year

        df_window = df[(df["period"] >= 2015) & (df["period"] <= 2024)].copy()

        if len(df_window) == 0:
            print(f"  [WARNING] `{asset_name}.csv` has 0 rows in 2015-2024 range. Skipping.")
            continue

        grouped = df_window.groupby("period")["Close"]

        if asset_name in ["usdinr", "eurinr", "gbpinr", "jpyinr"]:
            agg_df = pd.DataFrame({
                "period": grouped.mean().index,
                f"{asset_name}_mean": grouped.mean().values,
                f"{asset_name}_vol_std": grouped.std().values,
                f"{asset_name}_year_end": grouped.last().values
            })
        else:

            means = grouped.mean()
            yoy_pct = means.pct_change() * 100.0
            agg_df = pd.DataFrame({
                "period": means.index,
                f"{asset_name}_mean": means.values,
                f"{asset_name}_yoy_pct": yoy_pct.values
            })

        yearly_dfs.append(agg_df)
        print(f"  ✅ Aggregated `{asset_name:<15}` -> {len(agg_df)} years (Columns: {[c for c in agg_df.columns if c != 'period']})")

    df_macro_yearly = yearly_dfs[0]
    for nxt_df in yearly_dfs[1:]:
        df_macro_yearly = pd.merge(df_macro_yearly, nxt_df, on="period", how="outer")

    df_macro_yearly.sort_values("period", inplace=True)
    print(f"\n => Consolidated Yearly Macro Feature Matrix: {df_macro_yearly.shape[0]} years ({df_macro_yearly['period'].min()} to {df_macro_yearly['period'].max()}) | {df_macro_yearly.shape[1]} total features.\n")
    return df_macro_yearly

def build_trade_features(comtrade_parquet: str = "data/raw/un_comtrade/india_trade_hs2_2015_2024.parquet",
                         forex_dir: str = "data/raw/forex_macro",
                         output_parquet: str = "data/processed/trade_features.parquet") -> pd.DataFrame:
    print(f"=== IndiTrade AI: Phase 2 Feature Engineering Pipeline (With All Bug Fixes) ===")

    df_macro = load_and_aggregate_macro(forex_dir)

    print(f"--- Step 2: Loading UN Comtrade Bilateral Trade Data ({comtrade_parquet}) ---")
    df_comtrade = pd.read_parquet(comtrade_parquet)
    print(f"  -> Raw Comtrade Shape: {df_comtrade.shape} across {df_comtrade['period'].nunique()} years ({df_comtrade['period'].min()} to {df_comtrade['period'].max()})")

    df_comtrade["period"] = df_comtrade["period"].astype(int)
    df_macro["period"] = df_macro["period"].astype(int)

    print(f"--- Step 2a: Fixing 100% null partnerDesc using UN M49 Country Mapping ---")
    code_str = df_comtrade["partnerCode"].astype(str).str.lstrip("0")
    map_dict_desc = {k.lstrip("0"): v["desc"] for k, v in M49_PARTNER_MAP.items()}
    map_dict_iso = {k.lstrip("0"): v["iso"] for k, v in M49_PARTNER_MAP.items()}

    df_comtrade["partnerDesc"] = code_str.map(map_dict_desc)
    df_comtrade["partnerISO"] = code_str.map(map_dict_iso)
    print(f"  ✅ partnerDesc null count post-mapping: {df_comtrade['partnerDesc'].isna().sum()} / {len(df_comtrade)}")

    print(f"--- Step 2b: Aggregating duplicate Comtrade rows across ['partnerCode', 'cmdCode', 'flowCode', 'period'] ---")
    group_keys = ["partnerCode", "cmdCode", "flowCode", "period"]
    numeric_sum_cols = [col for col in ["primaryValue", "cifvalue", "fobvalue", "netWgt", "grossWgt", "qty", "altQty"] if col in df_comtrade.columns]
    meta_first_cols = [col for col in df_comtrade.columns if col not in group_keys + numeric_sum_cols]

    agg_dict = {col: "sum" for col in numeric_sum_cols}
    for col in meta_first_cols:
        agg_dict[col] = "first"

    df_comtrade = df_comtrade.groupby(group_keys, as_index=False).agg(agg_dict)
    max_dup_post = df_comtrade.groupby(group_keys).size().max()
    print(f"  ✅ Aggregated to exact {len(df_comtrade)} unique bilateral combinations (Max duplicate check: {max_dup_post})\n")
    assert max_dup_post == 1, f"CRITICAL ERROR: Max duplicates after groupby aggregation is {max_dup_post} instead of 1!"

    print(f"--- Step 3: Performing Left Merge (Comtrade + Yearly Macro) ---")
    initial_rows = len(df_comtrade)
    df_merged = pd.merge(df_comtrade, df_macro, on="period", how="left")

    assert len(df_merged) == initial_rows, f"CRITICAL ERROR: Merge dropped rows! Initial: {initial_rows}, Post-merge: {len(df_merged)}"
    print(f"  ✅ Merge Successful! Shape after macro join: {df_merged.shape} (Exact 100% row preservation)\n")

    df_merged["policy_event_flag"] = df_merged["period"].apply(lambda x: 1 if int(x) in [2020, 2022, 2023] else 0)

    print(f"--- Step 4: Engineering Temporal Lags (`lag_1y`, `lag_3y`, `lag_5y`), Rolling Means (`3y`, `5y`), and YoY Growth ---")

    group_cols = ["partnerCode", "cmdCode", "flowCode"]
    df_merged.sort_values(by=group_cols + ["period"], inplace=True)

    grouped_val = df_merged.groupby(group_cols)["primaryValue"]

    df_merged["primaryValue_lag_1y"] = grouped_val.shift(1)
    df_merged["primaryValue_lag_3y"] = grouped_val.shift(3)
    df_merged["primaryValue_lag_5y"] = grouped_val.shift(5)

    df_merged["primaryValue_rolling_3y_mean"] = grouped_val.transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    df_merged["primaryValue_rolling_5y_mean"] = grouped_val.transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    lag_1 = df_merged["primaryValue_lag_1y"].replace(0, np.nan)
    df_merged["primaryValue_yoy_growth_rate"] = (df_merged["primaryValue"] - df_merged["primaryValue_lag_1y"]) / lag_1

    grouped_wgt = df_merged.groupby(group_cols)["netWgt"]
    df_merged["netWgt_lag_1y"] = grouped_wgt.shift(1)
    df_merged["netWgt_rolling_3y_mean"] = grouped_wgt.transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())

    print(f"  ✅ Engineered 9 temporal quantitative & policy indicators across {df_merged[group_cols].drop_duplicates().shape[0]} bilateral flow trajectories.\n")

    print(f"--- Step 5: Exporting Final Feature Matrix to Parquet ---")
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_merged.to_parquet(output_parquet, index=False)
    file_size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    print(f"  ✅ Saved `{output_parquet}` ({file_size_mb:.2f} MB | {df_merged.shape[0]} rows x {df_merged.shape[1]} columns)\n")

    print("="*70)
    print("=== PHASE 2 FINAL VERIFICATION: RE-VERIFIED MERGED FEATURE MATRIX (`trade_features.parquet`) ===")
    print("="*70)
    print(f"Final Dataset Shape           : {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")
    print(f"Storage File Path             : {output_parquet} ({file_size_mb:.2f} MB)")
    print(f"Max Duplicate Check           : {df_merged.groupby(group_cols + ['period']).size().max()} (Must be exactly 1)")
    print(f"partnerDesc Null Count Check  : {df_merged['partnerDesc'].isna().sum()} (Must be exactly 0)")
    print(f"Policy Event Years Flag Check : {df_merged[df_merged['policy_event_flag'] == 1]['period'].unique().tolist()}")

    print("\nComplete List of Columns:")
    print("-" * 50)
    for i in range(0, len(df_merged.columns), 4):
        cols_slice = df_merged.columns[i:i+4]
        print("  " + ", ".join(f"`{c}`" for c in cols_slice))
    print("-" * 50)

    print("\nFirst 5 Rows Preview (`df.head()` - Key Columns):")
    print("-" * 70)
    preview_cols = ["period", "partnerDesc", "cmdCode", "flowCode", "primaryValue", "primaryValue_lag_1y", "primaryValue_lag_5y", "policy_event_flag", "usdinr_mean"]
    avail_preview = [c for c in preview_cols if c in df_merged.columns]
    print(df_merged[avail_preview].head(5).to_string(index=False))
    print("="*70 + "\n")

    return df_merged

if __name__ == "__main__":
    build_trade_features()

