"""
IndiTrade AI - Module D: Trade Flow Anomaly & Misinvoicing Detection (Isolation Forest)
Identifies abnormal trade volume shocks, price-per-kg deviations, and structural trade outliers across UN Comtrade bilateral flows.
Export: joblib (.pkl) checkpoint for real-time anomaly filtering.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def load_anomaly_features(data_path):
    print(f"[*] Loading trade features from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Calculate Unit Value (Price per KG) and Unit Value Shock
    df['unit_value'] = df['primaryValue'] / np.maximum(df['netWgt'].fillna(0), 1.0)
    df['value_vs_3y_mean'] = df['primaryValue'] / np.maximum(df['primaryValue_rolling_3y_mean'].fillna(0), 1.0)
    df['wgt_vs_3y_mean'] = df['netWgt'] / np.maximum(df['netWgt_rolling_3y_mean'].fillna(0), 1.0)
    
    # Select key structural features that define trade misinvoicing and macroeconomic shocks
    anomaly_feature_cols = [
        'primaryValue_yoy_growth_rate',
        'value_vs_3y_mean',
        'wgt_vs_3y_mean',
        'unit_value',
        'usdinr_yoy_pct',
        'brent_crude_yoy_pct',
        'policy_event_flag'
    ]
    
    # Ensure selected features exist and handle infinite/nulls
    cols_present = [c for c in anomaly_feature_cols if c in df.columns]
    X = df[cols_present].copy()
    
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype(np.float32)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"[OK] Anomaly feature matrix shape: {X.shape} across {len(cols_present)} indicators.")
    return X, df, cols_present

def main():
    parser = argparse.ArgumentParser(description="IndiTrade AI - Isolation Forest Trade Anomaly Detector")
    parser.add_argument("--data-path", type=str, default="data/processed/trade_features.parquet", help="Path to trade_features.parquet")
    parser.add_argument("--contamination", type=float, default=0.01, help="Expected anomaly ratio (default: 0.01 / 1%)")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save exported .pkl model")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("=== [PHASE 3: MODULE D] ISOLATION FOREST TRADE ANOMALY & MISINVOICING DETECTION ===")
    
    X, df_raw, feat_cols = load_anomaly_features(args.data_path)
    
    print(f"\n[*] Training IsolationForest (n_estimators=200, contamination={args.contamination})...")
    iso_forest = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.fit(X)
    
    # Predict (-1 for anomaly, 1 for normal) and calculate severity score
    preds = iso_forest.predict(X)
    # decision_function gives negative values for anomalies; multiply by -1 so higher = more anomalous
    scores = -1.0 * iso_forest.decision_function(X)
    
    df_raw['anomaly_pred'] = preds
    df_raw['anomaly_score'] = scores
    
    anomalies = df_raw[df_raw['anomaly_pred'] == -1].copy()
    anomalies = anomalies.sort_values('anomaly_score', ascending=False)
    
    print(f"\n[SUCCESS] Anomaly Detection Complete!")
    print(f"[TOTAL ANOMALIES FLAGGED] : {len(anomalies)} out of {len(df_raw)} trade flows ({len(anomalies)/len(df_raw)*100:.2f}%)")
    
    print("\n=== TOP 3 MOST EXTREME TRADE FLOW ANOMALIES BY ANOMALY SCORE ===")
    top3 = anomalies.head(3)
    for idx, row in top3.iterrows():
        p_desc = str(row.get('partnerDesc', 'N/A'))
        if pd.isna(p_desc) or p_desc == 'None' or p_desc == 'nan':
            p_desc = f"Partner Code {row.get('partnerCode', 'N/A')}"
            
        c_code = row.get('cmdCode', 'N/A')
        c_desc = str(row.get('cmdDesc', 'N/A'))
        if pd.isna(c_desc) or c_desc == 'None' or c_desc == 'nan':
            c_desc = f"HS Code {c_code}"
        else:
            c_desc = f"HS {c_code} ({c_desc[:40]})"
            
        f_desc = str(row.get('flowDesc', row.get('flowCode', 'N/A')))
        period = row.get('period', 'N/A')
        val = row.get('primaryValue', 0)
        yoy = row.get('primaryValue_yoy_growth_rate', 0)
        score = row.get('anomaly_score', 0)
        
        print(f"\n  [ANOMALY #{top3.index.get_loc(idx)+1}] Score: {score:.4f} | Year: {period}")
        print(f"    * Partner    : {p_desc}")
        print(f"    * Commodity  : {c_desc}")
        print(f"    * Flow Type  : {f_desc}")
        print(f"    * Trade Value: ${val:,.2f}")
        print(f"    * YoY Growth : {yoy:+.2f}%")
        
    # Export model
    pkl_path = os.path.join(args.output_dir, "isolation_forest_anomalies.pkl")
    joblib.dump({"model": iso_forest, "features": feat_cols, "contamination": args.contamination}, pkl_path)
    print(f"\n[Exported] trained anomaly detector -> {pkl_path}")
    print("🎯 Module D (Isolation Forest Trade Anomaly Detector) complete!")

if __name__ == "__main__":
    main()
