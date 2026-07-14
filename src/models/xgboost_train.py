"""
IndiTrade AI - Module A: Quantitative Trade Flow Forecasting (XGBoost + Optuna + ONNX)
Target: primaryValue (UN Comtrade standardized bilateral trade dollar value)
Validation: TimeSeriesSplit(5) across chronological trade flow periods (2015-2024).
Hyperparameter Tuning: Optuna Bayesian Optimization (150 trials default).
Export: joblib (.pkl) and ONNX (.onnx) formats for production deployment.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import optuna
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType

# Configure safe stdout encoding for Windows/remote terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Disable optuna chatty logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

METADATA_COLS_TO_DROP = [
    'typeCode', 'freqCode', 'refPeriodId', 'isLeaf', 'isAggregate', 'legacyEstimationFlag', 'isReported',
    'classificationCode', 'classificationSearchCode', 'isOriginalClassification', 'motCode', 'motDesc',
    'mosCode', 'customsCode', 'customsDesc', 'qtyUnitCode', 'qtyUnitAbbr', 'altQtyUnitCode', 'altQtyUnitAbbr',
    'isQtyEstimated', 'isAltQtyEstimated', 'isNetWgtEstimated', 'isGrossWgtEstimated', 'partner2Code',
    'partner2ISO', 'partner2Desc'
]

def load_and_preprocess_data(data_path, sample_size=None):
    """
    Loads trade_features.parquet, drops raw metadata/leaks, handles infinite/null values,
    and returns feature matrix X, log-transformed target y_log, and raw target y_raw.
    """
    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Sort chronologically and by trade flow identifiers for TimeSeriesSplit
    sort_cols = [c for c in ['period', 'partnerCode', 'cmdCode', 'flowCode'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
    if sample_size and len(df) > sample_size:
        print(f"[*] Subsampling dataset to {sample_size} chronological rows for rapid end-to-end verification...")
        df = df.tail(sample_size).reset_index(drop=True)
        
    # Drop raw metadata columns if present
    drop_cols = [c for c in METADATA_COLS_TO_DROP if c in df.columns]
    
    # Also drop exact target leak columns that represent alternative valuations not known at forecast time
    leak_cols = [c for c in ['cifvalue', 'fobvalue'] if c in df.columns]
    df_clean = df.drop(columns=drop_cols + leak_cols)
    
    # Ensure target primaryValue exists
    if 'primaryValue' not in df_clean.columns:
        raise ValueError("CRITICAL: Target column 'primaryValue' not found in dataset!")
        
    y_raw = df_clean['primaryValue'].astype(float).fillna(0)
    # Log-transform target for numerical stability across multi-scale commodities ($1k to $50B)
    y_log = np.log1p(np.maximum(y_raw, 0))
    
    # Prepare feature set: drop target and descriptive string metadata (keep numeric identifiers and all features)
    ignore_cols = ['primaryValue', 'partnerDesc', 'cmdDesc', 'flowDesc', 'partnerISO']
    feature_cols = [c for c in df_clean.columns if c not in ignore_cols]
    
    X = df_clean[feature_cols].copy()
    
    # Convert all columns to numerical float32 for clean ONNX export
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float32)
        
    # Replace any inf values
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"[OK] Preprocessed feature matrix shape: {X.shape} | Target shape: {y_log.shape}")
    print(f"[OK] Features included ({len(feature_cols)} total): {feature_cols[:8]} ... plus {len(feature_cols)-8} more")
    return X, y_log, y_raw, feature_cols, df_clean

def objective(trial, X, y):
    """Optuna objective function using TimeSeriesSplit(5)."""
    tscv = TimeSeriesSplit(n_splits=5)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        
    return np.mean(scores)

def main():
    parser = argparse.ArgumentParser(description="IndiTrade AI - XGBoost Trade Flow Forecast Trainer")
    parser.add_argument("--data-path", type=str, default="data/processed/trade_features.parquet", help="Path to trade_features.parquet")
    parser.add_argument("--trials", type=int, default=150, help="Number of Optuna Bayesian trials (default: 150)")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for quick verification (e.g. 2000)")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save exported .pkl and .onnx models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("=== [PHASE 3: MODULE A] XGBOOST BILATERAL TRADE FLOW FORECASTING ===")
    
    X, y_log, y_raw, feature_cols, df_raw = load_and_preprocess_data(args.data_path, args.sample)
    
    # Chronological Holdout Split: Train (<= 2021), Test (>= 2022)
    train_mask = df_raw['period'] <= 2021
    test_mask = df_raw['period'] >= 2022
    
    X_train, y_log_train, y_raw_train = X[train_mask], y_log[train_mask], y_raw[train_mask]
    X_test, y_log_test, y_raw_test = X[test_mask], y_log[test_mask], y_raw[test_mask]
    
    print(f"\n[*] Chronological Holdout Split -> Train rows (<=2021): {len(X_train)} | Test rows (>=2022): {len(X_test)}")
    
    print(f"\n[*] Launching Optuna Bayesian Optimization across {args.trials} trials with TimeSeriesSplit(5) on Train set...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_log_train), n_trials=args.trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    print(f"\n[SUCCESS] Optuna Optimization Complete!")
    print(f"[BEST CV LOG-RMSE] : {study.best_value:.4f}")
    print(f"[BEST HYPERPARAMS] : {best_params}")
    
    # Train final production model on chronological Train set (<=2021)
    print("\n[*] Training final production XGBRegressor on chronological Train set (<=2021)...")
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_log_train, eval_set=[(X_test, y_log_test)], verbose=False)
    
    # Evaluate ONLY on the held-out test set (2022-2024)
    preds_log_test = final_model.predict(X_test)
    preds_dollar_test = np.expm1(np.maximum(preds_log_test, 0))
    
    r2_val = r2_score(y_log_test, preds_log_test)
    mae_dollar = mean_absolute_error(y_raw_test, preds_dollar_test)
    rmse_dollar = np.sqrt(mean_squared_error(y_raw_test, preds_dollar_test))
    
    print("\n=== OUT-OF-SAMPLE TEST PERFORMANCE (Periods >= 2022) ===")
    print(f"  * Test Log-Scale R² Score : {r2_val:.4f}")
    print(f"  * Test Dollar-Scale MAE   : ${mae_dollar:,.2f}")
    print(f"  * Test Dollar-Scale RMSE  : ${rmse_dollar:,.2f}")
    
    # Top 10 Feature Importances
    importances = final_model.feature_importances_
    feat_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    print("\n=== TOP 10 MOST INFLUENTIAL TRADE & MACRO FEATURES ===")
    for idx, row in feat_df.head(10).iterrows():
        print(f"  [{row['feature']:30s}] : {row['importance']:.4f}")
        
    # Export to joblib (.pkl)
    pkl_path = os.path.join(args.output_dir, "xgboost_trade_forecast.pkl")
    joblib.dump({"model": final_model, "features": feature_cols, "params": best_params, "holdout_split_year": 2022}, pkl_path)
    print(f"\n[Exported] trained model checkpoint -> {pkl_path}")
    
    # Export to ONNX (.onnx)
    print("[*] Converting trained XGBoost model to ONNX format for zero-dependency inference...")
    try:
        # ONNX converter requires Booster feature names to follow f%d pattern
        booster = final_model.get_booster()
        booster.feature_names = [f"f{i}" for i in range(len(feature_cols))]
        initial_types = [('float_input', FloatTensorType([None, len(feature_cols)]))]
        onnx_model = onnxmltools.convert_xgboost(final_model, initial_types=initial_types)
        onnx_path = os.path.join(args.output_dir, "xgboost_trade_forecast.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[Exported] production ONNX binary -> {onnx_path} ({os.path.getsize(onnx_path)/1024:.2f} KB)")
    except Exception as e:
        print(f"[Warning] ONNX conversion note: {e}")
        
    # Export comprehensive meta.json for exact verification from disk
    meta_path = os.path.join(args.output_dir, "xgboost_trade_forecast_meta.json")
    meta_data = {
        "model_name": "XGBoost Bilateral Trade Flow Forecast (Module A)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "holdout_split_year": 2022,
        "n_samples_total": int(len(X)),
        "n_samples_train_le2021": int(len(X_train)),
        "n_samples_test_ge2022": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "feature_names": list(feature_cols),
        "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best_params.items()},
        "metrics": {
            "best_cv_log_rmse_train": float(study.best_value),
            "test_log_scale_r2": float(r2_val),
            "test_dollar_scale_rmse": float(rmse_dollar),
            "test_dollar_scale_mae": float(mae_dollar)
        },
        "target_statistics": {
            "target_column": "primaryValue",
            "mean": float(np.mean(y_raw)),
            "median": float(np.median(y_raw)),
            "std": float(np.std(y_raw)),
            "min": float(np.min(y_raw)),
            "max": float(np.max(y_raw))
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    print(f"[Exported] exact model metadata -> {meta_path}")
        
    print("\n[COMPLETE] Module A (XGBoost Trade Flow Forecast) pipeline complete!")

if __name__ == "__main__":
    main()
