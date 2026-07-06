"""
XGBoost & Optuna Training Module for TradeFlow Forecast (Module A).

Performs 150-trial hyperparameter optimization using TimeSeriesSplit cross-validation (5 folds).
Saves trained model to Pickle (.pkl) and converts to ONNX (.onnx) for low-memory (<512MB RAM)
inference on Render free tier.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

try:
    import optuna
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from xgboost.sklearn import XGBRegressor as SklearnXGBRegressor
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from onnxmltools.convert.xgboost.shape_calculators.Regressor import calculate_xgboost_regressor_output_shapes
    HAS_ONNX_CONVERT = True
except ImportError:
    HAS_ONNX_CONVERT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("XGBoostTrainer")


class TradeXGBoostTrainer:
    """Trains XGBoost forecasting model with Optuna optimization and ONNX export."""

    def __init__(self, 
                 features_path: str = "data/processed/trade_features.parquet",
                 models_dir: str = "models"):
        self.features_path = Path(features_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = [
            "lag_1y", "lag_3y", "lag_5y", "rolling_mean_3y", "rolling_mean_5y",
            "volatility", "inr_usd_rate", "crude_oil_price_usd", "policy_event_flag",
            "partner_gdp_growth"
        ]

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads processed features and prepares feature matrix X and target y."""
        if not self.features_path.exists():
            logger.error(f"Features file not found at {self.features_path}. Please run trade_features.py first.")
            raise FileNotFoundError(f"Missing {self.features_path}")
            
        df = pd.read_parquet(self.features_path)
        
        # Filter to required feature columns
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df[available_cols].fillna(0.0)
        y = df["trade_value_usd"].fillna(0.0)
        
        logger.info(f"Loaded feature matrix X: {X.shape}, target y: {y.shape}")
        return X, y

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 150) -> Dict[str, Any]:
        """Runs Optuna study with TimeSeriesSplit to prevent data leakage across time."""
        if not HAS_ML:
            logger.error("optuna or xgboost not installed. Cannot run optimization.")
            return {}
            
        logger.info(f"Starting Optuna hyperparameter study ({n_trials} trials, TimeSeriesSplit CV)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Check if CUDA GPU is available for XGBoost
        device_to_use = "cpu"
        try:
            test_xgb = XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
            test_xgb.fit(X.iloc[:5], y.iloc[:5])
            device_to_use = "cuda"
            logger.info("CUDA GPU detected and verified for XGBoost Optuna optimization.")
        except Exception as e:
            logger.info("CUDA GPU not available or fit failed. Using CPU for XGBoost Optuna optimization.")
            device_to_use = "cpu"

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),  # Capped at 600 for Render 512MB RAM safety
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "tree_method": "hist",
                "device": device_to_use,
                "random_state": 42,
            }
            model = XGBRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                # Avoid division by zero in MAPE
                mape = np.mean(np.abs((y_val - preds) / np.maximum(y_val, 1000.0)))
                scores.append(mape)
                
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Optuna Optimization Complete! Best CV MAPE: {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")
        
        # Save best params to json
        params_path = self.models_dir / "xgboost_best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)
            
        return study.best_params

    def train_and_export(self, n_trials: int = 150) -> Tuple[str, str]:
        """Trains final model on full dataset using best params and exports .pkl and .onnx."""
        X, y = self.load_data()
        
        if HAS_ML:
            best_params = self.optimize_hyperparameters(X, y, n_trials=n_trials)
            best_params["tree_method"] = "hist"
            # Detect device again for final fit
            try:
                test_xgb = XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
                test_xgb.fit(X.iloc[:5], y.iloc[:5])
                best_params["device"] = "cuda"
            except:
                best_params["device"] = "cpu"
            model = XGBRegressor(**best_params, random_state=42)
            model.fit(X, y)
        else:
            logger.warning("ML libs missing. Using mock/default training.")
            model = "MockXGBoostModel"
            
        # 1. Save Pickle (.pkl)
        pkl_path = self.models_dir / "xgboost_trade.pkl"
        joblib.dump(model, pkl_path)
        logger.info(f"Saved native XGBoost model to {pkl_path}")
        
        # 2. Convert and save ONNX (.onnx) for Render 512MB RAM limit
        onnx_path = self.models_dir / "xgboost_trade.onnx"
        if HAS_ML and HAS_ONNX_CONVERT and isinstance(model, XGBRegressor):
            try:
                logger.info("Converting XGBoost model to ONNX format...")
                update_registered_converter(
                    SklearnXGBRegressor,
                    "XGBoostXGBRegressor",
                    calculate_xgboost_regressor_output_shapes,
                    convert_xgboost
                )
                initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
                
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                logger.info(f"SUCCESS: Exported ONNX model ({os.path.getsize(onnx_path)/1024:.1f} KB) to {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX conversion encountered an issue: {e}. API will fallback to lazy loading .pkl.")
                
        return str(pkl_path), str(onnx_path)


if __name__ == "__main__":
    trainer = TradeXGBoostTrainer()
    # Using 10 trials for quick local verification; Week 2 Camber job will run full 150 trials
    pkl_out, onnx_out = trainer.train_and_export(n_trials=10)
    print(f"\n[+] XGBoost Training Complete. PKL: {pkl_out}, ONNX: {onnx_out}")
