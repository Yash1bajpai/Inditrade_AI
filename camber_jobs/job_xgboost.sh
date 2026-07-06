#!/bin/bash
# ==========================================
# Camber / Lightning AI Job — XGBoost Optuna
# ==========================================
# Executes 150-trial hyperparameter optimization for TradeFlow Forecast.
# Target: 4 CPU cores, 16GB RAM engine (~10 CPU hours)

echo "[+] Starting XGBoost Optuna Training Job on Cloud Compute..."
echo "[+] Installing dependencies..."
pip install xgboost optuna scikit-learn pandas pyarrow joblib skl2onnx onnxmltools --quiet

echo "[+] Verifying feature dataset..."
python -c "
from pathlib import Path
if not Path('data/processed/trade_features.parquet').exists():
    print('[!] Features file not found! Running feature engineering first...')
    from src.features.trade_features import TradeFeatureEngineer
    TradeFeatureEngineer().build_features()
"

echo "[+] Running Optuna 150-trial optimization & ONNX export..."
python -c "
from src.models.xgboost_train import TradeXGBoostTrainer
trainer = TradeXGBoostTrainer()
pkl_out, onnx_out = trainer.train_and_export(n_trials=150)
print(f'\n[✓] Job Complete! Models saved at:\n  - {pkl_out}\n  - {onnx_out}')
"
