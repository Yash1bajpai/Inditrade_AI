#!/bin/bash
# Export Lightning AI / Camber main environment to PATH
export PATH=/opt/jupyter/envs/main/bin:$PATH
# ==========================================
# Camber / Lightning AI Job — Anomaly Guard
# ==========================================
# Trains Isolation Forest ensemble on historical trade spikes.
# Target: 4 CPU cores, 16GB RAM engine (~3 CPU hours)

echo "[+] Starting Isolation Forest Anomaly Detection Job..."
pip install scikit-learn pandas pyarrow joblib --quiet

python -c "
from src.models.anomaly_train import TradeAnomalyTrainer
trainer = TradeAnomalyTrainer()
out = trainer.train_and_save(contamination=0.05)
print(f'\n[✓] AnomalyGuard Job Complete! Model saved at: {out}')
"
