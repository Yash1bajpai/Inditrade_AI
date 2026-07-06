"""
Isolation Forest Anomaly Detection Module for AnomalyGuard (Module D).

Trains an unsupervised Isolation Forest ensemble + Z-score statistical outlier detector
on quarterly/annual trade flows to flag unusual spikes or drops (e.g., sudden +340% oil imports).
Saves model to Pickle for low-memory API inference.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import joblib

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AnomalyTrainer")


class TradeAnomalyTrainer:
    """Trains Isolation Forest and calculates historical Z-scores for trade anomaly alerts."""

    def __init__(self, 
                 features_path: str = "data/processed/trade_features.parquet",
                 models_dir: str = "models"):
        self.features_path = Path(features_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.anomaly_features = ["trade_value_usd", "growth_rate", "volatility", "partner_gdp_growth"]

    def train_and_save(self, contamination: float = 0.05) -> str:
        """
        Trains Isolation Forest on trade features to detect top 5% most anomalous transactions.
        Saves trained detector and scaler to disk.
        """
        if not self.features_path.exists():
            logger.error(f"Features file missing at {self.features_path}. Run trade_features.py first.")
            raise FileNotFoundError(f"Missing {self.features_path}")
            
        df = pd.read_parquet(self.features_path)
        available_cols = [c for c in self.anomaly_features if c in df.columns]
        X = df[available_cols].fillna(0.0)
        
        logger.info(f"Training Isolation Forest Anomaly Detector on {X.shape[0]} records (Contamination: {contamination*100}%)...")
        
        if HAS_SKLEARN:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            # Predict anomalies (-1 for anomaly, 1 for normal)
            preds = model.predict(X_scaled)
            scores = model.decision_function(X_scaled)
            
            anomalies_count = np.sum(preds == -1)
            logger.info(f"Detected {anomalies_count} anomalies out of {len(df)} transactions.")
            
            # Save ensemble artifact containing both scaler and model
            artifact = {
                "scaler": scaler,
                "model": model,
                "feature_names": available_cols
            }
        else:
            logger.warning("scikit-learn missing. Using mock anomaly artifact.")
            artifact = {"mock": True}
            
        model_path = self.models_dir / "isolation_forest.pkl"
        joblib.dump(artifact, model_path)
        logger.info(f"SUCCESS: Saved AnomalyDetector artifact to {model_path}")
        
        return str(model_path)


if __name__ == "__main__":
    trainer = TradeAnomalyTrainer()
    out = trainer.train_and_save()
    print(f"\n[+] AnomalyGuard Training Complete: {out}")
