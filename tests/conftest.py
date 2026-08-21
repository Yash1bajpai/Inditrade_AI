"""
Shared pytest fixtures for the IndiTrade AI backend test suite.

These tests are deliberately hermetic: no Qdrant, no Groq, no Hugging Face, no
real model pickles, no real parquet. Every external dependency is monkeypatched
so the suite runs on a bare checkout with only requirements.txt installed.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def reset_data_cache():
    """Keep the mtime-based parquet/csv cache from leaking frames across tests."""
    from src.utils import data_cache

    data_cache.clear_cache()
    yield
    data_cache.clear_cache()


@pytest.fixture
def trade_df():
    """A minimal but realistic trade_features frame.

    Partner 643 (Russia) x commodity 27 (Mineral Fuels) is the "known good" pair.
    Partner 643 x commodity 09 exists but is negligible (<10k USD), which is the
    case the forecast pre-flight check is supposed to reject with a 400.
    """
    return pd.DataFrame(
        [
            {"partnerCode": "643", "cmdCode": "27", "period": 2020, "primaryValue": 5.0e9,
             "flowCode": "X", "partnerDesc": "Russian Federation", "usdinr_mean": 74.0,
             "brent_crude_mean": 41.0},
            {"partnerCode": "643", "cmdCode": "27", "period": 2021, "primaryValue": 6.0e9,
             "flowCode": "X", "partnerDesc": "Russian Federation", "usdinr_mean": 74.5,
             "brent_crude_mean": 70.0},
            {"partnerCode": "643", "cmdCode": "27", "period": 2022, "primaryValue": 7.5e9,
             "flowCode": "X", "partnerDesc": "Russian Federation", "usdinr_mean": 78.6,
             "brent_crude_mean": 99.0},
            {"partnerCode": "643", "cmdCode": "09", "period": 2022, "primaryValue": 500.0,
             "flowCode": "X", "partnerDesc": "Russian Federation", "usdinr_mean": 78.6,
             "brent_crude_mean": 99.0},
            {"partnerCode": "842", "cmdCode": "30", "period": 2021, "primaryValue": 3.0e9,
             "flowCode": "X", "partnerDesc": "USA", "usdinr_mean": 74.5,
             "brent_crude_mean": 70.0},
            {"partnerCode": "842", "cmdCode": "30", "period": 2022, "primaryValue": 4.0e9,
             "flowCode": "X", "partnerDesc": "USA", "usdinr_mean": 78.6,
             "brent_crude_mean": 99.0},
            {"partnerCode": "842", "cmdCode": "30", "period": 2023, "primaryValue": 4.4e9,
             "flowCode": "X", "partnerDesc": "USA", "usdinr_mean": 82.0,
             "brent_crude_mean": 82.0},
        ]
    )


@pytest.fixture
def patch_parquet(monkeypatch, trade_df):
    """Route every pd.read_parquet call to the in-memory fixture frame."""
    def fake_read_parquet(path, *args, **kwargs):
        return trade_df.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    return trade_df


@pytest.fixture
def missing_parquet(monkeypatch):
    """Simulate a missing/corrupt parquet file."""
    def raise_missing(path, *args, **kwargs):
        raise FileNotFoundError(f"No such file: {path}")

    monkeypatch.setattr(pd, "read_parquet", raise_missing)


class StubXGBModel:
    """Stands in for a fitted XGBRegressor without importing xgboost."""

    def __init__(self, features, prediction_log=22.0):
        self._features = features
        self._prediction_log = prediction_log
        self.feature_importances_ = [1.0 / len(features)] * len(features)

    def predict(self, df):
        return [self._prediction_log]


@pytest.fixture
def forecast_module(monkeypatch, patch_parquet):
    """The forecast router with a working stub model and a clean module state."""
    from src.backend.api import forecast as forecast_mod

    features = ["usdinr_mean", "brent_crude_mean", "period"]
    monkeypatch.setattr(
        forecast_mod,
        "xgboost_model",
        {"model": StubXGBModel(features), "features": features,
         "meta": {"metrics": {"test_log_scale_r2": 0.87}}},
        raising=False,
    )
    monkeypatch.setattr(forecast_mod, "combo_cache", None, raising=False)
    monkeypatch.setattr(forecast_mod, "load_model", lambda: None)
    return forecast_mod


@pytest.fixture
def client(forecast_module):
    """A TestClient over just the forecast router, mounted at the real prefix."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(forecast_module.router, prefix="/api/forecast")
    return TestClient(app, raise_server_exceptions=False)
