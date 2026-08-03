"""
Tests for the anomaly model loader's schema validation.

The audit flagged that load_model() accepted whatever joblib.load() returned and
the request handler then indexed loaded['model'], so a bare-estimator pickle
raised TypeError on every single request instead of failing once at load time.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class GoodEstimator:
    n_features_in_ = 6

    def predict(self, X):
        return [1] * len(X)

    def decision_function(self, X):
        return [0.1] * len(X)


class WrongShapeEstimator:
    n_features_in_ = 3

    def predict(self, X):
        return [1] * len(X)


@pytest.fixture
def anomaly_mod(monkeypatch):
    from src.backend.api import anomaly as mod
    monkeypatch.setattr(mod, "anomaly_model", None, raising=False)
    return mod


def _patch_joblib(monkeypatch, payload):
    import joblib
    monkeypatch.setattr(joblib, "load", lambda path: payload)


def test_valid_dict_pickle_loads(monkeypatch, anomaly_mod):
    _patch_joblib(monkeypatch, {"model": GoodEstimator(), "features": anomaly_mod.ANOMALY_FEATURES})
    anomaly_mod.load_model()
    assert anomaly_mod.anomaly_model != "FAILED"
    assert "model" in anomaly_mod.anomaly_model


def test_bare_estimator_pickle_is_rejected(monkeypatch, anomaly_mod):
    """AUDIT REGRESSION: previously accepted, then TypeError'd on every request."""
    _patch_joblib(monkeypatch, GoodEstimator())
    anomaly_mod.load_model()
    assert anomaly_mod.anomaly_model == "FAILED", \
        "a bare estimator pickle must be rejected at load time, not per-request"


def test_dict_without_model_key_is_rejected(monkeypatch, anomaly_mod):
    _patch_joblib(monkeypatch, {"features": ["a", "b"], "threshold": 0.5})
    anomaly_mod.load_model()
    assert anomaly_mod.anomaly_model == "FAILED"


def test_model_without_predict_is_rejected(monkeypatch, anomaly_mod):
    _patch_joblib(monkeypatch, {"model": object()})
    anomaly_mod.load_model()
    assert anomaly_mod.anomaly_model == "FAILED"


def test_feature_count_mismatch_is_rejected(monkeypatch, anomaly_mod):
    """A model trained on a different feature set must not be silently served."""
    _patch_joblib(monkeypatch, {"model": WrongShapeEstimator()})
    anomaly_mod.load_model()
    assert anomaly_mod.anomaly_model == "FAILED"


def test_failed_load_surfaces_as_503(monkeypatch, anomaly_mod):
    """A rejected pickle must produce a 503, not a 500 stack trace."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _patch_joblib(monkeypatch, GoodEstimator())  # bare estimator -> FAILED

    app = FastAPI()
    app.include_router(anomaly_mod.router, prefix="/api/anomaly")
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/anomaly/", json={"usd_inr": 83.0, "crude_price": 85.0})
    assert resp.status_code == 503, f"expected 503, got {resp.status_code}: {resp.text}"


def test_features_list_matches_declared_count(anomaly_mod):
    assert len(anomaly_mod.ANOMALY_FEATURES) == 6
