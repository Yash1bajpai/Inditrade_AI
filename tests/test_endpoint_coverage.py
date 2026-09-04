"""
Coverage for the endpoints that shipped with zero tests (audit TEST-01):
forecast GET routes, anomaly historical, network, and the app health check.

All data access is monkeypatched via the shared patch_parquet fixture, so
these tests run on a bare checkout with no parquet present.
"""
import pytest


@pytest.fixture
def full_client(patch_parquet, monkeypatch):
    """Full app TestClient with stubbed models and clean caches."""
    from src.backend.api import forecast as forecast_mod
    from src.backend.api import anomaly as anomaly_mod
    from src.backend.main import app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        forecast_mod,
        "xgboost_model",
        {"model": StubXGBModel(), "features": ["usdinr_mean", "brent_crude_mean", "period"],
         "meta": {"metrics": {}}},
        raising=False,
    )
    monkeypatch.setattr(forecast_mod, "combo_cache", None, raising=False)
    monkeypatch.setattr(forecast_mod, "load_model", lambda: None)
    monkeypatch.setattr(anomaly_mod, "anomaly_model", None, raising=False)
    return TestClient(app, raise_server_exceptions=False)


class StubXGBModel:
    def __init__(self):
        self._features = ["usdinr_mean", "brent_crude_mean", "period"]

    def predict(self, df):
        return [0.5]

    feature_importances_ = [0.4, 0.3, 0.3]


# --- forecast GET endpoints -------------------------------------------------

def test_valid_combinations_builds_map(full_client):
    resp = full_client.get("/api/forecast/valid_combinations")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("partners"), list)
    assert isinstance(body.get("map"), dict)
    # Russia (643) traded >$50M of commodity 27 across >=3 years in the fixture
    codes = [p["code"] for p in body["partners"]]
    assert "643" in codes
    assert "27" in body["map"]["643"]


def test_partner_signature_returns_top_commodities(full_client):
    resp = full_client.get("/api/forecast/partner_signature?partner_code=643")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert all("name" in item and "value_billions" in item for item in body)


def test_partner_signature_accepts_alias(full_client):
    resp = full_client.get("/api/forecast/partner_signature?partner_code=russia")
    assert resp.status_code == 200
    assert resp.json() != []


def test_history_returns_recent_years(full_client):
    resp = full_client.get("/api/forecast/history?partner_code=643&commodity_code=27")
    assert resp.status_code == 200
    history = resp.json()["history"]
    assert len(history) > 0
    assert all("year" in h and "value" in h for h in history)


def test_history_missing_dataset_is_503(full_client, missing_parquet):
    resp = full_client.get("/api/forecast/history")
    assert resp.status_code == 503


def test_country_series_shape(full_client):
    resp = full_client.get("/api/forecast/country_series?partner_code=643")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"yearly", "top_commodities"}
    assert len(body["yearly"]) > 0


def test_year_breakdown_validates_input(full_client):
    resp = full_client.get("/api/forecast/year_breakdown?year=1850&group_by=partner")
    assert resp.status_code == 422
    resp = full_client.get("/api/forecast/year_breakdown?year=2022&group_by=bogus")
    assert resp.status_code == 422


def test_year_breakdown_returns_rows(full_client):
    resp = full_client.get("/api/forecast/year_breakdown?year=2022&group_by=partner")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert any(row["code"] == "643" for row in body)


# --- anomaly historical ------------------------------------------------------

def test_anomaly_historical_shape(full_client, monkeypatch):
    import pandas as pd

    def fake_load_csv(path, **kw):
        return pd.DataFrame([
            {"period": 2022, "primaryValue": 1.0e9, "primaryValue_rolling_3y_mean": 5.0e8,
             "anomaly_score": -0.9, "cmdDesc": "Mineral Fuels", "partnerDesc": "Russia"},
            {"period": 2021, "primaryValue": 2.0e9, "primaryValue_rolling_3y_mean": None,
             "anomaly_score": -0.5, "cmdDesc": None, "partnerDesc": "China"},
        ])

    from src.backend.api import anomaly as anomaly_mod
    monkeypatch.setattr(anomaly_mod, "load_csv", fake_load_csv)
    resp = full_client.get("/api/anomaly/historical")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["table_data"]) == 2
    # NaN baseline must route to the no_baseline reason, not crash
    reasons = {row["reason_code"] for row in body["table_data"]}
    assert "no_baseline" in reasons
    # NaN cmdDesc must fall back to the HS-code map
    commodities = {row["commodity"] for row in body["table_data"]}
    assert any("Machinery" in c or "HS Code" in c for c in commodities)


# --- health ------------------------------------------------------------------

def test_health_reports_degraded_without_model(full_client, monkeypatch):
    """On a bare checkout (no parquet / pkl) health must degrade, not crash."""
    resp = full_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert body["service"]


# --- network ------------------------------------------------------------------

def test_network_history_for_known_country(full_client):
    resp = full_client.get("/api/network/history/Russia")
    assert resp.status_code == 200
    body = resp.json()
    assert "history" in body


def test_network_history_unknown_country_empty(full_client):
    resp = full_client.get("/api/network/history/Atlantis")
    assert resp.status_code == 200
    assert resp.json()["history"] == []
