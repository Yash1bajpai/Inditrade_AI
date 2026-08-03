"""
Regression tests for the forecast API.

Test 1 and 2 below are the two highest-value cases named in the audit issue: the
unknown-pair path used to raise NameError (undefined get_latest_trade_value) and
surface as an HTTP 500, making the documented 400 + suggested_commodities
contract permanently unreachable.
"""
import pytest


def test_get_latest_trade_value_returns_latest_period(forecast_module):
    """The helper the 400 path depends on must exist and pick the newest period."""
    assert hasattr(forecast_module, "get_latest_trade_value"), \
        "get_latest_trade_value is not defined — the forecast 400 path is dead code"

    # 643/27 has rows for 2020 (5e9), 2021 (6e9), 2022 (7.5e9) -> newest wins.
    assert forecast_module.get_latest_trade_value("643", "27") == pytest.approx(7.5e9)


def test_get_latest_trade_value_unknown_pair_is_zero(forecast_module):
    """An unseen pair returns 0.0 rather than raising."""
    assert forecast_module.get_latest_trade_value("999", "88") == 0.0


def test_get_latest_trade_value_survives_missing_parquet(monkeypatch, forecast_module, missing_parquet):
    """A missing dataset must not propagate an exception into the request path."""
    assert forecast_module.get_latest_trade_value("643", "27") == 0.0


def test_unknown_pair_returns_400_not_500(client):
    """AUDIT REGRESSION: this previously 500'd via NameError."""
    resp = client.post(
        "/api/forecast/",
        json={"usd_inr": 83.0, "crude_price": 85.0, "year": 2025,
              "partner_code": "643", "commodity_code": "09"},
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "suggested_commodities" in body, \
        "the suggested_commodities payload the frontend renders was never returned"
    assert "detail" in body


def test_unknown_pair_suggestions_are_populated(client):
    """The suggestion list built for a known partner must actually reach the client."""
    resp = client.post(
        "/api/forecast/",
        json={"usd_inr": 83.0, "crude_price": 85.0, "year": 2025,
              "partner_code": "643", "commodity_code": "09"},
    )
    assert resp.status_code == 400
    suggested = resp.json()["suggested_commodities"]
    assert isinstance(suggested, list)
    # 643's only qualifying commodity in the fixture is 27 (Mineral Fuels).
    assert any(s["code"] == "27" for s in suggested), suggested


def test_valid_pair_returns_forecast(client):
    """The happy path still works after the pre-flight fix."""
    resp = client.post(
        "/api/forecast/",
        json={"usd_inr": 83.0, "crude_price": 85.0, "year": 2025,
              "partner_code": "643", "commodity_code": "27"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert body["forecasted_trade_value_usd"] > 0
    assert body["year"] == 2025


def test_missing_model_returns_503(monkeypatch, forecast_module, client):
    """A failed/corrupt model pickle must be a 503, not a 500 stack trace."""
    monkeypatch.setattr(forecast_module, "xgboost_model", "FAILED", raising=False)
    resp = client.post(
        "/api/forecast/",
        json={"usd_inr": 83.0, "crude_price": 85.0, "year": 2025,
              "partner_code": "643", "commodity_code": "27"},
    )
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_history_missing_dataset_returns_503(monkeypatch, forecast_module, client, missing_parquet):
    """AUDIT: {'history': []} used to conflate 'no rows' with 'parquet missing'."""
    resp = client.get("/api/forecast/history")
    assert resp.status_code == 503, f"expected 503, got {resp.status_code}: {resp.text}"


def test_history_valid_query_returns_rows(client):
    """A real query returns real history, so the 503 above is not a blanket failure."""
    resp = client.get("/api/forecast/history?partner_code=643&commodity_code=27")
    assert resp.status_code == 200
    history = resp.json()["history"]
    assert len(history) == 3
    assert history[-1]["year"] == "2022"


def test_history_no_match_returns_empty_list(client):
    """No matching rows is a valid empty result, distinct from the 503 above."""
    resp = client.get("/api/forecast/history?partner_code=999&commodity_code=88")
    assert resp.status_code == 200
    assert resp.json() == {"history": []}


def test_year_breakdown_rejects_invalid_group_by(client):
    """AUDIT: silently returned [] instead of signalling bad input."""
    resp = client.get("/api/forecast/year_breakdown?year=2022&group_by=banana")
    assert resp.status_code == 422, resp.text
    assert "group_by" in resp.json()["detail"]


def test_year_breakdown_rejects_out_of_range_year(client):
    """AUDIT: accepted any year and returned [] for nonsense values."""
    resp = client.get("/api/forecast/year_breakdown?year=1200&group_by=partner")
    assert resp.status_code == 422, resp.text
    assert "year" in resp.json()["detail"]


def test_year_breakdown_valid_request_succeeds(client):
    resp = client.get("/api/forecast/year_breakdown?year=2022&group_by=partner")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list) and len(rows) >= 1
    assert {"code", "name", "value_billions"} <= set(rows[0])


def test_no_fabricated_metrics_when_meta_missing(monkeypatch):
    """AUDIT: a hardcoded 0.992 R² was shown to the UI when meta JSON was absent.

    Imports the module directly rather than using the forecast_module fixture,
    because that fixture stubs out load_model() and this test needs the real one.
    """
    import builtins
    import joblib
    from src.backend.api import forecast as forecast_mod

    features = ["usdinr_mean", "brent_crude_mean", "period"]

    monkeypatch.setattr(forecast_mod, "xgboost_model", None, raising=False)
    monkeypatch.setattr(
        joblib, "load",
        lambda path: {"model": object(), "features": features},
    )

    real_open = builtins.open

    def no_meta_file(path, *args, **kwargs):
        if "meta" in str(path):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_meta_file)

    forecast_mod.load_model()

    assert forecast_mod.xgboost_model != "FAILED", "model load unexpectedly failed"
    metrics = forecast_mod.xgboost_model["meta"]["metrics"]
    assert metrics == {}, f"expected no fabricated metrics, got {metrics}"
