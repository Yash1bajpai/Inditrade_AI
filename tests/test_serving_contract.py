"""Serving-contract tests: inference features must be built for the requested
year from history, never copied from the latest stored row (which shifts every
lag by the gap between the latest year and the target year)."""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.forecast_features import build_inference_frame


def history():
    rows = []
    for year, val in [(2019, 100.), (2020, 120.), (2022, 150.), (2023, 180.)]:
        rows.append({'partnerCode': 842, 'cmdCode': 27, 'flowCode': 'M',
                     'period': year, 'partner2Code': 0, 'primaryValue': val,
                     'usdinr_mean': 70. + year - 2019})
    return pd.DataFrame(rows)


def test_lags_resolve_against_the_target_year():
    out = build_inference_frame(history(), 2024)
    row = out.iloc[0]
    assert row['primaryValue_lag_1y'] == 180.   # 2023 actual, not 2022
    assert np.isnan(row['primaryValue_lag_3y'])  # 2021 is missing; lags never bridge gaps
    assert row['primaryValue_lag_5y'] == 100.   # 2019


def test_rolling_means_use_target_year_lags():
    out = build_inference_frame(history(), 2024)
    row = out.iloc[0]
    # lag_1=180, lag_2=150, lag_3=NaN -> mean of available matches pandas training semantics
    assert row['primaryValue_rolling_3y_mean'] == pytest.approx((180. + 150.) / 2)


def test_macro_lag_comes_from_target_year_minus_one():
    out = build_inference_frame(history(), 2024)
    assert out.iloc[0]['usdinr_mean_lag_1y'] == 74.  # 2023's value


def test_historical_target_year_also_uses_prior_years_only():
    out = build_inference_frame(history(), 2023)
    assert out.iloc[0]['primaryValue_lag_1y'] == 150.


def test_duplicate_grain_fails_closed():
    dup = pd.concat([history(), history().iloc[[-1]]], ignore_index=True)
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        build_inference_frame(dup, 2024)


def test_horizon_beyond_one_year_ahead_is_refused():
    with pytest.raises(ValueError, match='one year ahead'):
        build_inference_frame(history(), 2025)


def test_non_total_partner2_slices_are_excluded():
    f = history()
    f.loc[2, 'partner2Code'] = 36  # 2022 becomes a non-total slice
    out = build_inference_frame(f, 2024)
    assert np.isnan(out.iloc[0]['primaryValue_lag_2y'])


PAST_ONLY_FEATURES = ['period', 'partnerCode', 'cmdCode', 'flow_is_export',
                      'primaryValue_lag_1y', 'primaryValue_lag_3y', 'primaryValue_lag_5y',
                      'primaryValue_rolling_3y_mean', 'primaryValue_rolling_5y_mean',
                      'usdinr_mean_lag_1y', 'brent_crude_mean_lag_1y']


class RecordingModel:
    def __init__(self):
        self.feature_importances_ = [1.0 / len(PAST_ONLY_FEATURES)] * len(PAST_ONLY_FEATURES)
        self.seen = None

    def predict(self, df):
        self.seen = df.iloc[0].to_dict()
        return [22.0]


@pytest.fixture
def past_only_client(monkeypatch, patch_parquet):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.backend.api import forecast as forecast_mod

    model = RecordingModel()
    monkeypatch.setattr(forecast_mod, 'xgboost_model',
                        {'model': model, 'features': PAST_ONLY_FEATURES,
                         'meta': {'metrics': {'test_mae_usd': 1.0},
                                  'feature_contract': 'past_only_calendar_lags_v1'}},
                        raising=False)
    monkeypatch.setattr(forecast_mod, 'combo_cache', None, raising=False)
    monkeypatch.setattr(forecast_mod, 'load_model', lambda: None)
    app = FastAPI()
    app.include_router(forecast_mod.router, prefix='/api/forecast')
    return TestClient(app, raise_server_exceptions=False), model


def test_endpoint_feeds_target_year_lags_not_template_lags(past_only_client):
    """Fixture 643/27: 2020=5e9, 2021=6e9, 2022=7.5e9. A 2023 request must get
    lag_1y = 7.5e9 (2022 actual); the old template path would feed 6e9."""
    client, model = past_only_client
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2023, 'partner_code': '643',
                                               'commodity_code': '27'})
    assert resp.status_code == 200, resp.text
    assert model.seen['primaryValue_lag_1y'] == pytest.approx(7.5e9)
    assert model.seen['primaryValue_lag_3y'] == pytest.approx(5.0e9)
    assert model.seen['period'] == 2023.0
    assert model.seen['flow_is_export'] == 1.0


def test_endpoint_does_not_feed_scenario_values_to_lag_1y_inputs(past_only_client):
    """The request's usd_inr/crude_price are forecast-year scenario assumptions;
    the past-only contract's only macro slots are lag_1y observations. Mapping
    one onto the other is an unresolved semantics change (held for review), so
    the built lag_1y values must come from history - fixture 2022: 78.6 / 99.0."""
    client, model = past_only_client
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2023, 'partner_code': '643',
                                               'commodity_code': '27'})
    assert resp.status_code == 200, resp.text
    assert model.seen['usdinr_mean_lag_1y'] == pytest.approx(78.6)
    assert model.seen['brent_crude_mean_lag_1y'] == pytest.approx(99.0)


def test_endpoint_keeps_missing_lags_nan_not_zero(past_only_client):
    """842/30 has 2021-2023 only; a 2024 request has no 2019 value for lag_5y.
    The model was trained with NaN there, so serving must pass NaN."""
    client, model = past_only_client
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2024, 'partner_code': '842',
                                               'commodity_code': '30'})
    assert resp.status_code == 200, resp.text
    assert model.seen['primaryValue_lag_1y'] == pytest.approx(4.4e9)
    assert np.isnan(model.seen['primaryValue_lag_5y'])


def test_endpoint_refuses_horizon_beyond_history(past_only_client):
    """Out-of-range year is a client input problem: 422, matching the existing
    year_breakdown convention for out-of-range years."""
    client, _ = past_only_client
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2025, 'partner_code': '643',
                                               'commodity_code': '27'})
    assert resp.status_code == 422, resp.text
    assert 'one year ahead' in resp.json()['detail']


def test_endpoint_serves_export_flow_matching_the_chart_history(monkeypatch, forecast_module):
    """The forecast chart plots the prediction against get_forecast_history,
    which filters flowCode == 'X'. Serve the export series even when the pair's
    most recent actual is an import (legacy latest-row would have picked M)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    frame = pd.DataFrame([
        {'partnerCode': '643', 'cmdCode': '27', 'period': 2020, 'primaryValue': 5.0e9,
         'flowCode': 'X', 'partnerDesc': 'Russian Federation', 'usdinr_mean': 74.0, 'brent_crude_mean': 41.0},
        {'partnerCode': '643', 'cmdCode': '27', 'period': 2021, 'primaryValue': 6.0e9,
         'flowCode': 'X', 'partnerDesc': 'Russian Federation', 'usdinr_mean': 74.5, 'brent_crude_mean': 70.0},
        {'partnerCode': '643', 'cmdCode': '27', 'period': 2022, 'primaryValue': 7.5e9,
         'flowCode': 'X', 'partnerDesc': 'Russian Federation', 'usdinr_mean': 78.6, 'brent_crude_mean': 99.0},
        {'partnerCode': '643', 'cmdCode': '27', 'period': 2023, 'primaryValue': 9.9e9,
         'flowCode': 'M', 'partnerDesc': 'Russian Federation', 'usdinr_mean': 82.0, 'brent_crude_mean': 82.0},
    ])
    monkeypatch.setattr(pd, 'read_parquet', lambda path, *a, **k: frame.copy())
    model = RecordingModel()
    monkeypatch.setattr(forecast_module, 'xgboost_model',
                        {'model': model, 'features': PAST_ONLY_FEATURES,
                         'meta': {'metrics': {}, 'feature_contract': 'past_only_calendar_lags_v1'}},
                        raising=False)
    monkeypatch.setattr(forecast_module, 'combo_cache', None, raising=False)
    app = FastAPI()
    app.include_router(forecast_module.router, prefix='/api/forecast')
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2024, 'partner_code': '643',
                                               'commodity_code': '27'})
    assert resp.status_code == 200, resp.text
    assert model.seen['flow_is_export'] == 1.0
    assert model.seen['primaryValue_lag_3y'] == pytest.approx(6.0e9)  # 2021 export
    assert np.isnan(model.seen['primaryValue_lag_1y'])  # no 2023 export actual; NaN, not the import value


def test_endpoint_400_when_pair_has_no_export_history(monkeypatch, forecast_module):
    """A pair traded only as imports cannot produce a forecast consistent with
    the export-only chart history; refuse honestly instead of switching flows."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    frame = pd.DataFrame([
        {'partnerCode': '643', 'cmdCode': '27', 'period': y, 'primaryValue': v,
         'flowCode': 'M', 'partnerDesc': 'Russian Federation', 'usdinr_mean': 74.0, 'brent_crude_mean': 41.0}
        for y, v in [(2020, 5.0e9), (2021, 6.0e9), (2022, 7.5e9)]
    ])
    monkeypatch.setattr(pd, 'read_parquet', lambda path, *a, **k: frame.copy())
    model = RecordingModel()
    monkeypatch.setattr(forecast_module, 'xgboost_model',
                        {'model': model, 'features': PAST_ONLY_FEATURES,
                         'meta': {'metrics': {}, 'feature_contract': 'past_only_calendar_lags_v1'}},
                        raising=False)
    monkeypatch.setattr(forecast_module, 'combo_cache', None, raising=False)
    app = FastAPI()
    app.include_router(forecast_module.router, prefix='/api/forecast')
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2023, 'partner_code': '643',
                                               'commodity_code': '27'})
    assert resp.status_code == 400, resp.text
    assert 'export' in resp.json()['detail'].lower()
    assert model.seen is None


def test_legacy_model_with_lag_columns_stays_on_template_path(forecast_module, client):
    """The live legacy model lists primaryValue_lag_1y AND the leak marker
    primaryValue_yoy_growth_rate and declares no feature_contract. It must not
    be routed into the past-only builder: without the contract the legacy
    template path (and its fail-closed absent-feature check) applies."""
    from src.backend.api import forecast as forecast_mod
    legacy_features = ['period', 'primaryValue_lag_1y', 'primaryValue_yoy_growth_rate', 'usdinr_mean']
    model = RecordingModel()
    import pytest as _pt
    forecast_mod.xgboost_model = {'model': model, 'features': legacy_features,
                                  'meta': {'metrics': {}}}
    resp = client.post('/api/forecast/', json={'usd_inr': 83.0, 'crude_price': 85.0,
                                               'year': 2025, 'partner_code': '643',
                                               'commodity_code': '27'})
    # Fixture parquet lacks primaryValue_yoy_growth_rate -> legacy fail-closed 503,
    # proving the request never entered the past-only builder path.
    assert resp.status_code == 503, resp.text
    assert 'not been validated' in resp.json()['detail']
    assert model.seen is None
