import numpy as np
import pandas as pd
import pytest
from src.feature_engineering.forecast_features import prepare_forecast_frame


def sample():
    return pd.DataFrame({'partnerCode': [842]*4, 'cmdCode': ['27']*4,
                         'flowCode': ['M']*4, 'period': [2019, 2020, 2022, 2023],
                         'partner2Code': [0]*4, 'primaryValue': [100., 120., 150., 180.],
                         'netWgt': [1., 2., 3., 4.], 'usdinr_mean': [70., 72., 80., 82.],
                         'primaryValue_yoy_growth_rate': [0., .2, .25, .2]})


def test_current_target_and_macro_do_not_change_current_predictors():
    f = sample()
    X, *_ = prepare_forecast_frame(f)
    f.loc[3, ['primaryValue', 'netWgt', 'usdinr_mean', 'primaryValue_yoy_growth_rate']] = [999., 999., 999., 999.]
    changed, *_ = prepare_forecast_frame(f)
    pd.testing.assert_series_equal(X.iloc[3], changed.iloc[3])
    assert 'primaryValue_yoy_growth_rate' not in X
    assert 'netWgt' not in X
    assert 'usdinr_mean' not in X
    assert X.iloc[3].usdinr_mean_lag_1y == 80.


def test_calendar_lags_do_not_bridge_a_missing_year():
    X, *_ = prepare_forecast_frame(sample())
    assert np.isnan(X.loc[2, 'primaryValue_lag_1y'])
    assert X.loc[3, 'primaryValue_lag_1y'] == 150.
    assert X.loc[3, 'primaryValue_lag_3y'] == 120.


def test_non_total_partner2_is_not_a_bilateral_total():
    f = sample()
    f.loc[2, 'partner2Code'] = 36
    X, _, _, _, df, excluded = prepare_forecast_frame(f)
    assert excluded == 1
    assert 2022 not in df.period.values
    assert np.isnan(X.loc[df.period.eq(2023), 'primaryValue_lag_1y']).all()


def test_ambiguous_duplicate_totals_fail():
    f = sample()
    with pytest.raises(ValueError, match='duplicate'):
        prepare_forecast_frame(pd.concat([f, f.iloc[[0]]], ignore_index=True))


def test_fetch_requests_totals_and_rejects_second_partner_slice(monkeypatch):
    from src.data_ingestion import un_downloader as un
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({'period': [2023], 'partnerCode': [842], 'flowCode': ['M'],
                             'cmdCode': ['27'], 'partner2Code': [36],
                             'customsCode': ['C00'], 'motCode': [0]})
    monkeypatch.setattr(un.comtradeapicall, 'getFinalData', fake)
    monkeypatch.setattr(un, "get_api_keys", lambda: ["test-only-placeholder"])
    fetcher = un.ComtradeFetcher()
    fetcher.keys = ['test-only-placeholder']
    fetcher.current_key_idx = 0
    df, status = fetcher.fetch_slice('842', 2023, 'M')
    assert df.empty and status.startswith('ERROR:')
    assert calls[0]['partner2Code'] == '0'
    assert calls[0]['customsCode'] == 'C00'
    assert calls[0]['motCode'] == '0'


def test_feature_build_rejects_mixed_grain_before_aggregation(monkeypatch):
    from src.feature_engineering import trade_features as tf
    monkeypatch.setattr(tf, 'load_and_aggregate_macro', lambda *a: pd.DataFrame({'period': [2023]}))
    f = sample().assign(partnerDesc='USA')
    f.loc[2, 'partner2Code'] = 36
    monkeypatch.setattr(pd, 'read_parquet', lambda *a: f.copy())
    with pytest.raises(ValueError, match='partner2'):
        tf.build_trade_features()


def test_legacy_forecast_score_is_not_returned_as_accuracy(forecast_module):
    from src.backend.api.forecast import ForecastRequest
    forecast_module.xgboost_model['features'].append('primaryValue_yoy_growth_rate')
    result = forecast_module.get_forecast(ForecastRequest(usd_inr=83.5, crude_price=80, year=2025, partner_code='643', commodity_code='27'))
    assert result['metrics'] == {}
    assert result['validation_status'] == 'legacy_model_leakage_not_validated'


def test_precomputed_macro_lags_are_rebuilt_and_preparation_is_idempotent():
    f = sample()
    original, *_, prepared, _ = prepare_forecast_frame(f)
    prepared['usdinr_mean_lag_1y'] = 999999.
    rebuilt, *_, prepared_again, _ = prepare_forecast_frame(prepared)
    pd.testing.assert_frame_equal(original, rebuilt)
    twice, *_ = prepare_forecast_frame(prepared_again)
    pd.testing.assert_frame_equal(rebuilt, twice)
    assert not any(c.endswith(('_x', '_y')) for c in prepared_again)
