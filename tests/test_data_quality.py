import pandas as pd
from src.backend.api import forecast
from src.data_ingestion.un_downloader import TOP_20_PARTNERS


def sample():
    return pd.DataFrame([dict(period=2025, partnerCode=p, flowCode=f, partner2Code=0, customsCode='C00', motCode=0) for p in TOP_20_PARTNERS for f in ('M','X')])


def test_quality_passes_only_measured_coverage(monkeypatch):
    monkeypatch.setattr(forecast,'load_parquet',lambda _:sample())
    result=forecast.get_data_quality()
    assert result['status']=='coverage_checks_passed'
    assert result['forecast_validation']=='not_validated'


def test_quality_detects_wrong_transport_and_missing_flow(monkeypatch):
    data=sample().iloc[:-1].copy();data.loc[0,'motCode']=1
    monkeypatch.setattr(forecast,'load_parquet',lambda _:data)
    result=forecast.get_data_quality()
    assert result['status']=='repair_required'
    assert result['mixed_grain_rows']==1
    assert len(result['missing_slices'])==2


def test_quality_failure_is_not_a_pass(monkeypatch):
    def fail(_):raise FileNotFoundError()
    monkeypatch.setattr(forecast,'load_parquet',fail)
    assert forecast.get_data_quality()['status']=='unknown'


def test_canonical_missing_slice_is_partial(monkeypatch):
    data=sample().iloc[:-1].copy()
    monkeypatch.setattr(forecast,'load_parquet',lambda _:data)
    result=forecast.get_data_quality()
    assert result['status']=='partial'
    assert len(result['missing_slices'])==1
    assert result['mixed_grain_rows']==0
    assert result['forecast_validation']=='not_validated'
