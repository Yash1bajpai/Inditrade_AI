import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
from src.data_ingestion import pib_scraper as pib
from src.data_ingestion import forex_macro_fetcher as macro
from src.feature_engineering.trade_features import load_and_aggregate_macro

def test_pib_failure_preserves_corpus(tmp_path, monkeypatch):
    p = tmp_path / 'pib.jsonl'; p.write_text('{"prid":"existing"}\n')
    monkeypatch.setattr(pib,'OUTPUT_JSONL',str(p))
    monkeypatch.setattr(pib,'PROCESSED_DIR',str(tmp_path))
    monkeypatch.setattr(pib,'collect_candidate_prids',lambda: [])
    monkeypatch.setattr(pib,'scrape_full_articles',lambda *a,**k: [])
    with pytest.raises(RuntimeError): pib.main()
    assert p.read_text() == '{"prid":"existing"}\n'

def test_market_failure_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(macro,'OUTPUT_DIR',str(tmp_path))
    monkeypatch.setattr(macro,'TICKERS',{'USDINR':'USDINR=X'})
    monkeypatch.setattr(macro.yf,'download',lambda *a,**k: pd.DataFrame())
    with pytest.raises(RuntimeError): macro.fetch_and_verify_all()

def test_macro_includes_last_completed_year(tmp_path):
    y = pd.Timestamp.today().year - 1
    pd.DataFrame({'Date':[f'{y}-01-01',f'{y}-12-01'],'Close':[80.,82.]}).to_csv(tmp_path/'usdinr.csv',index=False)
    out=load_and_aggregate_macro(str(tmp_path))
    assert y in out.period.values

def test_broken_cny_not_requested():
    assert 'CNYINR' not in macro.TICKERS

def test_comtrade_append_preserves_historical_splits(tmp_path, monkeypatch):
    from src.data_ingestion import un_downloader as un
    p=tmp_path/'trade.parquet'
    original=pd.DataFrame({'period':['2024','2024'],'partnerCode':[842,842],'cmdCode':['01','01'],'flowCode':['M','M'],'primaryValue':[1.,2.],'customsCode':['A','B']})
    original.to_parquet(p,index=False)
    new=pd.DataFrame({'period':['2025'],'partnerCode':[251],'cmdCode':['01'],'flowCode':['M'],'primaryValue':[3.],'customsCode':['A']})
    class Fetcher:
        def fetch_slice(self,**kwargs): return new.copy(), 'SUCCESS'
    monkeypatch.setattr(un,'OUTPUT_PARQUET',str(p))
    monkeypatch.setattr(un,'ComtradeFetcher',Fetcher)
    monkeypatch.setattr(un,'TOP_20_PARTNERS',{'251':'France'})
    monkeypatch.setattr(un.time,'sleep',lambda *a:None)
    un.run_refresh_loop(2025)
    output=pd.read_parquet(p)
    pd.testing.assert_frame_equal(output.iloc[:2].reset_index(drop=True),original)


def test_comtrade_refresh_aligns_new_numeric_dtypes_and_preserves_existing(tmp_path, monkeypatch):
    from src.data_ingestion import un_downloader as un
    p=tmp_path/'trade.parquet'
    original=pd.DataFrame({'period':['2024','2024'],'partnerCode':[842,842],'cmdCode':['01','02'],'flowCode':['M','M'],
                           'primaryValue':[1.,2.],'cifvalue':[5.,None],'qtyUnitCode':[8,8]})
    original.to_parquet(p,index=False)
    # new slice: cifvalue all-null object, numeric strings, whole-number floats
    new=pd.DataFrame({'period':['2025','2025'],'partnerCode':[251,251],'cmdCode':['01','02'],'flowCode':['M','M'],
                      'primaryValue':['3.5',4.],'cifvalue':[None,None],'qtyUnitCode':[8.0,8.0]})
    class Fetcher:
        def fetch_slice(self,**kwargs): return new.copy(), 'SUCCESS'
    monkeypatch.setattr(un,'OUTPUT_PARQUET',str(p))
    monkeypatch.setattr(un,'ComtradeFetcher',Fetcher)
    monkeypatch.setattr(un,'TOP_20_PARTNERS',{'251':'France'})
    monkeypatch.setattr(un.time,'sleep',lambda *a:None)
    un.run_refresh_loop(2025)
    out=pd.read_parquet(p)
    pd.testing.assert_frame_equal(out.iloc[:2].reset_index(drop=True),original)
    assert len(out)==4
    assert out['cifvalue'].dtype==original['cifvalue'].dtype
    assert out.iloc[2:]['primaryValue'].tolist()==[3.5,4.0]
    assert out.iloc[2:]['cifvalue'].isna().all()

def test_comtrade_refresh_rejects_non_numeric_values_and_keeps_file(tmp_path, monkeypatch):
    from src.data_ingestion import un_downloader as un
    p=tmp_path/'trade.parquet'
    original=pd.DataFrame({'period':['2024'],'partnerCode':[842],'cmdCode':['01'],'flowCode':['M'],'primaryValue':[1.],'cifvalue':[5.]})
    original.to_parquet(p,index=False)
    before=p.read_bytes()
    new=pd.DataFrame({'period':['2025'],'partnerCode':[251],'cmdCode':['01'],'flowCode':['M'],'primaryValue':[3.],'cifvalue':['n/a']})
    class Fetcher:
        def fetch_slice(self,**kwargs): return new.copy(), 'SUCCESS'
    monkeypatch.setattr(un,'OUTPUT_PARQUET',str(p))
    monkeypatch.setattr(un,'ComtradeFetcher',Fetcher)
    monkeypatch.setattr(un,'TOP_20_PARTNERS',{'251':'France'})
    monkeypatch.setattr(un.time,'sleep',lambda *a:None)
    with pytest.raises(ValueError, match='Non-numeric'):
        un.run_refresh_loop(2025)
    assert p.read_bytes()==before
