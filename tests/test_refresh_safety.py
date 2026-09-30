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
