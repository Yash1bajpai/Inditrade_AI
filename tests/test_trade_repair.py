import pandas as pd
import pytest
from scripts.repair_trade_data import plan_repairs, repair
from src.data_ingestion.un_downloader import TOP_20_PARTNERS


def base():
    rows = [{'period':2025, 'partnerCode':int(p), 'flowCode':f, 'cmdCode':'27',
             'reporterCode':699, 'partner2Code':0, 'customsCode':'C00', 'motCode':0, 'primaryValue':100.}
            for p in TOP_20_PARTNERS for f in ['M','X']]
    rows += [{'period':2023, 'partnerCode':842, 'flowCode':'M', 'cmdCode':'27',
              'reporterCode':699, 'partner2Code':36, 'customsCode':'C00', 'motCode':0, 'primaryValue':3573.855}]
    return pd.DataFrame(rows)


def test_plan_finds_mixed_slice_and_missing_latest():
    df=base();df=df[~((df.partnerCode==250)&(df.flowCode=='X'))]
    assert plan_repairs(df)==[(2023,842,'M'),(2025,250,'X')]


def test_full_slice_repair_preserves_unaffected_rows():
    raw=base()
    class Fetcher:
        def count_slice(self,*a): return 1
        def fetch_slice(self,partner,year,flow):
            d=raw.iloc[[-1]].copy();d.partner2Code=0;d.primaryValue=10_000_000_000.
            return d,'SUCCESS'
    out,slices=repair(raw,Fetcher(),pause=lambda _:None)
    assert not plan_repairs(out)
    assert out[out.period==2023].primaryValue.iloc[0]==10_000_000_000.
    assert out[out.period==2025].primaryValue.sum()==raw[raw.period==2025].primaryValue.sum()
    assert raw.iloc[-1].partner2Code==36


def test_failed_fetch_never_changes_input():
    raw=base();copy=raw.copy(deep=True)
    class Fetcher:
        def count_slice(self,*a): return 1
        def fetch_slice(self,*a):return pd.DataFrame(),'ERROR: invalid key'
    with pytest.raises(RuntimeError):repair(raw,Fetcher(),pause=lambda _:None)
    pd.testing.assert_frame_equal(raw,copy)


def test_noncanonical_response_refused():
    raw=base()
    class Fetcher:
        def count_slice(self,*a): return 1
        def fetch_slice(self,*a):return raw.iloc[[-1]].copy(),'SUCCESS'
    with pytest.raises(ValueError,match='non-total'):repair(raw,Fetcher(),pause=lambda _:None)


def test_truncated_response_refused():
    raw=base()
    class Fetcher:
        def count_slice(self,*a):return 2
        def fetch_slice(self,*a):
            d=raw.iloc[[-1]].copy();d.partner2Code=0
            return d,'SUCCESS'
    with pytest.raises(ValueError,match='completeness'):repair(raw,Fetcher(),pause=lambda _:None)
