"""Committed regression coverage for the Comtrade key-rotation fix.

comtradeapicall swallows non-200 responses (prints the body, returns None), so
an invalid key looked like empty data and rotation (403/429-only) never fired.
fetch_slice/count_slice probe the real HTTP status on a non-DataFrame result
and rotate on 401/403/429, failing closed when every key fails.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_ingestion import un_downloader
from src.data_ingestion.un_downloader import ComtradeFetcher


@pytest.fixture
def fetcher(monkeypatch):
    monkeypatch.setattr(un_downloader, "get_api_keys", lambda: ["key_A", "key_B"])
    return ComtradeFetcher()


def _good_df():
    return pd.DataFrame(
        {
            "period": ["2024"],
            "partnerCode": ["842"],
            "flowCode": ["M"],
            "cmdCode": ["27"],
            "partner2Code": ["0"],
            "customsCode": ["C00"],
            "motCode": ["0"],
            "primaryValue": [1_000_000.0],
        }
    )


def test_valid_dataframe_never_probes_or_rotates(fetcher):
    with patch.object(un_downloader.comtradeapicall, "getFinalData", return_value=_good_df()) as g, \
         patch.object(ComtradeFetcher, "_probe_status", side_effect=AssertionError("must not probe")):
        df, status = fetcher.fetch_slice("842", "2024", "M")
    assert status == "SUCCESS" and len(df) == 1
    assert fetcher.get_active_key() == "key_A"
    assert g.call_count == 1


def test_swallowed_401_probes_and_rotates(fetcher):
    calls = {"n": 0}

    def fake_get(**kw):
        calls["n"] += 1
        return None if calls["n"] == 1 else _good_df()

    with patch.object(un_downloader.comtradeapicall, "getFinalData", side_effect=fake_get), \
         patch.object(ComtradeFetcher, "_probe_status", return_value=401):
        df, status = fetcher.fetch_slice("842", "2024", "M")
    assert status == "SUCCESS" and len(df) == 1
    assert fetcher.get_active_key() == "key_B"


def test_swallowed_401_with_all_keys_dead_fails_closed(fetcher):
    with patch.object(un_downloader.comtradeapicall, "getFinalData", return_value=None), \
         patch.object(ComtradeFetcher, "_probe_status", return_value=401):
        df, status = fetcher.fetch_slice("842", "2024", "M")
    assert df is None
    assert status.startswith("AUTH/RATE_EXHAUSTED")


def test_raised_401_exception_also_rotates(fetcher):
    """Regression: the exception path matched 403/429 only; a raised 401 fell
    through to a plain ERROR with no rotation."""
    calls = {"n": 0}

    def fake_get(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("HTTP 401 Unauthorized")
        return _good_df()

    with patch.object(un_downloader.comtradeapicall, "getFinalData", side_effect=fake_get):
        df, status = fetcher.fetch_slice("842", "2024", "M")
    assert status == "SUCCESS" and len(df) == 1
    assert fetcher.get_active_key() == "key_B"


def test_probe_transport_error_fails_closed_not_raw_exception(fetcher):
    """Regression: _probe_status was called without try/except; a network
    failure escaped as a raw urllib3 error instead of a clean failure."""
    with patch.object(un_downloader.comtradeapicall, "getFinalData", return_value=None), \
         patch.object(ComtradeFetcher, "_probe_status", side_effect=Exception("connection reset")):
        df, status = fetcher.fetch_slice("842", "2024", "M")
    assert df is None
    assert "status probe failed" in status


def test_probe_200_reports_honest_status(fetcher):
    with patch.object(un_downloader.comtradeapicall, "getFinalData", return_value=None), \
         patch.object(ComtradeFetcher, "_probe_status", return_value=200):
        df, status = fetcher.fetch_slice("842", "2024", "M")
    assert isinstance(df, pd.DataFrame) and df.empty
    assert status == "ERROR: Comtrade get failed with HTTP 200"


def test_count_slice_rotates_on_401(fetcher):
    count_df = pd.DataFrame({"count": [5]})
    calls = {"n": 0}

    def fake_get(**kw):
        calls["n"] += 1
        return "garbage" if calls["n"] == 1 else count_df

    with patch.object(un_downloader.comtradeapicall, "getFinalData", side_effect=fake_get), \
         patch.object(ComtradeFetcher, "_probe_status", return_value=401):
        assert fetcher.count_slice("842", "2024", "M") == 5
    assert fetcher.get_active_key() == "key_B"


def test_count_slice_probe_error_is_valueerror(fetcher):
    with patch.object(un_downloader.comtradeapicall, "getFinalData", return_value="garbage"), \
         patch.object(ComtradeFetcher, "_probe_status", side_effect=Exception("dns failed")):
        with pytest.raises(ValueError, match="status probe failed"):
            fetcher.count_slice("842", "2024", "M")


def test_count_slice_all_keys_dead_raises(fetcher):
    with patch.object(un_downloader.comtradeapicall, "getFinalData", return_value="garbage"), \
         patch.object(ComtradeFetcher, "_probe_status", return_value=401):
        with pytest.raises(ValueError, match="all API keys failed"):
            fetcher.count_slice("842", "2024", "M")
