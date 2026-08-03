"""
Tests for the utility layer fixes: atomic JSONL writes and the scraper helper
that resolves the NameError in dgft_pib_scraper.py.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.file_io import read_jsonl, write_jsonl
from src.utils import scraping_utils


def test_write_then_read_roundtrip(tmp_path):
    target = tmp_path / "qa.jsonl"
    rows = [{"q": "duty on cotton?", "a": "5%"}, {"q": "FTP year?", "a": "2023"}]

    write_jsonl(str(target), rows)

    assert read_jsonl(str(target)) == rows


def test_write_jsonl_preserves_non_ascii(tmp_path):
    """ensure_ascii=False must survive the atomic-write refactor."""
    target = tmp_path / "unicode.jsonl"
    write_jsonl(str(target), [{"a": "₹ 1,20,000 — Mineral Fuels"}])

    text = target.read_text(encoding="utf-8")
    assert "₹" in text and "—" in text


def test_write_jsonl_leaves_original_intact_on_failure(tmp_path):
    """AUDIT REGRESSION: the old plain-'w' write truncated the dataset on crash.

    A row that json.dumps cannot serialise raises partway through. With atomic
    writes the original file must still hold its previous contents.
    """
    target = tmp_path / "qa.jsonl"
    original = [{"q": "kept", "a": "yes"}]
    write_jsonl(str(target), original)

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        write_jsonl(str(target), [{"q": "new"}, {"bad": Unserialisable()}])

    assert read_jsonl(str(target)) == original, \
        "original dataset was clobbered by a failed write"


def test_write_jsonl_leaves_no_temp_files(tmp_path):
    """The temp file must be cleaned up on both success and failure."""
    target = tmp_path / "qa.jsonl"
    write_jsonl(str(target), [{"a": 1}])

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        write_jsonl(str(target), [{"bad": Unserialisable()}])

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_read_jsonl_missing_file_returns_none(tmp_path):
    assert read_jsonl(str(tmp_path / "nope.jsonl")) is None


def test_read_jsonl_skips_blank_lines(tmp_path):
    target = tmp_path / "gaps.jsonl"
    target.write_text('{"a":1}\n\n   \n{"a":2}\n', encoding="utf-8")
    assert read_jsonl(str(target)) == [{"a": 1}, {"a": 2}]


HTML = b"""
<html><body>
  <table><tr><td>row one</td></tr><tr><td>row two</td></tr></table>
  <a href="/one.pdf">One</a><a href="/two.pdf">Two</a>
</body></html>
"""


class FakeResponse:
    def __init__(self, status_code=200, content=HTML):
        self.status_code = status_code
        self.content = content


def test_fetch_page_soup_exists():
    """AUDIT REGRESSION: dgft_pib_scraper.py:154 referenced an undefined `soup`.

    The fallback anchor-scan path needs the parsed document, but fetch_table_rows
    only ever returned <tr> elements.
    """
    assert hasattr(scraping_utils, "fetch_page_soup")


def test_fetch_page_soup_returns_parsed_document(monkeypatch):
    monkeypatch.setattr(scraping_utils.requests, "get", lambda *a, **k: FakeResponse())

    soup = scraping_utils.fetch_page_soup("http://example.test/notices")

    assert soup is not None
    assert len(soup.find_all("a", href=True)) == 2


def test_fetch_page_soup_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(scraping_utils.requests, "get", lambda *a, **k: FakeResponse(status_code=404))
    assert scraping_utils.fetch_page_soup("http://example.test/gone") is None


def test_fetch_page_soup_returns_none_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise scraping_utils.requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(scraping_utils.requests, "get", boom)
    assert scraping_utils.fetch_page_soup("http://example.test/slow") is None


def test_fetch_page_soup_passes_a_timeout(monkeypatch):
    """No unbounded requests.get calls — a hung scrape blocks the whole pipeline."""
    captured = {}

    def capture(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(scraping_utils.requests, "get", capture)
    scraping_utils.fetch_page_soup("http://example.test/x")

    assert captured.get("timeout"), "requests.get was called without a timeout"


def test_fetch_table_rows_still_returns_rows(monkeypatch):
    """The existing caller contract must not change."""
    monkeypatch.setattr(scraping_utils.requests, "get", lambda *a, **k: FakeResponse())
    rows = scraping_utils.fetch_table_rows("http://example.test/notices")
    assert len(rows) == 2


def test_fetch_table_rows_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(scraping_utils.requests, "get", lambda *a, **k: FakeResponse(status_code=500))
    assert scraping_utils.fetch_table_rows("http://example.test/boom") == []
