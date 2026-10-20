import pytest
from src.data_ingestion import apeda_mirror as a

HTML = """<table><tr><th>No</th></tr>
<tr><td><!-- x --> 41/2026-27 </td><td>30 September, 2026</td><td>13 October, 2026</td>
<td>Continuation of RoDTEP</td><td><a href="/sites/default/files/n41.pdf">pdf</a></td></tr>
<tr><td>36/2026-27</td><td>15 September, 2026</td><td>25 September, 2026</td><td>De minimis</td>
<td><a href="/sites/default/files/n36.pdf">pdf</a></td></tr></table>"""


def test_parse_listing_keeps_dates_and_url():
    rows = a.parse_listing(HTML, "Notification")
    assert rows[0]["notification_no"] == "41/2026-27"
    assert rows[0]["date"] == "30/09/2026" and rows[0]["uploaded_date"] == "13/10/2026"
    assert rows[0]["pdf_url"] == "https://apeda.gov.in/sites/default/files/n41.pdf"


def test_empty_listing_fails_closed():
    with pytest.raises(RuntimeError):
        a.parse_listing("<html>blocked</html>", "Notification")


def test_only_gap_rows_selected():
    rows = a.parse_listing(HTML, "Notification")
    gap = a.missing_rows(rows, {("36/2026-27", "15/09/2026")})
    assert [r["notification_no"] for r in gap] == ["41/2026-27"]


def test_record_carries_provenance():
    row = a.parse_listing(HTML, "Notification")[0]
    rec = a.build_record(row, "text")
    assert rec["source"] == "APEDA_MIRROR_OF_DGFT" and rec["mirror_uploaded_date"] == "13/10/2026"
    assert "lag" in rec["source_note"] and rec["chunk_id"].startswith("APEDA_")


def test_same_title_under_different_number_is_not_a_gap():
    rows = a.parse_listing(HTML, "Notification")
    gap = a.missing_rows(rows, set(), ["continuation of rodtep"])
    assert [r["notification_no"] for r in gap] == ["36/2026-27"]


def test_ocr_scans_are_quarantined():
    row = a.parse_listing(HTML, "Notification")[0]
    good, bad = a.split_by_ocr([a.build_record(row, "t", ocr=False), a.build_record(row, "t", ocr=True)])
    assert len(good) == 1 and not good[0]["ocr_used"] and len(bad) == 1 and bad[0]["ocr_warning"]


def test_garbled_text_layer_fails_header_check():
    row = a.parse_listing(HTML, "Notification")[0]
    assert a.header_matches_listing(row, "Notification No: 41/2026-27 Dated 30th")
    assert not a.header_matches_listing(row, "Notification No: \\ I tZOZe-Zl")
