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




def _vetted(rec, sha, text, **over):
    import hashlib
    e = {"chunk_id": rec["chunk_id"], "pdf_sha256": sha, "pages": [1], "first_pass": "x",
         "corrections": [{"find": "t3xt", "replace": "text", "count": 1, "reason": "r"}],
         "transcription_sha256": hashlib.sha256(text.encode()).hexdigest(),
         "second_pass": {"by": "b", "method": "m", "verified": True}}
    e.update(over)
    return {rec["chunk_id"]: e}


def _rec():
    row = a.parse_listing(HTML, "Notification")[0]
    return a.build_record(row, "t3xt", ocr=False)


def test_default_is_quarantine_even_with_clean_header_and_no_ocr():
    rec = _rec()
    good, bad = a.split_by_vetting([rec], {rec["chunk_id"]: "h"}, {})
    assert not good and bad[0]["verification"] == "quarantined"


def test_vetted_entry_makes_record_eligible_with_corrected_text():
    rec = _rec()
    good, bad = a.split_by_vetting([rec], {rec["chunk_id"]: "h"}, _vetted(rec, "h", "text"))
    assert len(good) == 1 and good[0]["clean_text"] == "text" and not bad


def test_hash_mismatch_unverified_second_pass_and_drift_are_rejected():
    rec = _rec()
    h = {rec["chunk_id"]: "h"}
    assert a.split_by_vetting([rec], h, _vetted(rec, "other", "text"))[0] == []
    assert a.split_by_vetting([rec], h, _vetted(rec, "h", "text", second_pass={"by": "b", "method": "m", "verified": False}))[0] == []
    assert a.split_by_vetting([rec], h, _vetted(rec, "h", "text", transcription_sha256="0" * 64))[0] == []
    drift = {**rec, "clean_text": "something else"}
    assert a.split_by_vetting([drift], h, _vetted(rec, "h", "text"))[0] == []


def test_correction_count_must_match():
    import pytest
    with pytest.raises(ValueError):
        a.apply_corrections("aa", [{"find": "a", "replace": "b", "count": 1, "reason": ""}])
