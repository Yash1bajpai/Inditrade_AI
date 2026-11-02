"""Regression tests for policy refresh resilience, corpus validation, and source-blocking recovery.

Covers:
1. Blocked listing + existing corpus -> exit 0, corpus bytes UNCHANGED (hash before/after),
   marker has dgft=blocked with reason and timestamp.
2. Blocked listing + NO existing corpus -> raises/fails closed.
3. Successful listing -> download/process called, marker has dgft=ok, no stale carry-over.
4. Marker merge: pre-existing pib entry survives adding a dgft entry (no clobber).
5. Zero-chunk extraction after successful downloads -> still fatal (corruption path not masked),
   and records error in freshness marker before re-raising.
6. Workflow file: yaml parses; the corpus-usability gate exists before publish steps;
   the auto-push step is byte-identical to before this change; calls the shared Python validator.
7. Shared validator coverage: HTML junk, malformed JSONL, error objects, missing keys, empty fields fail;
   real reviewed valid corpus samples pass; CLI/workflow entrypoint fails closed.
8. Non-SourceBlockedError failure in DGFT writes error to freshness marker before re-raising.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from unittest.mock import MagicMock
import pandas as pd
import pytest
import yaml

from src.data_ingestion import dgft_pdf_processor as dgft
from src.data_ingestion.dgft_pdf_processor import (
    SourceBlockedError,
    main as dgft_main,
    record_source_freshness,
)
from src.data_ingestion import pib_scraper as pib
from src.utils.freshness_utils import (
    CorpusValidationError,
    is_usable_corpus,
    read_source_freshness,
    validate_corpus_file,
    validate_gate,
)


def _hash_file(filepath: str) -> str:
    """Returns the SHA256 hex digest of a file's contents."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_valid_dgft_record(chunk_id: str = "DGFT_NOTIFICATION_1", notif_no: str = "01/2024") -> dict:
    """Returns a schema-compliant DGFT policy chunk."""
    return {
        "chunk_id": chunk_id,
        "notification_no": notif_no,
        "type": "Notification",
        "date": "2024-05-10",
        "financial_year": "2024-2025",
        "subject": "Export of agricultural commodities under bilateral trade agreement",
        "clean_text": "Government of India Directorate General of Foreign Trade Notification No 01/2024 regarding foreign trade regulation.",
        "url": "https://www.dgft.gov.in/CP/notif_01.pdf",
    }


def make_valid_pib_record(prid: str = "2001", title: str = "PIB Commerce Release") -> dict:
    """Returns a schema-compliant PIB press release record."""
    return {
        "prid": prid,
        "title": title,
        "date": "2024-05-10",
        "clean_text": "Press Information Bureau Government of India Ministry of Commerce and Industry export growth and economic measures announcement.",
        "category": "Trade & Tariff Policy",
        "ministry": "Ministry of Commerce & Industry",
        "url": f"https://pib.gov.in/PressReleasePage.aspx?PRID={prid}",
    }


def test_blocked_listing_with_existing_corpus_preserves_corpus_and_records_marker(tmp_path):
    """1. Blocked listing + existing corpus -> exit 0, corpus bytes UNCHANGED (hash before/after),
    marker has dgft=blocked with reason and timestamp.
    """
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    rec = make_valid_dgft_record()
    corpus_initial = (json.dumps(rec) + "\n").encode("utf-8")
    corpus_file.write_bytes(corpus_initial)
    hash_before = _hash_file(str(corpus_file))

    marker_file = tmp_path / "source_freshness.json"

    def mock_scrape_blocked():
        raise SourceBlockedError("HTTP 403 Forbidden: Egress blocked by DGFT")

    exit_code = dgft_main(
        scrape_fn=mock_scrape_blocked,
        marker_path=str(marker_file),
        output_jsonl=str(corpus_file),
    )

    assert exit_code == 0, "Blocked listing with existing corpus must exit 0"
    assert _hash_file(str(corpus_file)) == hash_before, "Corpus bytes must remain UNCHANGED"
    assert corpus_file.read_bytes() == corpus_initial, "Corpus content must match original bytes"

    assert marker_file.exists(), "Source freshness marker must be created"
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert "dgft" in marker, "Marker must contain 'dgft' entry"
    assert marker["dgft"]["status"] == "blocked"
    assert "HTTP 403 Forbidden" in marker["dgft"]["reason"]
    assert "blocked_at" in marker["dgft"]
    datetime.fromisoformat(marker["dgft"]["blocked_at"])


def test_blocked_listing_without_existing_corpus_fails_closed(tmp_path):
    """2. Blocked listing + NO existing corpus -> raises/fails closed."""
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    assert not corpus_file.exists()

    marker_file = tmp_path / "source_freshness.json"

    def mock_scrape_blocked():
        raise SourceBlockedError("DGFT listing empty or blocked; refresh failed")

    with pytest.raises(SourceBlockedError) as exc_info:
        dgft_main(
            scrape_fn=mock_scrape_blocked,
            marker_path=str(marker_file),
            output_jsonl=str(corpus_file),
        )

    assert "DGFT listing empty or blocked" in str(exc_info.value)
    assert not corpus_file.exists(), "Never invent an empty or fabricated corpus on failure"


def test_blocked_listing_with_empty_corpus_file_fails_closed(tmp_path):
    """2b. Blocked listing + 0-byte existing corpus -> fails closed."""
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    corpus_file.write_bytes(b"")  # 0 bytes, not usable

    marker_file = tmp_path / "source_freshness.json"

    def mock_scrape_blocked():
        raise SourceBlockedError("DGFT listing empty or blocked; refresh failed")

    with pytest.raises(SourceBlockedError):
        dgft_main(
            scrape_fn=mock_scrape_blocked,
            marker_path=str(marker_file),
            output_jsonl=str(corpus_file),
        )


def test_successful_listing_downloads_processes_and_records_ok_marker(tmp_path):
    """3. Successful listing -> download/process called, marker has dgft=ok, no stale carry-over."""
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    marker_file = tmp_path / "source_freshness.json"

    stale_marker = {
        "dgft": {
            "status": "blocked",
            "reason": "Old 403 Forbidden from yesterday",
            "blocked_at": "2026-11-01T03:15:00Z",
        }
    }
    marker_file.write_text(json.dumps(stale_marker), encoding="utf-8")

    sample_df = pd.DataFrame(
        [
            {
                "Type": "Notification",
                "Notification_No": "42/2024",
                "Financial_Year": "2024-2025",
                "Subject": "Export of onions permitted",
                "Issue_Date": "01.11.2024",
                "PDF_Link": "https://www.dgft.gov.in/test.pdf",
                "Local_PDF_Path": str(tmp_path / "test.pdf"),
            }
        ]
    )

    mock_scrape = MagicMock(return_value=sample_df)
    mock_download = MagicMock(return_value=1)
    mock_process = MagicMock(
        return_value=[
            make_valid_dgft_record(chunk_id="DGFT_NOTIFICATION_42_2024", notif_no="42/2024")
        ]
    )

    exit_code = dgft_main(
        scrape_fn=mock_scrape,
        download_fn=mock_download,
        process_fn=mock_process,
        marker_path=str(marker_file),
        output_jsonl=str(corpus_file),
    )

    assert exit_code == 0
    assert mock_scrape.called
    assert mock_download.called
    assert mock_process.called

    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker["dgft"]["status"] == "ok"
    assert marker["dgft"]["chunk_count"] == 1
    assert "last_refreshed_at" in marker["dgft"]
    assert "reason" not in marker["dgft"], "Stale reason must not carry over into ok marker"
    assert "blocked_at" not in marker["dgft"], "Stale blocked_at must not carry over into ok marker"


def test_marker_merge_preserves_pib_entry(tmp_path):
    """4. Marker merge: pre-existing pib entry survives adding a dgft entry (no clobber)."""
    marker_file = tmp_path / "source_freshness.json"

    initial_marker = {
        "pib": {
            "status": "ok",
            "article_count": 55,
            "last_refreshed_at": "2026-11-02T03:20:00+00:00",
        }
    }
    marker_file.write_text(json.dumps(initial_marker), encoding="utf-8")

    record_source_freshness(
        "dgft",
        {
            "status": "blocked",
            "reason": "HTTP 403 Forbidden to runner egress",
            "blocked_at": "2026-11-02T03:25:00+00:00",
        },
        marker_path=str(marker_file),
    )

    merged = json.loads(marker_file.read_text(encoding="utf-8"))
    assert "pib" in merged, "Pre-existing 'pib' entry must not be clobbered"
    assert merged["pib"]["status"] == "ok"
    assert merged["pib"]["article_count"] == 55
    assert merged["pib"]["last_refreshed_at"] == "2026-11-02T03:20:00+00:00"

    assert "dgft" in merged, "'dgft' entry must be added"
    assert merged["dgft"]["status"] == "blocked"
    assert merged["dgft"]["reason"] == "HTTP 403 Forbidden to runner egress"


def test_zero_chunk_extraction_after_successful_downloads_is_fatal_and_records_marker(tmp_path):
    """5. Zero-chunk extraction after successful downloads -> still fatal (corruption path not masked),
    and records error in freshness marker before re-raising.
    """
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    corpus_file.write_text(json.dumps(make_valid_dgft_record()) + "\n", encoding="utf-8")
    marker_file = tmp_path / "source_freshness.json"

    sample_df = pd.DataFrame(
        [
            {
                "Type": "Notification",
                "Notification_No": "1",
                "Subject": "Corrupt doc",
                "Local_PDF_Path": str(tmp_path / "corrupt.pdf"),
            }
        ]
    )

    mock_scrape = MagicMock(return_value=sample_df)
    mock_download = MagicMock(return_value=2)
    mock_process = MagicMock(return_value=[])

    with pytest.raises(RuntimeError) as exc_info:
        dgft_main(
            scrape_fn=mock_scrape,
            download_fn=mock_download,
            process_fn=mock_process,
            marker_path=str(marker_file),
            output_jsonl=str(corpus_file),
        )

    assert "corruption" in str(exc_info.value).lower() or "no valid dgft pdf text" in str(exc_info.value).lower()
    assert marker_file.exists(), "Failure must record status before re-raising"
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker.get("dgft", {}).get("status") == "failed"
    assert "failed_at" in marker.get("dgft", {})
    assert "reason" in marker.get("dgft", {})


def test_zero_chunk_extraction_after_zero_downloads_is_treated_as_blocked(tmp_path):
    """4b. Zero-chunk extraction following zero downloads is treated as blocked source."""
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    corpus_file.write_text(json.dumps(make_valid_dgft_record()) + "\n", encoding="utf-8")
    marker_file = tmp_path / "source_freshness.json"

    sample_df = pd.DataFrame(
        [
            {
                "Type": "Notification",
                "Notification_No": "1",
                "Subject": "Failed doc",
                "Local_PDF_Path": str(tmp_path / "none.pdf"),
            }
        ]
    )

    mock_scrape = MagicMock(return_value=sample_df)
    mock_download = MagicMock(return_value=0)
    mock_process = MagicMock(return_value=[])

    exit_code = dgft_main(
        scrape_fn=mock_scrape,
        download_fn=mock_download,
        process_fn=mock_process,
        marker_path=str(marker_file),
        output_jsonl=str(corpus_file),
    )

    assert exit_code == 0
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker["dgft"]["status"] == "blocked"


def test_call_time_constant_monkeypatching(tmp_path, monkeypatch):
    """Verifies that constants like OUTPUT_JSONL and FRESHNESS_MARKER_PATH are read at call time."""
    custom_corpus = tmp_path / "custom_corpus.jsonl"
    custom_corpus.write_text(json.dumps(make_valid_dgft_record()) + "\n", encoding="utf-8")
    custom_marker = tmp_path / "custom_marker.json"

    monkeypatch.setattr(dgft, "OUTPUT_JSONL", str(custom_corpus))
    monkeypatch.setattr(dgft, "FRESHNESS_MARKER_PATH", str(custom_marker))

    def mock_blocked():
        raise SourceBlockedError("Blocked")

    exit_code = dgft_main(scrape_fn=mock_blocked)
    assert exit_code == 0
    assert custom_marker.exists()
    marker = json.loads(custom_marker.read_text(encoding="utf-8"))
    assert marker["dgft"]["status"] == "blocked"


def test_pib_scraper_updates_freshness_marker_on_success_and_failure(tmp_path, monkeypatch):
    """Verifies that pib_scraper records ok on success and stale on failure in source_freshness.json."""
    marker_file = tmp_path / "source_freshness.json"
    corpus_file = tmp_path / "pib_press_releases.jsonl"
    corpus_file.write_text(json.dumps(make_valid_pib_record()) + "\n", encoding="utf-8")

    monkeypatch.setattr(pib, "OUTPUT_JSONL", str(corpus_file))
    monkeypatch.setattr(pib, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(pib, "FRESHNESS_MARKER_PATH", str(marker_file))

    # Failure path: scrape returns empty
    monkeypatch.setattr(pib, "collect_candidate_prids", lambda: [])
    monkeypatch.setattr(pib, "scrape_full_articles", lambda *a, **k: [])

    with pytest.raises(RuntimeError):
        pib.main(marker_path=str(marker_file), output_jsonl=str(corpus_file))

    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker["pib"]["status"] == "stale"
    assert "Live PIB scraping returned no valid articles" in marker["pib"]["reason"]

    # Success path: scrape returns valid article
    monkeypatch.setattr(pib, "collect_candidate_prids", lambda: [{"prid": "1001", "title": "New Policy", "source_url": "http://pib"}])
    monkeypatch.setattr(
        pib,
        "scrape_full_articles",
        lambda *a, **k: [
            make_valid_pib_record(prid="1001", title="New Policy on Trade")
        ],
    )

    articles = pib.main(marker_path=str(marker_file), output_jsonl=str(corpus_file))
    assert len(articles) >= 1
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker["pib"]["status"] == "ok"
    assert marker["pib"]["article_count"] >= 1
    assert "reason" not in marker["pib"]


def test_dgft_non_source_blocked_failure_records_freshness_marker_before_reraising(tmp_path):
    """Requirement C: On non-SourceBlockedError failure, write reason + timestamp before re-raising."""
    corpus_file = tmp_path / "dgft_policy_chunks.jsonl"
    corpus_file.write_text(json.dumps(make_valid_dgft_record()) + "\n", encoding="utf-8")
    marker_file = tmp_path / "source_freshness.json"

    def mock_scrape_crash():
        raise ValueError("Unexpected network parser explosion: bad payload format")

    with pytest.raises(ValueError) as exc_info:
        dgft_main(
            scrape_fn=mock_scrape_crash,
            marker_path=str(marker_file),
            output_jsonl=str(corpus_file),
        )

    assert "Unexpected network parser explosion" in str(exc_info.value)
    assert marker_file.exists(), "Failure marker must be recorded before exception bubbles up"
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker["dgft"]["status"] == "failed"
    assert "Unexpected network parser explosion" in marker["dgft"]["reason"]
    assert "failed_at" in marker["dgft"]


def test_validator_rejects_html_error_page(tmp_path):
    """Shared validator rejects raw HTML 403/error pages saved as corpus JSONL."""
    html_file = tmp_path / "dgft_policy_chunks.jsonl"
    html_file.write_text(
        "<!DOCTYPE html>\n<html><head><title>403 Forbidden</title></head><body><h1>403 Forbidden</h1></body></html>\n",
        encoding="utf-8",
    )

    res = validate_corpus_file(str(html_file), source="dgft")
    assert not res["valid"]
    assert not is_usable_corpus(str(html_file), source="dgft")
    assert validate_gate([str(html_file)]) == 1


def test_validator_rejects_malformed_jsonl(tmp_path):
    """Shared validator rejects non-JSON lines / syntax errors."""
    corrupt_file = tmp_path / "pib_press_releases.jsonl"
    valid_line = json.dumps(make_valid_pib_record())
    corrupt_file.write_text(
        f"{valid_line}\n{{not valid json at all}}\n",
        encoding="utf-8",
    )
    res = validate_corpus_file(str(corrupt_file), source="pib")
    assert not res["valid"]
    assert "Not valid JSON" in res["error"]
    assert not is_usable_corpus(str(corrupt_file), source="pib")


def test_validator_rejects_error_objects(tmp_path):
    """Shared validator rejects JSON error response objects masquerading as policy records."""
    error_obj_file = tmp_path / "dgft_policy_chunks.jsonl"
    error_obj_file.write_text(
        '{"error": "Forbidden", "status_code": 403, "message": "Egress runner blocked"}\n',
        encoding="utf-8",
    )

    res = validate_corpus_file(str(error_obj_file), source="dgft")
    assert not res["valid"]
    assert "Missing required schema keys" in res["error"]
    assert not is_usable_corpus(str(error_obj_file), source="dgft")


def test_validator_rejects_missing_keys(tmp_path):
    """Shared validator verifies every required key for DGFT and PIB."""
    # DGFT missing notification_no and financial_year
    dgft_bad = tmp_path / "dgft_policy_chunks.jsonl"
    rec_dgft = make_valid_dgft_record()
    del rec_dgft["notification_no"]
    del rec_dgft["financial_year"]
    dgft_bad.write_text(json.dumps(rec_dgft) + "\n", encoding="utf-8")

    res_dgft = validate_corpus_file(str(dgft_bad), source="dgft")
    assert not res_dgft["valid"]
    assert "Missing required schema keys" in res_dgft["error"]
    assert "notification_no" in res_dgft["error"]

    # PIB missing ministry and category
    pib_bad = tmp_path / "pib_press_releases.jsonl"
    rec_pib = make_valid_pib_record()
    del rec_pib["ministry"]
    del rec_pib["category"]
    pib_bad.write_text(json.dumps(rec_pib) + "\n", encoding="utf-8")

    res_pib = validate_corpus_file(str(pib_bad), source="pib")
    assert not res_pib["valid"]
    assert "Missing required schema keys" in res_pib["error"]
    assert "ministry" in res_pib["error"]


def test_validator_rejects_empty_fields(tmp_path):
    """Shared validator rejects records with empty strings in required fields."""
    bad_file = tmp_path / "dgft_policy_chunks.jsonl"
    rec = make_valid_dgft_record()
    rec["notification_no"] = "   "
    bad_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    res = validate_corpus_file(str(bad_file), source="dgft")
    assert not res["valid"]
    assert "Empty required field" in res["error"]


def test_validator_rejects_trivial_or_html_clean_text(tmp_path):
    """Shared validator rejects trivial text or HTML error strings masquerading in clean_text."""
    # Trivial text (< 20 chars)
    file_short = tmp_path / "short.jsonl"
    rec_short = make_valid_dgft_record()
    rec_short["clean_text"] = "too short"
    file_short.write_text(json.dumps(rec_short) + "\n", encoding="utf-8")

    res_short = validate_corpus_file(str(file_short), source="dgft")
    assert not res_short["valid"]
    assert "clean_text" in res_short["error"]

    # HTML content embedded in clean_text
    file_html_text = tmp_path / "html_text.jsonl"
    rec_html = make_valid_dgft_record()
    rec_html["clean_text"] = "<html><head><title>403 Forbidden</title></head><body>Access Denied</body></html>"
    file_html_text.write_text(json.dumps(rec_html) + "\n", encoding="utf-8")

    res_html = validate_corpus_file(str(file_html_text), source="dgft")
    assert not res_html["valid"]
    assert "HTML/error content" in res_html["error"]


def test_validator_accepts_valid_reviewed_corpus_samples(tmp_path):
    """Shared validator accepts valid reviewed DGFT and PIB corpus records and reports counts."""
    dgft_file = tmp_path / "dgft_policy_chunks.jsonl"
    dgft_records = [
        make_valid_dgft_record(chunk_id=f"DGFT_NOTIF_{i}", notif_no=f"{i}/2024")
        for i in range(1, 15)
    ]
    # Add one duplicate identity to verify duplicate flagging without arbitrary failure
    dgft_records.append(make_valid_dgft_record(chunk_id="DGFT_NOTIF_1", notif_no="1/2024"))
    dgft_file.write_text("\n".join(json.dumps(r) for r in dgft_records) + "\n", encoding="utf-8")

    pib_file = tmp_path / "pib_press_releases.jsonl"
    pib_records = [
        make_valid_pib_record(prid=f"{2000 + i}", title=f"Commerce release {i}")
        for i in range(1, 10)
    ]
    pib_file.write_text("\n".join(json.dumps(r) for r in pib_records) + "\n", encoding="utf-8")

    res_dgft = validate_corpus_file(str(dgft_file), source="dgft")
    assert res_dgft["valid"]
    assert res_dgft["record_count"] == 15
    assert res_dgft["unique_count"] == 14
    assert res_dgft["duplicate_count"] == 1

    res_pib = validate_corpus_file(str(pib_file), source="pib")
    assert res_pib["valid"]
    assert res_pib["record_count"] == 9
    assert res_pib["unique_count"] == 9
    assert res_pib["duplicate_count"] == 0

    assert is_usable_corpus(str(dgft_file), source="dgft")
    assert is_usable_corpus(str(pib_file), source="pib")
    assert validate_gate([str(dgft_file), str(pib_file)]) == 0


def test_shared_workflow_gate_cli_invocation(tmp_path):
    """CLI execution of python -m src.utils.freshness_utils behaves correctly in subprocess."""
    dgft_file = tmp_path / "dgft_policy_chunks.jsonl"
    dgft_file.write_text(json.dumps(make_valid_dgft_record()) + "\n", encoding="utf-8")

    pib_file = tmp_path / "pib_press_releases.jsonl"
    pib_file.write_text(json.dumps(make_valid_pib_record()) + "\n", encoding="utf-8")

    # Pass case
    proc = subprocess.run(
        [sys.executable, "-m", "src.utils.freshness_utils", str(dgft_file), str(pib_file)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "[GATE PASSED]" in proc.stdout

    # Fail case with corrupt file
    pib_file.write_text("<!DOCTYPE html><html>403</html>", encoding="utf-8")
    proc_fail = subprocess.run(
        [sys.executable, "-m", "src.utils.freshness_utils", str(dgft_file), str(pib_file)],
        capture_output=True,
        text=True,
    )
    assert proc_fail.returncode == 1
    assert "[GATE FAILED]" in proc_fail.stdout


def test_workflow_file_usability_gate_and_auto_push_integrity():
    """6. Workflow file: yaml parses; the corpus-usability gate exists before publish steps;
    the auto-push step is byte-identical to before this change; calls python -m src.utils.freshness_utils.
    """
    workflow_path = ".github/workflows/refresh_rag.yml"
    assert os.path.exists(workflow_path), f"Workflow {workflow_path} must exist"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_content = f.read()

    # 1. YAML parses cleanly
    parsed = yaml.safe_load(workflow_content)
    assert parsed is not None
    assert "jobs" in parsed and "refresh" in parsed["jobs"]

    steps = parsed["jobs"]["refresh"]["steps"]
    step_names = [s.get("name", s.get("uses", "")) for s in steps]

    # 2. Corpus usability gate exists BEFORE publish steps
    gate_indices = [i for i, name in enumerate(step_names) if "corpus usability gate" in name.lower()]
    assert len(gate_indices) == 1, f"Expected 1 corpus usability gate step, found {len(gate_indices)}"
    gate_idx = gate_indices[0]
    gate_step = steps[gate_idx]

    # Verify gate calls python validator with both corpora
    run_cmd = gate_step.get("run", "")
    assert "src.utils.freshness_utils" in run_cmd
    assert "dgft_policy_chunks.jsonl" in run_cmd
    assert "pib_press_releases.jsonl" in run_cmd

    # Identify publish steps
    publish_step_names = [
        "Archive dense index and publish corpus",
        "Refresh deployment archive without changing model weights",
        "Commit production sparse index",
    ]
    for p_name in publish_step_names:
        p_indices = [i for i, name in enumerate(step_names) if name == p_name]
        assert len(p_indices) == 1, f"Expected publish step '{p_name}' in workflow"
        assert gate_idx < p_indices[0], f"Corpus usability gate must come BEFORE publish step '{p_name}'"

    # 3. Artifact step includes data/cache/source_freshness.json
    artifact_steps = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
    assert len(artifact_steps) >= 1
    artifact_paths = artifact_steps[0].get("with", {}).get("path", "")
    assert "data/cache/source_freshness.json" in artifact_paths

    # 4. GITHUB_STEP_SUMMARY step appends source_freshness
    summary_steps = [s for s in steps if "GITHUB_STEP_SUMMARY" in s.get("run", "")]
    assert len(summary_steps) >= 1, "Expected step appending to GITHUB_STEP_SUMMARY"
    assert "source_freshness.json" in summary_steps[0].get("run", "")

    # 5. Auto-push step "Commit production sparse index" is byte-identical to original HEAD
    head_content = subprocess.check_output(
        ["git", "show", f"99fa9680bd47f37d51297d73b8cf3521963e7961:{workflow_path}"],
        text=True,
    )

    def extract_step_raw(text: str, name: str) -> str:
        lines = text.splitlines()
        extracted = []
        recording = False
        for line in lines:
            if line.strip() == f"- name: {name}":
                recording = True
                extracted.append(line)
            elif recording:
                if line.startswith("      - ") or (
                    line.startswith("  ")
                    and not line.startswith("        ")
                    and not line.startswith("      ")
                ):
                    break
                extracted.append(line)
        return "\n".join(extracted)

    head_auto_push = extract_step_raw(head_content, "Commit production sparse index")
    current_auto_push = extract_step_raw(workflow_content, "Commit production sparse index")

    assert head_auto_push != "", "Could not locate 'Commit production sparse index' in original HEAD workflow"
    assert current_auto_push != "", "Could not locate 'Commit production sparse index' in current workflow"
    assert current_auto_push == head_auto_push, (
        f"Auto-push step must be byte-identical to original HEAD!\nExpected:\n{head_auto_push}\n\nGot:\n{current_auto_push}"
    )
