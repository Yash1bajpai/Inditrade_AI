"""
Utilities for tracking source freshness and verifying corpus usability across ingestion pipelines.
Provides a shared validator for both CI workflow gates and runtime resilience fallbacks.
"""

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_MARKER_PATH = os.path.join("data", "cache", "source_freshness.json")
DEFAULT_DGFT_PATH = os.path.join("data", "processed", "dgft_policy_chunks.jsonl")
DEFAULT_PIB_PATH = os.path.join("data", "processed", "pib_press_releases.jsonl")

DGFT_REQUIRED_KEYS = [
    "chunk_id",
    "clean_text",
    "date",
    "financial_year",
    "notification_no",
    "subject",
    "type",
]

PIB_REQUIRED_KEYS = [
    "prid",
    "title",
    "date",
    "clean_text",
    "category",
    "ministry",
]

_HTML_SIGNATURES = [
    re.compile(r"<!doctype\s+html", re.IGNORECASE),
    re.compile(r"<html[\s>]", re.IGNORECASE),
    re.compile(r"<head[\s>]", re.IGNORECASE),
    re.compile(r"<body[\s>]", re.IGNORECASE),
    re.compile(r"<title>\s*(403|404|500|502|503|error|forbidden|access denied)", re.IGNORECASE),
]

_ERROR_PHRASES = {
    "403 forbidden",
    "access denied",
    "unauthorized",
    "blocked",
    "bad gateway",
    "service unavailable",
    "internal server error",
}


class CorpusValidationError(ValueError):
    """Raised when a policy corpus fails schema, structure, or content validation."""
    pass
def is_html_document(text: str) -> bool:
    """Detects if raw file content begins with an HTML page or error document rather than JSON."""
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    if not stripped:
        return False
    lower = stripped[:200].lower()
    if lower.startswith(("<!doctype html", "<html", "<head", "<?xml")):
        return True
    for pat in _HTML_SIGNATURES:
        if pat.search(stripped[:200]):
            return True
    return False


def is_html_or_error_content(text: str) -> bool:
    """Detects raw HTML pages, HTTP error documents, or error strings masquerading as policy text."""
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    lower = stripped.lower()
    if lower in _ERROR_PHRASES:
        return True
    for pat in _HTML_SIGNATURES:
        if pat.search(stripped):
            return True
    return False


def read_source_freshness(marker_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Reads the source freshness marker JSON file.
    Returns an empty dictionary if the file does not exist or is invalid.
    """
    path = marker_path or DEFAULT_MARKER_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def record_source_freshness(
    source: str,
    status_payload: Dict[str, Any],
    marker_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Atomically records source freshness into the status marker JSON file.
    Merges status_payload under the source key, preserving other sources.
    Uses tempfile + flush + fsync + os.replace to guarantee atomic updates.
    """
    path = marker_path or DEFAULT_MARKER_PATH
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    current = read_source_freshness(path)
    # Cleanly replace this source's entry without stale carry-over from prior runs
    current[source] = dict(status_payload)

    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return current


def _detect_source(filepath: str, sample_record: Optional[Dict[str, Any]] = None) -> str:
    """Infers whether the corpus belongs to DGFT or PIB from filename or record keys."""
    fname = os.path.basename(filepath).lower()
    if "dgft" in fname:
        return "dgft"
    if "pib" in fname:
        return "pib"
    if sample_record and isinstance(sample_record, dict):
        if "notification_no" in sample_record or "chunk_id" in sample_record:
            return "dgft"
        if "prid" in sample_record or "ministry" in sample_record:
            return "pib"
    return "dgft"


def validate_corpus_file(
    filepath: Optional[str],
    source: str = "auto",
    min_records: int = 1,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Shared corpus validator for both runtime blocked-source fallbacks and CI usability gates.
    
    Invariants checked:
    1. File exists and is non-empty (>0 bytes).
    2. File is not an HTML error document or truncated stream.
    3. EVERY non-blank line must be a valid JSON object.
    4. Required schema keys must all exist and have non-empty, non-trivial values:
       - DGFT: chunk_id, clean_text, date, financial_year, notification_no, subject, type, url.
       - PIB: prid, title, date, clean_text, category, ministry, url.
    5. clean_text must be non-trivial (>=20 chars) and not HTML/error masquerading as content.
    6. Minimum record count enforced without arbitrary high floors.
    """
    errors: List[str] = []

    if not filepath or not os.path.exists(filepath):
        msg = f"Corpus file does not exist: {filepath}"
        if raise_on_error:
            raise CorpusValidationError(msg)
        return {"valid": False, "source": source, "record_count": 0, "unique_count": 0, "duplicate_count": 0, "errors": [msg], "error": msg}

    try:
        file_size = os.path.getsize(filepath)
    except OSError as e:
        msg = f"Could not stat corpus file {filepath}: {e}"
        if raise_on_error:
            raise CorpusValidationError(msg)
        return {"valid": False, "source": source, "record_count": 0, "unique_count": 0, "duplicate_count": 0, "errors": [msg], "error": msg}

    if file_size == 0:
        msg = f"Corpus file {filepath} is empty (0 bytes)."
        if raise_on_error:
            raise CorpusValidationError(msg)
        return {"valid": False, "source": source, "record_count": 0, "unique_count": 0, "duplicate_count": 0, "errors": [msg], "error": msg}

    # Inspect first bytes for raw HTML error pages (e.g. 403 Forbidden dump)
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_chunk = f.read(512)
    except Exception as e:
        msg = f"Failed to read corpus file {filepath}: {e}"
        if raise_on_error:
            raise CorpusValidationError(msg)
        return {"valid": False, "source": source, "record_count": 0, "unique_count": 0, "duplicate_count": 0, "errors": [msg], "error": msg}

    if is_html_document(first_chunk):
        msg = f"Corpus file {filepath} starts with HTML markup or error response rather than JSONL."
        if raise_on_error:
            raise CorpusValidationError(msg)
        return {"valid": False, "source": source, "record_count": 0, "unique_count": 0, "duplicate_count": 0, "errors": [msg], "error": msg}
    resolved_source = source
    records_count = 0
    seen_identities: Set[Any] = set()
    duplicate_count = 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue  # Ignore purely blank lines

                try:
                    obj = json.loads(raw)
                except Exception as e:
                    errors.append(f"Line {line_no}: Not valid JSON ({e}). Content: {raw[:80]!r}")
                    break  # Fatal syntax error: JSONL is corrupt

                if not isinstance(obj, dict):
                    errors.append(f"Line {line_no}: Record must be a JSON object, got {type(obj).__name__}.")
                    break

                # Resolve source type from first valid record if auto
                if resolved_source == "auto":
                    resolved_source = _detect_source(filepath, obj)

                req_keys = DGFT_REQUIRED_KEYS if resolved_source == "dgft" else PIB_REQUIRED_KEYS
                missing = [k for k in req_keys if k not in obj]
                if missing:
                    errors.append(f"Line {line_no}: Missing required schema keys {missing}.")
                    break

                # Validate field values are non-empty
                empty_fields = [k for k in req_keys if obj.get(k) is None or str(obj.get(k)).strip() == ""]
                if empty_fields:
                    errors.append(f"Line {line_no}: Empty required field(s) {empty_fields}.")
                    break
                # Validate optional url when present
                if "url" in obj and obj.get("url") is not None:
                    raw_url = str(obj.get("url")).strip()
                    if raw_url == "":
                        errors.append(f"Line {line_no}: Field 'url' is present but empty.")
                        break
                    elif not (raw_url.startswith(("http://", "https://", "/")) or "." in raw_url):
                        errors.append(f"Line {line_no}: Field 'url' is malformed: {raw_url!r}.")
                        break

                # Validate non-trivial clean text
                clean_text = str(obj.get("clean_text", ""))
                if len(clean_text.strip()) < 20 or is_html_or_error_content(clean_text):
                    errors.append(f"Line {line_no}: 'clean_text' is trivial ({len(clean_text)} chars) or contains HTML/error content.")
                    break

                # Track deduplication
                if resolved_source == "dgft":
                    identity = (str(obj.get("chunk_id")), str(obj.get("financial_year", "")))
                else:
                    identity = str(obj.get("prid"))

                if identity in seen_identities:
                    duplicate_count += 1
                else:
                    seen_identities.add(identity)

                records_count += 1

    except Exception as e:
        errors.append(f"Unexpected reading error at line {line_no if 'line_no' in locals() else 'start'}: {e}")

    if not errors and records_count < min_records:
        errors.append(f"Corpus file contains only {records_count} record(s); minimum required is {min_records}.")

    is_valid = len(errors) == 0
    composite_error = "; ".join(errors) if errors else ""

    if raise_on_error and not is_valid:
        raise CorpusValidationError(f"{filepath} failed validation: {composite_error}")

    return {
        "valid": is_valid,
        "source": resolved_source,
        "record_count": records_count,
        "unique_count": len(seen_identities),
        "duplicate_count": duplicate_count,
        "errors": errors,
        "error": composite_error,
    }


def is_usable_corpus(
    filepath: Optional[str],
    source: str = "auto",
    min_records: int = 1,
) -> bool:
    """
    Verifies that a policy corpus file exists, is non-empty, and every non-blank
    line is a valid, non-trivial record meeting the required schema.
    Delegates directly to validate_corpus_file to prevent code drift.
    """
    res = validate_corpus_file(
        filepath=filepath,
        source=source,
        min_records=min_records,
        raise_on_error=False,
    )
    return bool(res.get("valid", False))


def validate_gate(
    corpus_paths: Optional[List[str]] = None,
    min_records: int = 1,
) -> int:
    """
    CLI/workflow gate entrypoint.
    Inspects each required corpus file; outputs explicit per-corpus diagnostics.
    Exits with code 0 on complete pass, or code 1 if any corpus fails.
    """
    paths = corpus_paths or [DEFAULT_DGFT_PATH, DEFAULT_PIB_PATH]
    print("\n" + "=" * 70)
    print("=== [POLICY CORPUS USABILITY GATE] ===")
    print("=" * 70)

    all_passed = True
    for path in paths:
        print(f"[*] Validating corpus: {path} ...", flush=True)
        result = validate_corpus_file(path, source="auto", min_records=min_records, raise_on_error=False)
        if result["valid"]:
            dups_note = f", {result['duplicate_count']} duplicates flagged" if result['duplicate_count'] > 0 else ""
            print(
                f"    [PASS] {result['source'].upper()} corpus usable: "
                f"{result['record_count']} valid records ({result['unique_count']} unique{dups_note})."
            )
        else:
            all_passed = False
            print(f"    [FAIL] {result['source'].upper()} corpus INVALID: {result['error']}")

    print("=" * 70)
    if all_passed:
        print("[GATE PASSED] All required policy corpora are valid and usable.")
        print("=" * 70 + "\n")
        return 0
    else:
        print("[GATE FAILED] Policy corpus usability check failed. Failing closed before publish.")
        print("=" * 70 + "\n")
        return 1


def main() -> int:
    """Main CLI entrypoint for invoking the gate from shell workflows."""
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    return validate_gate(targets)


if __name__ == "__main__":
    sys.exit(main())
