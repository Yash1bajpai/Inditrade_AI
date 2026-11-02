"""
Utilities for tracking source freshness and verifying corpus usability across ingestion pipelines.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_MARKER_PATH = os.path.join("data", "cache", "source_freshness.json")


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
    # Replace the specific source dictionary cleanly (no stale fields carry over)
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


def is_usable_corpus(filepath: Optional[str]) -> bool:
    """
    Verifies that a policy corpus file exists, has non-zero size,
    and contains at least one non-empty line of content.
    """
    if not filepath or not os.path.exists(filepath):
        return False
    try:
        if os.path.getsize(filepath) == 0:
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False
    return False
