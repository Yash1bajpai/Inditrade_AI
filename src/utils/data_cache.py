"""Mtime-based cached loaders for hot DataFrames.

Every API endpoint used to call pd.read_parquet() per request, re-parsing the
trade_features parquet and blocking the event loop each time. These loaders
cache by (mtime_ns, size) so dev-time edits still invalidate automatically,
while production requests after the first are pure memory hits.

pd.read_parquet / pd.read_csv are resolved as attributes at call time so that
test suites which monkeypatch them (see tests/conftest.py) keep working.
"""

import logging
import os
import threading

import pandas as pd

logger = logging.getLogger("utils.data_cache")

_lock = threading.Lock()
_cache = {}


def _current_sig(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _load(kind, path, **read_kwargs):
    key = os.path.abspath(path)
    sig = _current_sig(key)

    entry = _cache.get(key)
    if entry is not None and sig is not None and entry[0] == sig:
        return entry[1]

    with _lock:
        entry = _cache.get(key)
        if entry is not None and sig is not None and entry[0] == sig:
            return entry[1]

        reader = getattr(pd, f"read_{kind}")
        df = reader(path, **read_kwargs)
        logger.info("data_cache: loaded %s (%d rows)", key, len(df))
        _cache[key] = (sig, df)
        return df


def load_parquet(path, **read_kwargs):
    """Return a DataFrame for `path`, reloading when file mtime/size changes.

    Returns a copy: callers (e.g. forecast endpoints) add derived columns
    in place, and sharing the cached frame across threads would corrupt it.
    The copy is still far cheaper than re-parsing the parquet.
    """
    return _load("parquet", path, **read_kwargs).copy()


def load_csv(path, **read_kwargs):
    """Return a DataFrame for `path`, reloading when file mtime/size changes.

    Returns a copy — see load_parquet.
    """
    return _load("csv", path, **read_kwargs).copy()


def clear_cache():
    """Drop all cached frames (used by tests)."""
    with _lock:
        _cache.clear()
