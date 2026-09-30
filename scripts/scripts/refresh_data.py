"""Refresh real inputs before training. Fail closed, never generate seed data."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_ingestion.forex_macro_fetcher import fetch_and_verify_all
from src.data_ingestion.un_downloader import run_refresh_loop, OUTPUT_PARQUET
from src.feature_engineering.trade_features import build_trade_features

def refresh_training_inputs():
    fetch_and_verify_all()
    target = datetime.now(timezone.utc).year - 1
    raw = pd.read_parquet(OUTPUT_PARQUET)
    from src.data_ingestion.un_downloader import TOP_20_PARTNERS
    need = {(str(int(p)), f) for p in TOP_20_PARTNERS for f in ('M', 'X')}
    def coverage(frame):
        return {(str(int(r.partnerCode)), r.flowCode) for r in frame[frame.period.astype(int) == target].itertuples()}
    if not need.issubset(coverage(raw)):
        result = run_refresh_loop(year=target)
        if result is None:
            raise RuntimeError(f'Annual trade refresh for {target} failed')
        raw = pd.read_parquet(OUTPUT_PARQUET)
    if not need.issubset(coverage(raw)):
        raise RuntimeError(f'Incomplete annual trade coverage for {target}: {sorted(need-coverage(raw))}')
    build_trade_features()
    features = Path('data/processed/trade_features.parquet')
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'annual_trade_latest_year': target,
              'feature_sha256': hashlib.sha256(features.read_bytes()).hexdigest(),
              'macro_latest_dates': {p.stem: str(pd.to_datetime(pd.read_csv(p).Date).max().date())
                                     for p in Path('data/raw/forex_macro').glob('*.csv')}}
    Path('data/cache/data_freshness_meta.json').write_text(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    refresh_training_inputs()
