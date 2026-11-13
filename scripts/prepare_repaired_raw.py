"""Prepare the repaired raw Comtrade input for the refresh workflow, fail closed.

The data-v1 raw asset india_trade_hs2_2015_2024.parquet is the old file with
mixed-grain rows. The repaired dataset is the released trade_features.parquet
(first 47 columns are the raw schema). Prefer the original repair artifact when
it is still available and verify it against those columns; otherwise rebuild the
raw from them (period -> str, cmdCode -> 2-digit str). Never fetches data and
never writes anywhere except the local raw parquet.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_ingestion.un_downloader import TOP_20_PARTNERS

KEYS = ['period', 'partnerCode', 'flowCode', 'cmdCode']
NUM = ['primaryValue', 'netWgt', 'grossWgt', 'qty', 'altQty', 'cifvalue', 'fobvalue']
RAW_COLUMNS = 47
ARTIFACT_RUN = '41393714519'
ARTIFACT_NAME = 'partial-trade-repair-candidate'


def canon(df):
    d = df.copy()
    d['period'] = pd.to_numeric(d.period).astype(int)
    d['partnerCode'] = pd.to_numeric(d.partnerCode).astype(int)
    d['cmdCode'] = d.cmdCode.astype(str).str.replace(r'\.0$', '', regex=True).astype(int)
    return d.sort_values(KEYS).reset_index(drop=True)


def check_grain(df, label):
    if not pd.to_numeric(df.partner2Code).eq(0).all():
        raise ValueError(f'{label}: partner2Code is not all 0')
    if not (df.customsCode.eq('C00').all() and pd.to_numeric(df.motCode).eq(0).all()):
        raise ValueError(f'{label}: customsCode/motCode are not the total grain')
    c = canon(df)
    if c.duplicated(KEYS).any():
        raise ValueError(f'{label}: duplicate keys')
    values = pd.to_numeric(df.primaryValue, errors='coerce')
    if values.isna().any() or (values < 0).any():
        raise ValueError(f'{label}: missing or negative primaryValue')
    latest = int(c.period.max())
    need = {(int(p), f) for p in TOP_20_PARTNERS for f in ('M', 'X')}
    have = set(zip(c[c.period == latest].partnerCode, c[c.period == latest].flowCode))
    if not need.issubset(have):
        raise ValueError(f'{label}: incomplete {latest} coverage: {sorted(need - have)}')
    return c


def prepare(features_path, out_path, artifact_dir=None, download=True):
    feat = pd.read_parquet(features_path)
    if len(feat.columns) <= RAW_COLUMNS or feat.columns[RAW_COLUMNS - 1] != 'isAggregate':
        raise ValueError('Unexpected trade_features layout: raw columns not first')
    raw_cols = list(feat.columns[:RAW_COLUMNS])
    expected = check_grain(feat[raw_cols], 'features raw columns')
    prov = {'features_rows': len(feat)}
    raw, source = None, None
    cand_path = pathlib.Path(artifact_dir or '/tmp/repair-artifact') / 'candidate_raw.parquet'
    try:
        if download:
            subprocess.run(['gh', 'run', 'download', ARTIFACT_RUN, '-n', ARTIFACT_NAME, '-D', str(cand_path.parent)],
                           check=True, capture_output=True, text=True)
        if cand_path.exists():
            cand = pd.read_parquet(cand_path)
            got = check_grain(cand, 'artifact candidate_raw')
            if len(got) != len(expected) or not (got[KEYS].values == expected[KEYS].values).all():
                raise ValueError('artifact keys differ from deployed features')
            for c in NUM:
                x, y = pd.to_numeric(got[c]), pd.to_numeric(expected[c])
                if not ((x == y) | (x.isna() & y.isna())).all():
                    raise ValueError(f'artifact values differ from deployed features: {c}')
            raw = cand
            source = f'artifact {ARTIFACT_NAME} (run {ARTIFACT_RUN}), verified equal to deployed features on keys and numeric values'
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        prov['artifact_unavailable'] = str(e)[:200]
    if raw is None:
        raw = feat[raw_cols].copy()
        raw['period'] = raw.period.astype(int).astype(str)
        raw['cmdCode'] = raw.cmdCode.astype(int).astype(str).str.zfill(2)
        source = 'reconstructed from the 47 raw columns of trade_features.parquet (period -> str, cmdCode -> 2-digit str)'
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out, index=False)
    prov.update({'source': source, 'rows': len(raw)})
    print('RAW INPUT PROVENANCE ' + json.dumps(prov))
    return prov


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='data/processed/trade_features.parquet')
    p.add_argument('--out', default='data/raw/un_comtrade/india_trade_hs2_2015_2024.parquet')
    a = p.parse_args()
    prepare(a.features, a.out)
