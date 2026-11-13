import pandas as pd
import pytest
from scripts import prepare_repaired_raw as prep
from src.data_ingestion.un_downloader import TOP_20_PARTNERS


def frame(year=2025, partner2=0):
    rows = []
    for p in TOP_20_PARTNERS:
        for f in ('M', 'X'):
            rows.append({'period': year, 'partnerCode': int(p), 'flowCode': f, 'cmdCode': 1, 'partner2Code': partner2,
                         'customsCode': 'C00', 'motCode': 0, 'primaryValue': 5.0, 'netWgt': 1.0, 'grossWgt': 1.0,
                         'qty': 1.0, 'altQty': 1.0, 'cifvalue': 2.0, 'fobvalue': 2.0})
    df = pd.DataFrame(rows)
    pad = {f'c{i}': 0 for i in range(47 - len(df.columns) - 1)}
    for k, v in pad.items():
        df[k] = v
    df['isAggregate'] = False
    df['brent_crude_mean'] = 1.0
    return df


def test_reconstructs_types_and_keeps_values(tmp_path):
    f = frame(); path = tmp_path / 'f.parquet'; f.to_parquet(path)
    out = tmp_path / 'raw.parquet'
    prov = prep.prepare(path, out, download=False, artifact_dir=tmp_path / 'none')
    raw = pd.read_parquet(out)
    assert prov['rows'] == len(f) and raw.period.iloc[0] == '2025' and raw.cmdCode.iloc[0] == '01'
    assert raw.primaryValue.tolist() == f.primaryValue.tolist()
    assert 'reconstructed' in prov['source']


def test_rejects_mixed_partner2_grain(tmp_path):
    f = frame(); f.loc[0, 'partner2Code'] = 36
    path = tmp_path / 'f.parquet'; f.to_parquet(path)
    with pytest.raises(ValueError, match='partner2Code'):
        prep.prepare(path, tmp_path / 'raw.parquet', download=False, artifact_dir=tmp_path / 'none')
    assert not (tmp_path / 'raw.parquet').exists()


def test_rejects_incomplete_latest_year(tmp_path):
    f = frame().iloc[:-1]
    path = tmp_path / 'f.parquet'; f.to_parquet(path)
    with pytest.raises(ValueError, match='incomplete'):
        prep.prepare(path, tmp_path / 'raw.parquet', download=False, artifact_dir=tmp_path / 'none')


def test_artifact_mismatch_is_rejected(tmp_path):
    f = frame(); path = tmp_path / 'f.parquet'; f.to_parquet(path)
    art = tmp_path / 'art'; art.mkdir()
    bad = f.iloc[:, :47].copy(); bad['primaryValue'] = 9.0
    bad.to_parquet(art / 'candidate_raw.parquet')
    with pytest.raises(ValueError, match='artifact values differ'):
        prep.prepare(path, tmp_path / 'raw.parquet', download=False, artifact_dir=art)
