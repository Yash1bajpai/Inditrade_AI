"""Atomic repair of mixed-grain annual slices, never substitute missing data.

Writes a new candidate parquet/report. Does not alter source, releases or models.
Existing slices are replaced in full because filtering a bad commodity cell alone
would hide that other canonical commodities can be missing in that slice too.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_ingestion.un_downloader import ComtradeFetcher, TOP_20_PARTNERS

SLICE = ['period', 'partnerCode', 'flowCode']
KEY = SLICE + ['cmdCode']


def normalized(frame):
    df = frame.copy()
    for c in ['period', 'partnerCode', 'partner2Code', 'motCode']:
        df[c] = pd.to_numeric(df[c], errors='raise').astype(int)
    df['cmdCode'] = df.cmdCode.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(2)
    return df


def plan_repairs(raw, latest_year=2025):
    df = normalized(raw)
    bad = ~(df.partner2Code.eq(0) & df.customsCode.eq('C00') & df.motCode.eq(0))
    slices = {tuple(r) for r in df.loc[bad, SLICE].itertuples(index=False, name=None)}
    have = {tuple(r) for r in df[SLICE].itertuples(index=False, name=None)}
    slices |= {(latest_year, int(p), f) for p in TOP_20_PARTNERS for f in ['M', 'X']} - have
    return sorted(slices)


def repair(raw, fetcher, pause=time.sleep, allow_partial=False, missing_slices=None):
    df = normalized(raw)
    repairs = plan_repairs(df)
    replacements = []
    missing = missing_slices if missing_slices is not None else []
    repaired = []
    for year, partner, flow in repairs:
        fresh, status = fetcher.fetch_slice(partner, year, flow)
        if status != 'SUCCESS' or fresh is None:
            raise RuntimeError(f'Canonical refetch failed for {year}/{partner}/{flow}: {status}; source untouched')
        pause(1.5)
        expected = fetcher.count_slice(partner, year, flow)
        if fresh.empty:
            if not allow_partial or expected != 0:
                raise ValueError('Empty response requires partial mode and independent zero source count')
            missing.append({'year': year, 'partner': str(partner), 'flow': flow,
                            'reason': 'canonical_query_empty', 'source_count': expected})
            pause(1.5)
            continue
        if expected != len(fresh) or expected <= 0 or expected >= 500:
            raise ValueError('Refetch completeness check failed: source count differs or limit reached')
        fresh = normalized(fresh)
        if not (fresh.period.eq(year) & fresh.partnerCode.eq(partner) & fresh.flowCode.eq(flow)).all():
            raise ValueError('Response does not match requested slice')
        if not (fresh.partner2Code.eq(0) & fresh.customsCode.eq('C00') & fresh.motCode.eq(0)).all():
            raise ValueError('Refetch returned non-total grain')
        if fresh.duplicated(KEY).any() or not fresh.cmdCode.str.fullmatch(r'\d{2}').all():
            raise ValueError('Refetch contains duplicate/non-HS2 cells')
        if pd.to_numeric(fresh.primaryValue, errors='coerce').isna().any():
            raise ValueError('Refetch missing trade values')
        if "reporterCode" not in fresh or not pd.to_numeric(fresh.reporterCode, errors='coerce').eq(699).all():
            raise ValueError('Response reporter is not India')
        if pd.to_numeric(fresh.primaryValue, errors='coerce').lt(0).any():
            raise ValueError('Negative trade values')
        replacements.append(fresh)
        repaired.append((year, partner, flow))
        pause(1.5)
    replace_mask = pd.Series([tuple(r) in set(repairs) for r in df[SLICE].itertuples(index=False, name=None)], index=df.index)
    candidate = pd.concat([df.loc[~replace_mask], *replacements], ignore_index=True)
    if candidate.duplicated(KEY).any():
        raise ValueError('Candidate contains duplicate canonical cells')
    unresolved = set(plan_repairs(candidate))
    documented = {(s['year'], int(s['partner']), s['flow']) for s in missing}
    if unresolved != {gap for gap in documented if gap[0] == 2025}:
        raise ValueError('Candidate still has mixed/missing slices')
    return candidate.sort_values(KEY).reset_index(drop=True), repaired


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError('Repair must write a separate candidate, not overwrite source')
    raw = pd.read_parquet(source)
    print(json.dumps({'repair_slices': plan_repairs(raw), 'source_rows': len(raw)}))
    if args.plan_only:
        return
    missing = []
    candidate, repaired = repair(raw, ComtradeFetcher(), allow_partial=args.allow_partial, missing_slices=missing)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp.parquet')
    candidate.to_parquet(temp, index=False)
    temp.replace(output)
    output.with_suffix('.repair.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'repaired_slices': repaired, 'source_rows': len(raw), 'candidate_rows': len(candidate),
        'production_publication': False, 'model_retraining': False
    }, indent=2))
    coverage = {
        'schema_version': 1, 'status': 'partial' if missing else 'checked_repair_candidate',
        'as_of': datetime.now(timezone.utc).isoformat(), 'missing_slices': missing,
        'candidate_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'planned_slices': repairs_as_records(plan_repairs(raw)),
        'repaired_slices': repairs_as_records(repaired),
        'candidate_rows': len(candidate), 'model_promoted': False,
        'scope': 'Repair plan and latest-year top-20 partner M/X slice presence; not all historical HS2 completeness',
        'note': 'Empty canonical query plus zero count is a gap, not zero trade or proof of global unavailability.'}
    output.with_name('coverage_manifest.json').write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage))


def repairs_as_records(slices):
    return [{'year': y, 'partner': str(p), 'flow': f} for y, p, f in slices]

if __name__ == '__main__':
    main()
