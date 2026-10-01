"""Regenerate descriptive and past-only features without touching releases/models.

Annual macro values are retained from the existing feature source. They are not
refreshed here; missing annual values stay missing, never synthetic zeros.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.feature_engineering.forecast_features import prepare_forecast_frame
from src.feature_engineering.trade_features import M49_PARTNER_MAP


def rebuild(raw, prior):
    for column, expected in [('partner2Code',0),('customsCode','C00'),('motCode',0)]:
        if column not in raw or not raw[column].astype(str).str.replace(r'\.0$','',regex=True).eq(str(expected)).all():
            raise ValueError('Candidate grain must be canonical')
    if raw.duplicated(['partnerCode','cmdCode','flowCode','period']).any():
        raise ValueError('Duplicate candidate source')
    raw=raw.copy();prior=prior.copy()
    raw['period']=pd.to_numeric(raw.period,errors='raise').astype(int)
    prior['period']=pd.to_numeric(prior.period,errors='raise').astype(int)
    raw['partnerCode']=pd.to_numeric(raw.partnerCode,errors='raise').astype(int)
    macro_cols=[c for c in prior if c.endswith(('_mean','_vol_std','_year_end','_yoy_pct')) and not c.startswith(('primaryValue','netWgt'))]
    for c in macro_cols:
        if prior.groupby('period')[c].nunique(dropna=True).gt(1).any():
            raise ValueError('Conflicting annual macro values')
    macro=prior.groupby('period',as_index=False)[macro_cols].first()
    merged=raw.merge(macro,on='period',how='left',validate='many_to_one')
    code=merged.partnerCode.astype(str).str.lstrip('0')
    merged['partnerDesc']=code.map({k.lstrip('0'):v['desc'] for k,v in M49_PARTNER_MAP.items()})
    merged['partnerISO']=code.map({k.lstrip('0'):v['iso'] for k,v in M49_PARTNER_MAP.items()})
    merged['flowDesc']=merged.flowCode.map({'M':'Import','X':'Export'})
    X,y,values,features,frame,excluded=prepare_forecast_frame(merged)
    if excluded:raise ValueError('Candidate excluded noncanonical rows')
    return frame,X,features


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--coverage-manifest', required=True)
    p.add_argument('--raw',required=True);p.add_argument('--prior-features',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    coverage=json.loads(Path(a.coverage_manifest).read_text())
    if coverage['candidate_sha256'] != hashlib.sha256(Path(a.raw).read_bytes()).hexdigest():
        raise ValueError('Coverage manifest does not match raw candidate')
    raw=pd.read_parquet(a.raw);prior=pd.read_parquet(a.prior_features)
    for gap in coverage['missing_slices']:
        if ((raw.period.astype(int)==gap['year']) & (raw.partnerCode.astype(int)==int(gap['partner'])) & (raw.flowCode==gap['flow'])).any():
            raise ValueError('Documented missing slice contains candidate rows')
    frame,X,features=rebuild(raw,prior)
    frame.to_parquet(out/'candidate_trade_features.parquet',index=False)
    X.to_parquet(out/'past_only_predictors.parquet',index=False)
    coverage['feature_sha256']=hashlib.sha256((out/'candidate_trade_features.parquet').read_bytes()).hexdigest()
    (out/'coverage_manifest.json').write_text(json.dumps(coverage,indent=2))
    report={'coverage_status':coverage['status'],'missing_slices':coverage['missing_slices'],'candidate_rows':len(frame),'forecast_features':features,'macro_source':'retained annual values from prior feature file, not refreshed',
      'macro_source_sha256':hashlib.sha256(Path(a.prior_features).read_bytes()).hexdigest(),
      'production_publication':False,'model_retraining':False,
      'missing_predictor_cells':X.isna().sum().to_dict(),
      'artifacts_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.parquet')}}
    (out/'feature_regeneration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
