import pandas as pd
from scripts.build_repair_candidate import rebuild


def test_candidate_uses_calendar_lags_and_lagged_retained_macro():
    raw=pd.DataFrame([{'period':y,'partnerCode':842,'cmdCode':'27','flowCode':'M','partner2Code':0,'customsCode':'C00','motCode':0,'primaryValue':v} for y,v in [(2022,100),(2024,400),(2025,500)]])
    prior=pd.DataFrame({'period':[2022,2023,2024,2025],'usdinr_mean':[80.,81.,82.,83.]})
    frame,X,features=rebuild(raw,prior)
    assert frame.loc[frame.period.eq(2024),'primaryValue_lag_1y'].isna().all()
    assert X.loc[frame.period.eq(2025),'usdinr_mean_lag_1y'].iloc[0]==82.
    assert 'primaryValue' not in features
    assert 'usdinr_mean' not in features
    assert frame.partnerISO.eq('USA').all()
