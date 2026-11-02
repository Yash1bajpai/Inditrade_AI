"""Past-only features for rolling one-year forecasts, not a multi-year backtest."""
import numpy as np
import pandas as pd

KEYS = ['partnerCode', 'cmdCode', 'flowCode']
LAGS = [1, 3, 5]

def prepare_forecast_frame(frame):
    """Exclude non-total partner2 slices and build exact calendar-year lags.

    Missing years stay missing. Current-year trade, quantities and macro values
    never enter X. Prior holdout-year observations may feed the following year,
    so the evaluation is rolling one-year, not a frozen-origin multi-year test.
    """
    df = frame.drop(columns=[c for c in frame if c.startswith(('primaryValue_lag_', 'primaryValue_rolling_')) or c.endswith('_lag_1y')], errors='ignore').copy()
    excluded = 0
    if 'partner2Code' in df:
        totals = pd.to_numeric(df.partner2Code, errors='coerce').eq(0)
        excluded = int((~totals).sum())
        df = df.loc[totals].copy()
    df['period'] = pd.to_numeric(df.period, errors='raise').astype(int)
    for c in ['partnerCode', 'cmdCode']:
        df[c] = pd.to_numeric(df[c], errors='raise').astype(int)
    if not df.flowCode.isin(['M', 'X']).all():
        raise ValueError('Forecast data must contain only import/export flows')
    if df.duplicated(KEYS + ['period']).any():
        raise ValueError('Ambiguous trade grain: duplicate series/year totals')
    df = df.sort_values(['period'] + KEYS).reset_index(drop=True)
    values = pd.to_numeric(df.primaryValue, errors='coerce')
    if values.isna().any() or (values < 0).any():
        raise ValueError('Missing/negative targets are not zero trade')
    df['primaryValue'] = values
    for lag in LAGS + [2, 4]:
        past = df[KEYS + ['period', 'primaryValue']].copy()
        past['period'] += lag
        past.rename(columns={'primaryValue': f'primaryValue_lag_{lag}y'}, inplace=True)
        df = df.merge(past, on=KEYS + ['period'], how='left', validate='one_to_one')
    for window in [3, 5]:
        cols = [f'primaryValue_lag_{k}y' for k in range(1, window + 1)]
        df[f'primaryValue_rolling_{window}y_mean'] = df[cols].mean(axis=1)
    df['flow_is_export'] = df.flowCode.eq('X').astype(float)
    features = ['period', 'partnerCode', 'cmdCode', 'flow_is_export']
    features += [f'primaryValue_lag_{k}y' for k in LAGS]
    features += [f'primaryValue_rolling_{k}y_mean' for k in [3, 5]]
    # Annual macro statistics are available only after their year closes.
    macro_cols = [c for c in frame if c.endswith(('_mean', '_vol_std', '_year_end', '_yoy_pct')) and not c.startswith(('primaryValue', 'netWgt'))]
    for c in macro_cols:
        by_year = frame.groupby('period')[c].nunique(dropna=True)
        if (by_year > 1).any():
            raise ValueError(f'Macro column {c} has conflicting annual values')
        macro = frame[['period', c]].copy()
        macro['period'] = pd.to_numeric(macro.period).astype(int) + 1
        macro = macro.groupby('period', as_index=False)[c].first()
        name = c + '_lag_1y'
        macro.rename(columns={c: name}, inplace=True)
        df = df.merge(macro, on='period', how='left', validate='many_to_one')
        features.append(name)
    X = df[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return X, np.log1p(df.primaryValue), df.primaryValue, features, df, excluded


class HorizonError(ValueError):
    """Requested forecast year has no anchoring history (client input problem)."""


def build_inference_frame(frame, target_year, series=None):
    """Construct past-only feature rows for `target_year` from raw history.

    A training row for year Y carries values from years < Y only. Serving must
    follow the same contract: copying the latest stored row as a template feeds
    lags that are stale by (target_year - latest_year), so a 2026 request would
    score 2024 values against a model validated on calendar-exact lags.

    Returns one row per (partnerCode, cmdCode, flowCode) series with the same
    predictor columns prepare_forecast_frame emits. Macro statistics follow the
    training rule: the model's *_lag_1y inputs come from year target_year - 1.

    `series` may restrict the expensive lag/rolling construction to one
    (partnerCode, cmdCode) pair; horizon, grain, value and macro checks still
    run on the full frame so a scoped call cannot weaken validation.

    Fails closed: raises ValueError on ambiguous grain (duplicate series/year
    totals) or non-positive/missing trade values; raises HorizonError for a
    horizon more than one year past the latest observation or at/before the
    first observed year, where no lag_1y actual exists to anchor the forecast.
    """
    if 'primaryValue' not in frame or 'period' not in frame:
        raise ValueError('History lacks the raw trade columns needed to build forecast inputs')
    df = frame.copy()
    if 'partner2Code' in df:
        totals = pd.to_numeric(df.partner2Code, errors='coerce').eq(0)
        df = df.loc[totals].copy()
    df['period'] = pd.to_numeric(df.period, errors='raise').astype(int)
    for c in ['partnerCode', 'cmdCode']:
        df[c] = pd.to_numeric(df[c], errors='raise').astype(int)
    if not df.flowCode.isin(['M', 'X']).all():
        raise ValueError('Forecast data must contain only import/export flows')
    if df.duplicated(KEYS + ['period']).any():
        raise ValueError('Ambiguous trade grain: duplicate series/year totals')
    target_year = int(target_year)
    latest = int(df.period.max())
    if target_year > latest + 1:
        raise HorizonError(
            f'Cannot forecast {target_year}: history ends {latest}; only one year ahead is supported'
        )
    earliest = int(df.period.min())
    if target_year <= earliest:
        raise HorizonError(
            f'Cannot forecast {target_year}: history starts {earliest}; '
            'no prior-year observation anchors that year'
        )
    df['primaryValue'] = pd.to_numeric(df.primaryValue, errors='coerce')
    if df['primaryValue'].isna().any() or (df['primaryValue'] < 0).any():
        raise ValueError('Missing/negative trade values are not zero trade')
    if series is not None:
        partner, commodity = (int(series[0]), int(series[1]))
        df = df[(df.partnerCode == partner) & (df.cmdCode == commodity)]
        if df.empty:
            raise ValueError(f'No history for series partner={partner} commodity={commodity}')

    out = df[KEYS].drop_duplicates().copy()
    out['period'] = target_year
    base = df[KEYS + ['period', 'primaryValue']]
    for lag in range(1, 6):
        past = base.copy()
        past['period'] += lag
        past.rename(columns={'primaryValue': f'primaryValue_lag_{lag}y'}, inplace=True)
        out = out.merge(past, on=KEYS + ['period'], how='left', validate='one_to_one')
    for window in [3, 5]:
        cols = [f'primaryValue_lag_{k}y' for k in range(1, window + 1)]
        out[f'primaryValue_rolling_{window}y_mean'] = out[cols].mean(axis=1)
    out['flow_is_export'] = out.flowCode.eq('X').astype(float)

    macro_cols = [c for c in frame if c.endswith(('_mean', '_vol_std', '_year_end', '_yoy_pct')) and not c.startswith(('primaryValue', 'netWgt'))]
    for c in macro_cols:
        by_year = frame.groupby('period')[c].nunique(dropna=True)
        if (by_year > 1).any():
            raise ValueError(f'Macro column {c} has conflicting annual values')
        macro = frame[['period', c]].copy()
        macro['period'] = pd.to_numeric(macro.period).astype(int) + 1
        macro = macro.groupby('period', as_index=False)[c].first()
        macro.rename(columns={c: c + '_lag_1y'}, inplace=True)
        out = out.merge(macro, on='period', how='left', validate='many_to_one')
    return out
