# Trade data repair and dashboard branch

Status: repair prepared, not executed. A working UN Comtrade subscription key is required. No repaired data or model has been promoted.

## Source gaps
The retained source has 1,072 non-total second-partner cells across 16 annual partner/flow slices in 2023-2025. France and Switzerland each lack 2025 imports and exports, adding four slices. Refetch all 20 complete slices rather than deleting bad cells or filling them with zero.

The repair script writes a separate candidate parquet and a SHA256 provenance report. It checks canonical partner2/customs/transport grain, response identity, duplicates, HS2 codes, value presence, and an independent source count. Source failure, missing count, or truncation stops repair. Original source files, releases and model artifacts remain unchanged.

```
python scripts/repair_trade_data.py --source SOURCE.parquet --output CANDIDATE.parquet --plan-only
python scripts/repair_trade_data.py --source SOURCE.parquet --output CANDIDATE.parquet
```

Provide COMTRADE_API_KEY through a secure environment, never in source control. This does not rebuild production feature files, refresh deployment, or retrain a model. Those are separate steps after candidate review.

## Forecast validation
The CPU-only past-data diagnostic achieved log R2 0.831, but dollar MAE ($95.46M) and RMSE ($1.166B) were worse than a previous-year baseline ($63.75M MAE and $605.99M RMSE). It used a rolling one-year evaluation, not a frozen-origin multi-year forecast. It is not production approval.

The deployed legacy model is retained and its forecast accuracy remains unvalidated. Its old retrospective R2 is not a prospective accuracy claim. Do not promote the diagnostic model without repaired data and validation that includes dollar errors and flow/horizon-safe inference.

## Branch dashboard
A responsive trade-desk header, coverage-status card and clear scenario workspace put data limits ahead of model output. A new /forecast/data_quality endpoint reports the current loaded feature dataset's grain checks and 2025 France/Switzerland flow presence. These narrow coverage checks do not establish source freshness, all commodity completeness, or model validation. A missing endpoint remains unknown, not a pass. Missing partner/commodity data disables forecast submission.

Main and the live production site are not changed by these branch edits.
