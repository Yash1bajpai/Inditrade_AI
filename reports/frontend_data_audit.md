# IndiTrade AI — Frontend Data-Usage Audit

**Audit date:** 2026-07-21 (rescanned against the current frontend source)  
**Scope:** Frontend data flow only (`frontend/src/app/page.tsx`), validated read-only against the API response contracts and the processed project datasets.  
**Code changes:** None. This document is an audit report, not an implementation.

## Executive summary

The dashboard correctly formats several values (notably anomaly and network tooltip monetary values in USD billions), but it currently presents some incompatible or incomplete data as if it were directly comparable business information.

The highest-risk issue is the forecast chart: the current frontend does not load real forecast history, displays hard-coded mock values, and appends a raw-USD model output to a chart labelled in USD billions. The form supplies only three of the model's 42 features, while the API sets the rest to zero. The displayed forecast must not be interpreted as a global India trade forecast.

## Verified data contracts

| Frontend area | Endpoint | Frontend fields used | Contract/use verdict |
|---|---|---|---|
| Forecast history | No request is made | Hard-coded `chartData` | Incorrect: the real history endpoint and processed trade history are not used. |
| Forecast submission | `POST /api/forecast/` | `usd_inr`, `crude_price`, `year` | Request field names are correct, but inputs are insufficient for the model's intended record-level prediction. |
| Forecast output | `forecasted_trade_value_usd`, `feature_importance` | Appends raw forecast to a chart labelled USD billions | Incorrect: missing `/ 1e9` conversion; semantic interpretation is also not valid. |
| Historical anomalies | `GET /api/anomaly/historical` | `date`, `value`, `partner`, `commodity`, `reason` | Values are correctly converted from USD to USD billions. Ranking/confidence is not exposed. |
| Trade network | `GET /api/network/` | `country_name`, `x`, `y`, `val`, `trade_volume` | PCA and monetary rendering are correct. Volume represents flagged-anomaly volume, not total trade volume. |
| Country history | No request is made | Random six-month `mockData` | Incorrect: the modal shows fabricated values instead of project data. |
| Policy chat | `POST /api/query/` | `answer`, `source`, `citation` | Request and answer/citation mapping are correct; `source` is stored but never shown. |

## Findings

### F-01 — Forecast chart displays mock history and a raw-USD prediction as USD billions

**Severity: Critical**

The current frontend does not request `GET /api/forecast/history`. Instead, it starts with three hard-coded values. On forecast, it sends only `usd_inr`, `crude_price`, and `year` to a model whose metadata identifies it as **“XGBoost Bilateral Trade Flow Forecast”** with 42 input features; the API fills the remaining features with zero.

It then appends `forecasted_trade_value_usd` directly to a Y-axis formatted as `$...B`. The API output is raw USD, so this misses a required `/ 1e9` conversion.

**Impact:** The chart is always built on fabricated history, and a result such as `$10,000,000,000` is rendered as `$10,000,000,000B` rather than `$10B`. It is also a per-record model output, not a valid aggregate/global forecast. The graph and headline R² can lead users to make incorrect policy or business decisions.

**Evidence:**

- Frontend mock history: `frontend/src/app/page.tsx:93-97`
- Frontend input/request and raw-USD chart insertion: `frontend/src/app/page.tsx:180-204`
- Frontend chart labels every value as USD billions: `frontend/src/app/page.tsx:279-290`
- The project's real global history covers 2015–2024. Latest aggregate totals are 2021 `$623.753B`, 2022 `$782.194B`, 2023 `$1,468.297B`, and 2024 `$1,522.327B`.

**Required fix (choose one):**

1. Replace the mock series with `GET /api/forecast/history`, keep every plotted value in USD billions, and always convert `forecasted_trade_value_usd / 1e9`; then
2. Make the model and API produce a genuinely global aggregate forecast, using aggregate-compatible features and training data; or
3. Make the UI a bilateral-flow forecaster: require partner, HS/commodity, flow direction, weights/quantity, and valid lag/rolling values, then compare against the matching bilateral historical series.

Do not present the current prediction as a global trade forecast until one of these approaches is implemented.

### F-02 — Forecast history is permanently fabricated, not merely a fallback

**Severity: High**

The UI always uses `2022: 453.2`, `2023: 437.1`, and `2024: 442.8` USD billions. No frontend request loads the actual history. These values do not match the processed data totals (`782.194`, `1,468.297`, and `1,522.327` USD billions respectively).

**Impact:** Users always see incorrect historical figures, including after a successful forecast.

**Evidence:** `frontend/src/app/page.tsx:109-113`

**Required fix:** Fetch the real history on page load, start with an empty/loading state, and show a recoverable error if the request fails. Do not use mock figures in a production analytics chart.

### F-03 — “Prediction Drivers” are global model importances, not this prediction’s drivers

**Severity: High**

The API returns the trained model's static `feature_importances_`; these values do not change for different user inputs. The UI labels them “Prediction Drivers,” which implies a local explanation for the current forecast.

**Impact:** The explanation is misleading. It cannot explain why a specific forecast changed.

**Evidence:** `frontend/src/app/page.tsx:294-307`

**Required fix:** Rename this block to “Global Model Feature Importance” and add explanatory copy, or replace it with a per-prediction explanation method (for example, SHAP) returned by the API.

### F-04 — Network “trade volume” is anomaly-only volume, not total trade

**Severity: High**

The network API derives each node's `trade_volume` by summing `primaryValue` from `flagged_trade_anomalies.csv`, not from the full trade dataset. In this project, all 18 displayed network partners receive a value from that anomaly file. The sum of all flagged-anomaly values is only about **34.3%** of the full processed trade value.

**Impact:** The choropleth and scatter tooltip label the value as “Volume,” so users can reasonably assume it is total bilateral trade. It is actually volume among anomalous rows only.

**Evidence:** Frontend presentation: `frontend/src/app/page.tsx:379-446`

**Required fix:** Either provide full-trade country volumes, or rename the UI and legend to “Flagged anomaly trade value (USD)” and document the aggregation period.

### F-05 — Country drill-down is fabricated random data

**Severity: High**

The country modal generates six random monthly values every time it renders and labels them as a country “6 Month Trend.” It does not call the available country-history API and does not use any project trade data.

**Impact:** The dashboard displays invented monetary figures as real country trade history. Each reopen can show a different trend for the same country.

**Evidence:**

- Mock-data generation: `frontend/src/app/page.tsx:36-42`
- Modal chart rendering: `frontend/src/app/page.tsx:53-65`
- Map country selection: `frontend/src/app/page.tsx:397-407`

**Required fix:** Replace mock data with `GET /api/network/history/:country` and render its yearly USD-billion values. Return both a canonical data key (for example, `partner_id` or original `partnerDesc`) and a display/map name from the API. Use the canonical key for the history request and the display name only in the UI.

**Known identifier safeguard:** If the API is reconnected, preserve the original data identifiers. `USA` → `United States of America` and `Russian Federation` → `Russia` are display-name mappings that otherwise produce no history match.

### F-06 — Anomaly confidence/ranking data is discarded

**Severity: Medium**

The source anomaly data contains `anomaly_score`, but the API does not return it and the frontend cannot show severity or confidence. The API correctly selects the top 50 scores, then sorts them chronologically. The frontend table shows the first five chronological entries among those 50—not the five highest-scoring entries.

**Impact:** Users cannot distinguish the most severe anomalies, and the table ordering can be misunderstood.

**Evidence:** `frontend/src/app/page.tsx:320-368`

**Required fix:** Return `anomaly_score`; show it as severity/confidence; add an explicit sort/order label and, ideally, controls for “highest score” versus “chronological.”

### F-07 — Query source is captured but not displayed

**Severity: Low**

The chat response's `source` field is saved to message state, but the UI renders only the citation. Users cannot tell whether the answer came from the primary provider or a fallback.

**Evidence:** `frontend/src/app/page.tsx:166` and `frontend/src/app/page.tsx:513-517`

**Required fix:** Render a small source/status label next to each assistant answer, including clear error/fallback states.

### F-08 — API errors can be mistaken for valid empty data

**Severity: Medium**

The frontend does not check `response.ok`, validate response shapes, or provide section-level error messages. For example, a failed history request leaves the fallback chart, while a failed network request produces an empty visualization.

**Evidence:** `frontend/src/app/page.tsx:110-118` and `frontend/src/app/page.tsx:150-204`

**Required fix:** Add typed API response models, `response.ok` checks, error state per data panel, and request cancellation or request IDs to prevent stale responses overwriting current selections.

## Data usage that is currently correct

- Anomaly `value` is raw USD and the frontend correctly converts it to USD billions for the chart axis: `frontend/src/app/page.tsx:384`.
- Network PCA coordinates use `x` and `y`; bubble size uses normalized `val`; tooltip converts raw `trade_volume` to USD billions: `frontend/src/app/page.tsx:428-446`.
- Chat request `{ question: userMessage }`, answer, and citation names match the API contract: `frontend/src/app/page.tsx:160-166`.

## Additional frontend improvements from the rescan

### F-09 — The frontend does not currently pass lint or TypeScript production checks

**Severity: High**

The previously run `npm run lint`, `npx tsc --noEmit`, and `npm run build` fail. Causes include invalid Framer Motion transition/variant typing, the unsupported `title` prop on the Lucide icon, `any` API types, and a component declared inside render.

**Required fix:** Resolve all lint and TypeScript errors before deployment; add a CI job that runs `npm run lint`, `npx tsc --noEmit`, and `npm run build`.

### F-10 — Deployment configuration is hard-coded for local development

**Severity: High**

`API_BASE` is `http://localhost:8000/api`. A deployed browser will call the visitor's own machine, and an HTTPS frontend can be blocked for mixed content. The map also depends on an external `unpkg.com` geography URL at runtime.

**Required fix:** Use a `NEXT_PUBLIC_API_BASE_URL` environment variable or a same-origin Next.js proxy. Host/version the geography file in `public/` or add a reliable fallback and integrity/version policy.

**Evidence:** `frontend/src/app/page.tsx:11-12`.

### F-11 — Mobile and accessibility support is incomplete

**Severity: Medium**

The stylesheet has no mobile media queries. The analytics panels contain fixed two-column inline grids, while the chat drawer relies on mouse-only resizing. The modal lacks dialog semantics, Escape handling, focus management, and labelled close controls. Table rows, map regions, and icon controls are not reliably keyboard-operable. Input labels are not linked through `htmlFor`/`id`.

**Required fix:** Add responsive breakpoints, touch/keyboard alternatives, `aria-label`s, linked form labels, accessible modal behavior, and textual summaries/downloads for charts.

### F-12 — Dashboard credibility and data communication need improvement

**Severity: Medium**

- The R-squared badge renders as `RÂ²`, indicating an encoding defect.
- The metric is model test **log-scale** R-squared, so the label should disclose that scope instead of implying general forecast accuracy.
- `forecast` is saved in state but never shown as a headline result; users must infer it from a chart.
- The anomaly-table click only pre-fills a chat prompt; it does not actually start the requested investigation.

**Required fix:** Fix encoding, state the metric definition, render a clearly formatted result with unit and confidence/limitations, and either submit the anomaly investigation automatically or change the instruction text.

### F-13 — Product hygiene gaps

**Severity: Low**

- Page metadata still says “Create Next App.”
- There are no frontend test files for transformations, error states, or core user flows.
- `npm audit --omit=dev` previously reported seven dependency vulnerabilities (five high), including a D3 chain used by `react-simple-maps`; automatic remediation is breaking and needs deliberate upgrades.

**Required fix:** Set production metadata, add unit/component/e2e coverage for each data panel, and create a dependency-upgrade plan with regression tests.

## Implementation priority

1. **Block/replace F-01** before presenting forecast results as global trade predictions.
2. Fix F-02 and F-05 so displayed values and country drill-downs are reliable.
3. Correct F-04 terminology or change its data source to full trade data.
4. Expose anomaly score and transparent ordering (F-06).
5. Add resilient typed request handling and visible panel errors (F-08).
6. Improve explanation and provider transparency (F-03, F-07).

## Validation performed

- Inspected the frontend's fetch calls, transformations, and rendering bindings.
- Verified response field names and units against the corresponding API endpoints.
- Loaded and checked the three processed datasets used by these endpoints:
  - `data/processed/trade_features.parquet` (32,072 rows; 2015–2024)
  - `data/processed/flagged_trade_anomalies.csv` (321 rows)
  - `data/processed/node2vec_trade_embeddings.parquet` (115 nodes; 18 partner nodes)
- No external network data, backend mutation, or source-code modification was performed.
