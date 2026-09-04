# Inditrade AI - Project Graph Memory

This document serves as the core graph memory and context for AI assistants (like Claude) working on the Inditrade AI project. It outlines the architecture, locked constraints, recent architectural migrations, and API contracts.

## 1. Tech Stack
*   **Frontend:** Next.js (App Router), React, TypeScript, Recharts (charts), react-simple-maps (with local `/countries-110m.json` to guarantee offline/restricted network reliability).
    *   **Port:** `3000`
    *   **Main path:** `frontend/`
*   **Backend:** Python, FastAPI, Pandas, XGBoost (Forecasting), Node2Vec (Network embeddings).
    *   **Port:** `8000`
    *   **Main path:** `src/backend/`
*   **Data Structure:** Raw trade data is processed into `.parquet` files (e.g., `trade_features.parquet` downloaded via Github release). Small processed datasets (`flagged_trade_anomalies.csv` and `node2vec_trade_embeddings.parquet`) are tracked in Git and copied into the Docker image directly.

## 2. Core Architecture & Routing (Frontend)
The frontend was recently migrated from a monolithic single-page React app to **Next.js App Router**:
*   `frontend/src/app/layout.tsx`: Server-side root layout. Boilerplate Geist font loader removed to prevent internet-dependent build failures.
*   `frontend/src/app/page.tsx`: The main dashboard containing the Globe Heatmap, Country Drill-Down Drawer, Anomaly scatter charts, the Forecast Panel, and the **Vanijya Chatbot** root (`<div id="vanijya-chat-root">` with its `useEffect` singleton logic).
    *   *Locked Constraint:* The Vanijya chat root must remain in `page.tsx` (the only page route) and must keep its explicit DOM injection so it works with the Chat endpoints.
*   `frontend/src/app/page.tsx`: The main dashboard containing the Globe Heatmap, Country Drill-Down Drawer, Anomaly scatter charts, and the Forecast Panel.

## 3. API Endpoints & Contracts (Backend)
The FastAPI backend (`src/backend/api/forecast.py`) exposes several endpoints that the frontend heavily relies on.
*   **CORS Configurations**: Allowed origins must explicitly permit both `http://localhost:3000` and `http://127.0.0.1:3000` to prevent browser blockages on client-side requests.
*   `GET /api/forecast/valid_combinations`: Returns `{ "partners": [...], "map": { "partnerCode": ["cmdCode1", ...] } }`. Used to strictly filter which commodities can be forecasted for which partner.
*   `GET /api/forecast/history?partner_code={code}&commodity_code={code}`: Returns historical trend for a specific partner/commodity combo.
*   `GET /api/forecast/year_breakdown?year={year}&group_by={partner|commodity}`: Returns aggregated data for a specific year.
*   `GET /api/forecast/country_series?partner_code={code}`: Returns historical yearly trade data (`import_billions`, `export_billions`, `value_billions`) and top commodities for a specific country.
    *   *Locked Constraint:* Always pass the numeric M49 code (e.g., `"643"`) to this endpoint, NOT the country name (e.g., `"Russia"`). The backend uses `resolve_partner_code` as a fallback, but the frontend heatmap `onClick` is wired to pass the code.
*   `GET /api/forecast/partner_signature?partner_code={code}`: Returns the top traded commodities for a partner. Used on the frontend to auto-select the #1 commodity when a user changes the partner dropdown.
*   `POST /api/forecast/`: Runs the XGBoost prediction.
    *   *Locked Constraint (Defense-in-depth):* Before running the model, the backend validates if the `(partner_code, commodity_code)` pair actually exists in historical data. If invalid, it returns a 400 error with a `suggested_commodities` array. The frontend catches this and renders clickable chips to auto-correct the user's input.

## 4. UI/UX Rules & Locked Behaviors
*   **Tooltips & Formatting:** All Recharts `<Tooltip>` components and `<YAxis>` ticks must use the `formatMoney` utility to display values cleanly (e.g., `$9.68B` instead of `9.6820595309`). For country history, we use a `ComposedChart` with `<Bar>` for Imports/Exports and `<Line>` for Total Trade.
*   **Forecast Dropdowns (The 5 Rules):**
    1. Fetches `/valid_combinations` on mount.
    2. Commodity `<select>` options dynamically update based on `validMap[selectedPartnerCode]`.
    3. `CMD_MAP` and `PARTNER_MAP` are hardcoded in the frontend to safely map codes to full names (e.g., `'27'` -> `'Mineral Fuels'`).
    4. Auto-selects the top commodity via `/partner_signature` when the partner changes.
    5. Forecast Years are strictly limited to `[2025, 2026]`.
*   **Styling:** Vanilla CSS modules. Dark theme by default (`NIGHT_SLATE`, `CARD_SURFACE`, `MINTED_BRASS`, `CRIMSON_WAX`). Avoid Tailwind. Use CSS animations (`.floating-node` in `globals.css`) instead of Framer Motion for simple infinite loops to reduce JS overhead.

## 5. Known Data Workarounds
*   To generate Excel exports for the user (Imports vs. Exports lists), the raw `trade_features.parquet` file is processed by splitting `flowCode == 'M'` (Imports) and `flowCode == 'X'` (Exports), grouping by the raw `partnerDesc` to guarantee 100% name coverage without relying on the limited top-18 `PARTNER_MAP`.

## DO NOT DO:
*   Do not retrain the XGBoost model manually. It is refreshed automatically by the monthly GitHub Actions workflow (`retrain_models.yml`) using the 2015-2024 baseline; manual retraining risks diverging the deployed artifact from the CI-produced one.
*   Do not touch the `vanijya` chat initialization logic in `page.tsx` or `ClientShell.tsx` (it injects DOM elements explicitly to ensure it runs well with the Chat endpoints).
*   Do not replace native standard CSS with Tailwind without explicit instruction.
