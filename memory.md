# 🧠 Antigravity Memory & Project Context (`memory.md`)

*This document serves as the persistent memory and operational context for Antigravity (AI Assistant) working on **IndiTrade AI** with Yash Bajpai. Read this file immediately upon session start or server restart.*

---

## 🛑 Critical Behavioral Rules & Learnings (NEVER VIOLATE)

1. **Rule 1: Real Data First, Loud Synthetic Fallback Only:**
   - NEVER use synthetic/seed data as a silent default or silent fallback.
   - Real data acquisition (`un_downloader.py`, `rbi_downloader.py`, `dgft_pdf_extract.py`, `pib_scraper.py`, `generate_qa_dataset.py`, `trade_features.py`) MUST run and succeed first.
   - Synthetic fallback is allowed **ONLY** as a clearly-logged, loudly-flagged fallback:
     ```text
     [WARNING: USING SYNTHETIC DATA — REAL SOURCE FAILED]
     ```
2. **Rule 2: Disk Verification Before Documentation (Rule 4 from User):**
   - NEVER write "completed X" or "trained X" in `CONTEXT_HANDOFF.md` or logs right after writing code.
   - Only document completion AFTER running `list_dir` / filesystem verification to confirm the output artifact physically exists on disk and checking its size.
3. **Rule 3: Data Provenance Tracking:**
   - Always log whether data/models originated from `REAL` or `SYNTHETIC FALLBACK` in `CONTEXT_HANDOFF.md` under the `Data provenance:` field.
4. **Rule 4: License & Open Source:**
   - The repository uses the **MIT License** (Copyright (c) 2026 Yash Bajpai). Do not revert or change to proprietary terms unless explicitly requested.
5. **Rule 5: LLM Fallback Visibility:**
   - In `llm_qlora.py`, if training falls back from Meta Llama-3.2-1B to TinyLlama (due to HuggingFace gated auth errors or 403 Forbidden), it MUST throw a visible error banner and log so we know exactly which model actually got trained.

---

## 📊 Current Project Stats & Disk Verification

*Last Verified: 2026-07-08*

### 1. Data Pipeline & Date Range (2005–2024 — Full 20 Years)
- **`data/raw/un_comtrade/india_trade_hs2_2005_2024.parquet`**: 4,000 records across 2005–2024 (HS 2-digit chapters).
- **`data/raw/rbi/rbi_macro_indicators_2005_2024.parquet`**: 20 years of macroeconomic indicators (INR/USD exchange rate, Crude Oil index, GDP growth benchmarks).
- **`data/processed/trade_features.parquet`**: **350.8 KB** — Master feature matrix containing 15+ time-series lag features (`lag_1y`, `lag_3y`, `lag_5y`), rolling statistics, and policy flags across 4,000 rows.
- **`data/processed/dgft_policy_chunks.jsonl`**: **4.8 KB** (3 DGFT policy documents).
- **`data/processed/pib_press_releases.jsonl`**: **3.4 KB** (3 PIB trade press releases).
- **`data/processed/policy_qa_dataset.jsonl`**: **7.5 KB** (6 instruction Q&A pairs for LLM fine-tuning).

### 2. Trained Models & Machine Learning Artifacts (`models/`)
- **Module A (TradeFlow Forecast - XGBoost):**
  - **`models/xgboost_trade.pkl`**: **2.23 MB** (Real trained XGBoost Regressor optimized via Optuna 10/150 trials, TimeSeriesSplit CV MAPE: 1.4258).
  - **`models/xgboost_best_params.json`**: **243 bytes** (Stored optimal hyperparameters).
  - **`models/xgboost_trade.onnx`**: Ready for low-memory (<50MB RAM) production inference.
- **Module C (Trade Network Graph - Node2Vec):**
  - **`models/node2vec_embeddings.npy`**: **12 KB** (64-dimensional topological graph embeddings for 20 global partners).
  - **`data/processed/graph_edges.parquet`**: **4.2 KB** (Precomputed graph topology and centrality metrics).
  - **`data/processed/graph_nodes.parquet`**: **2.9 KB**.
- **Module D (AnomalyGuard - Isolation Forest):**
  - **`models/isolation_forest.pkl`**: **1.68 MB** (Real trained unsupervised anomaly detector flagging top 5% historical trade anomalies, bundled with StandardScaler).
- **Module B (PolicyGPT - Llama-3.2-1B QLoRA):**
  - Code written in `src/models/llm_qlora.py` and cloud job in `camber_jobs/job_llm.sh`.
  - Status: Ready for GPU execution once cloud compute is awakened.

### 3. Backend API & Cloud Deployment (`src/backend/`)
- **Framework:** FastAPI (`src/backend/main.py`), Uvicorn server.
- **Endpoints:** `/forecast` (XGBoost), `/query` (RAG / LLM), `/network` (Graph analytics), `/anomaly` (Isolation Forest).
- **Deployment Infrastructure:** `Dockerfile`, `docker-compose.yml`, `render.yaml` (Render 512MB RAM free tier optimized), GitHub Actions (`.github/workflows/deploy.yml`).

---

## ☁️ Computing Environment & Cloud Compute State

- **Current Status:** **ON SLEEP / INACTIVE** 😴
- **Details:** The remote GPU computing environment (Camber Cloud / Lightning AI / T4 GPU instances) is currently on sleep/suspended due to inactivity timeouts (10-min idle limit / 4-hour session limit) and server restart.
- **Operational Strategy:**
  1. **Local Work (Active):** All tabular ML models (XGBoost, Isolation Forest, Node2Vec graph metrics), data pipelines, feature engineering, and backend FastAPI development/testing execute locally on Windows CPU without needing cloud GPU.
  2. **GPU Work (Paused):** For heavy LLM QLoRA fine-tuning (`llm_qlora.py` / `job_llm.sh`), do NOT attempt to run locally on CPU. We must first wake up / re-authenticate the Camber/Lightning GPU instance (`camber login` / studio resume) when Yash is ready to run GPU jobs.

---

## 🗺️ Next Operational Steps

1. **Backend API Testing:** Test FastAPI endpoints locally using the generated disk artifacts (`xgboost_trade.pkl`, `isolation_forest.pkl`, `trade_features.parquet`).
2. **Cloud GPU Wakeup (When needed):** Resume Camber Cloud / Lightning studio to execute full 150-trial XGBoost Optuna job (`job_xgboost.sh`) and Llama-3.2-1B QLoRA fine-tuning (`job_llm.sh`).
3. **Documentation Sync:** Ensure `README.md` and `CONTEXT_HANDOFF.md` remain synced with actual disk states.
