# 🇮🇳 IndiTrade AI
### **AI-Powered Macroeconomic & Trade Policy Intelligence Engine for India**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Phase 1 Complete](https://img.shields.io/badge/Status-Phase%201%20Data%20Ingestion%20Complete-00C853.svg)](#-current-data-inventory--metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**IndiTrade AI** is an advanced, production-grade macroeconomic intelligence and quantitative analysis system tailored specifically for Indian markets and global trade dynamics. By combining high-frequency financial market data with automated policy scraping (UN Comtrade, DGFT circulars, and PIB press releases), IndiTrade AI constructs an institutional-grade data foundation for trade flow modeling, policy analysis, and macro forecasting.

---

## 🏗️ System Architecture & Phase 1 Scope

Phase 1 focuses exclusively on establishing highly reliable, fault-tolerant data ingestion pipelines across four critical pillars:

```
                  +---------------------------------------+
                  |            IndiTrade AI               |
                  |     Institutional Data Pipeline       |
                  +---------------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
+------------------+         +------------------+         +------------------+
| Forex & Indian   |         | UN Comtrade M49  |         | Official Policy  |
| Macro (`yfinance`)|         | Bilateral Trade  |         | Circulars & News |
+------------------+         +------------------+         +------------------+
         |                            |                            |
         |---> 5 Currency Pairs       |---> HS 2-Digit Flow        |---> DGFT Circulars
         |---> Nifty 50 & Sensex      |---> Top 20 Partners        |     (PDF -> JSONL)
         |---> Brent Crude & Gold     |---> 10-Yr Historical       |---> PIB Live Releases
         v                            v                            v
+----------------------------------------------------------------------------+
|             Standardized Data Storage Layer (`data/raw/` & `data/processed/`)  |
+----------------------------------------------------------------------------+
```

### 1. 💱 Foreign Exchange & Indian Macro Indices (`src/data_ingestion/forex_macro_fetcher.py`)
* **Currency Pairs Ingested**: `USD/INR`, `EUR/INR`, `GBP/INR`, `JPY/INR`, `CNY/INR`.
* **Macro Benchmarks**: `Nifty 50 (^NSEI)`, `BSE Sensex (^BSESN)`, `Gold Futures (GC=F)`, `Brent Crude Oil (BZ=F)`.
* **Coverage Range**: `2005-01-03` through `July 2026` (~21+ years of historical daily data).
* **Validation**: Automated retry logic with strict schema and null-check validation.

### 2. 🌍 UN Comtrade Bilateral Trade Flows (`src/data_ingestion/un_downloader.py`)
* **API Integration**: Uses official `comtradeapicall` SDK with API key authorization.
* **Scope**: India (`M49: 699`) bilateral trade across **Top 20 Trade Partners** (`USA, China, UAE, Saudi Arabia, Russia, Germany, UK, etc.`).
* **Granularity**: `HS 2-Digit` commodity chapters for both **Imports (`M`)** and **Exports (`X`)**.
* **Temporal Coverage**: 10-year historical window (`2015–2024`), optimized to respect daily 500-call API thresholds with polite 1.5s rate delays.

### 3. 📜 DGFT Official Notifications & Circulars (`src/data_ingestion/dgft_pdf_processor.py`)
* **Live Scraping**: Automated ASP.NET pagination harvester across `Notification`, `Public Notice`, and `Trade Notice` endpoints from `dgft.gov.in`.
* **PDF Archival**: Downloads and stores **500 official circular PDFs** locally (`data/raw/dgft_notifications/pdfs/`).
* **Text Extraction & Chunking**: Uses `pdfplumber` to strip headers/footers, normalize layout, and output structured, machine-readable policy chunks (`data/processed/dgft_policy_chunks.jsonl`).

### 4. 📰 PIB Press Release Intelligence (`src/data_ingestion/pib_scraper.py`)
* **Policy Filtering**: Scrapes `pib.gov.in` (`BeautifulSoup4`) with custom headers and strict 1s rate delays.
* **Domain Focus**: Automatically identifies and categorizes articles into `Trade & Tariff Policy`, `Macro & Finance Policy`, and `Industrial & Economic Policy`.
* **Clean HTML-to-Text**: Strips JavaScript containers, navigation prompts, and boilerplates to preserve pure policy text.

---

## 📊 Current Data Inventory & Metrics

All pipelines have been executed, verified, and reconciled on disk as of **July 2026**:

| Data Domain | Source / Endpoint | Record Count | File Format & Count | Total Storage | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Forex Pairs** | Yahoo Finance (`yfinance`) | **21,966 rows** | `CSV` (5 files) | `1.85 MB` | ✅ **100% Complete** |
| **Macro Indices** | Yahoo Finance (`yfinance`) | **20,027 rows** | `CSV` (4 files) | `1.47 MB` | ✅ **100% Complete** |
| **UN Comtrade Trade** | UN Comtrade API v1 | **62,026 rows** | `Parquet` (1 file, 47 cols) | `1.11 MB` | ✅ **100% Complete** |
| **DGFT Circular PDFs** | `dgft.gov.in` Circulars | **500 PDFs** | `PDF` Binary Documents | `346.02 MB` | ✅ **100% Complete** |
| **DGFT Policy Chunks** | `pdfplumber` Processed | **117 chunks** | `JSONL` (UTF-8 encoded) | `5.89 MB` | ✅ **100% Complete** |
| **PIB Press Releases** | `pib.gov.in` Live Portals | **65 articles** | `JSONL` (UTF-8 encoded) | `1.30 MB` | ✅ **100% Complete** |

> **Summary Volume**: **104,136 tabular rows / policy chunks** and **500 raw circular PDFs** (`~357.6 MB total storage`).

---

## 🚀 Quickstart & Installation

### Prerequisites
* **Python**: Version `3.10` or higher.
* **Operating System**: Windows (`PowerShell`) or Linux/macOS.

### 1. Clone & Setup Workspace
```powershell
# Clone repository
git clone https://github.com/Yash1bajpai/IndiTrade_AI.git
cd IndiTrade_AI

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template configuration and insert your API keys (`.env` is excluded from git):
```powershell
Copy-Item .env.example .env
```
Ensure your `.env` contains:
```env
COMTRADE_API_KEY="your_actual_un_comtrade_key"
GROQ_API_KEY="your_actual_groq_key"
HF_TOKEN="your_actual_huggingface_token"
```

### 3. Run Ingestion Pipelines
Execute any data ingestion module independently:
```powershell
# 1. Fetch Forex & Indian Macro daily timeseries (2005-2026)
python src/data_ingestion/forex_macro_fetcher.py

# 2. Download UN Comtrade HS 2-Digit bilateral trade flows (2015-2024)
python src/data_ingestion/un_downloader.py

# 3. Scrape DGFT circulars, download PDFs, and generate text chunks
python src/data_ingestion/dgft_pdf_processor.py

# 4. Scrape real-time trade & tariff press releases from PIB
python src/data_ingestion/pib_scraper.py
```

---

## 📁 Repository Directory Structure

```text
IndiTrade_AI/
├── .env.example               # Template environment configuration (no real secrets)
├── .gitignore                 # Strict rules excluding secrets, data, and handoff logs
├── README.md                  # System overview and operational guide
├── requirements.txt           # Python package dependencies
├── config/                    # System configuration schemas
├── data/                      # Local data repository (Excluded via .gitignore)
│   ├── cache/                 # Temporary request caches
│   ├── processed/             # Cleaned JSONL policy chunks (`dgft`, `pib`)
│   └── raw/                   # Raw downloads (`forex_macro/`, `un_comtrade/`, `dgft/pdfs/`)
├── logs/                      # Application runtime logs
├── notebooks/                 # Exploratory Data Analysis (EDA) notebooks
├── scripts/                   # Helper utility scripts and inventory verifiers
├── src/
│   ├── data_ingestion/        # Core scrapers & API fetchers
│   │   ├── dgft_pdf_processor.py   # DGFT official notification scraper & chunker
│   │   ├── forex_macro_fetcher.py  # Yahoo finance daily timeseries harvester
│   │   ├── pib_scraper.py          # PIB live press release intelligence scraper
│   │   └── un_downloader.py        # UN Comtrade multi-year bilateral flow downloader
│   ├── feature_engineering/   # Feature engineering pipeline (Phase 2)
│   ├── models/                # Analytical & predictive models (Phase 3)
│   └── utils/                 # Common logger & formatting helpers
└── tests/                     # Unit & integration test suite
```

---

## 🔒 Security & Data Governance (`.gitignore` Rules)

To maintain a lightweight git tree and strictly prevent secret leakage, `IndiTrade_AI` adheres to the following governance rules:
1. **Zero-Secret Commit Guarantee**: `.env`, `.env.local`, and all token stores are permanently blocked by `.gitignore`. Only `.env.example` with placeholder strings is tracked.
2. **Data & Binary Isolation**: All downloaded PDFs (`~346 MB`), `Parquet` tables, and processed `JSONL` datasets inside `data/raw/*` and `data/processed/*` are excluded from version control. Only empty directory placeholders (`.gitkeep`) are committed.
3. **Session Handoff Protection**: Internal tracking and handoff files (`CONTEXT_HANDOFF.md`, `context_handoff.md`) are explicitly excluded to keep the public repository history clean and focused purely on code and documentation.

---

## 📄 License & Maintainer
* **Author / Lead**: Yash Bajpai (`@Yash1bajpai`)
* **License**: Open-source under the MIT License.
