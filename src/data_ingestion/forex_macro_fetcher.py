"""
IndiTrade AI - Forex, Indian Macro & RBI Proxy Data Ingestion Module
Fetches multi-decade historical daily data using yfinance (Free, no API key required).
Strictly verifies individual datasets, records row counts, date ranges, and saves clean CSVs.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime

TICKERS = {
    "USDINR": "USDINR=X",
    "EURINR": "EURINR=X",
    "GBPINR": "GBPINR=X",
    "JPYINR": "JPYINR=X",
    "BRENT_CRUDE": "BZ=F",
    "GOLD_FUTURES": "GC=F",
    "NIFTY_50": "^NSEI",
    "SENSEX": "^BSESN"
}

OUTPUT_DIR = os.path.join("data", "raw", "forex_macro")

def fetch_and_verify_all(start_date="2005-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== INDITRADE AI: FETCHING FOREX & MACRO DATA ({start_date} to {end_date}) ===")
    print(f"Target Output Directory: {OUTPUT_DIR}\n")

    results_summary = []
    # CNYINR provider returns a degenerate one-row series; excluded from features.
    cny_path = os.path.join(OUTPUT_DIR, "cnyinr.csv")
    if os.path.exists(cny_path):
        os.remove(cny_path)

    for name, ticker in TICKERS.items():
        print(f"[*] Fetching {name:<14} ({ticker})...", end=" ", flush=True)
        try:

            df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=30)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) <= 1:
                print(f"[FAILED / EMPTY] No data returned for {ticker}.")
                results_summary.append({"Ticker": name, "Symbol": ticker, "Status": "EMPTY", "Rows": 0, "File": "None"})
                continue

            df = df.reset_index()

            file_path = os.path.join(OUTPUT_DIR, f"{name.lower()}.csv")
            latest = pd.to_datetime(df["Date"]).max()
            if (pd.Timestamp(end_date) - latest).days > 7:
                raise ValueError(f"Stale market series: latest {latest.date()}")
            temp_path = file_path + ".tmp"
            df.to_csv(temp_path, index=False)
            os.replace(temp_path, file_path)

            row_count = len(df)
            min_date = df["Date"].min().strftime("%Y-%m-%d") if "Date" in df.columns else "N/A"
            max_date = df["Date"].max().strftime("%Y-%m-%d") if "Date" in df.columns else "N/A"

            print(f"[VERIFIED] Rows: {row_count:<5} | Range: {min_date} -> {max_date} | Saved: {file_path}")

            results_summary.append({
                "Ticker": name,
                "Symbol": ticker,
                "Status": "SUCCESS",
                "Rows": row_count,
                "Start_Date": min_date,
                "End_Date": max_date,
                "File": file_path
            })

        except Exception as e:
            print(f"[ERROR] Exception while fetching {ticker}: {str(e)}")
            results_summary.append({"Ticker": name, "Symbol": ticker, "Status": f"ERROR: {str(e)}", "Rows": 0, "File": "None"})

    print("\n=== INDIVIDUAL VERIFICATION SUMMARY TABLE ===")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    if any(row["Status"] != "SUCCESS" for row in results_summary):
        raise RuntimeError("One or more macro refreshes failed; refusing a successful stale run")
    return summary_df

if __name__ == "__main__":
    fetch_and_verify_all()
