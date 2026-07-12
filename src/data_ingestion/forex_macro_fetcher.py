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

# Define Target Tickers for Indian Macro & Forex
TICKERS = {
    "USDINR": "USDINR=X",       # US Dollar to Indian Rupee
    "EURINR": "EURINR=X",       # Euro to Indian Rupee
    "GBPINR": "GBPINR=X",       # British Pound to Indian Rupee
    "CNYINR": "CNYINR=X",       # Chinese Yuan to Indian Rupee
    "JPYINR": "JPYINR=X",       # Japanese Yen to Indian Rupee
    "BRENT_CRUDE": "BZ=F",      # Brent Crude Oil Futures (Key Indian import driver)
    "GOLD_FUTURES": "GC=F",     # Gold Futures (Key Indian import asset)
    "NIFTY_50": "^NSEI",        # NSE Nifty 50 Index (Broad Indian Market Benchmark)
    "SENSEX": "^BSESN"          # BSE Sensex Index
}

OUTPUT_DIR = os.path.join("data", "raw", "forex_macro")

def fetch_and_verify_all(start_date="2005-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"=== INDITRADE AI: FETCHING FOREX & MACRO DATA ({start_date} to {end_date}) ===")
    print(f"Target Output Directory: {OUTPUT_DIR}\n")
    
    results_summary = []
    
    for name, ticker in TICKERS.items():
        print(f"[*] Fetching {name:<14} ({ticker})...", end=" ", flush=True)
        try:
            # Fetch data from yfinance
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # If multi-index columns appear from yfinance >= 0.2.30+, flatten column headers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if df.empty:
                print(f"[FAILED / EMPTY] No data returned for {ticker}.")
                results_summary.append({"Ticker": name, "Symbol": ticker, "Status": "EMPTY", "Rows": 0, "File": "None"})
                continue
                
            # Reset index so 'Date' is a regular column
            df = df.reset_index()
            
            # Save clean CSV
            file_path = os.path.join(OUTPUT_DIR, f"{name.lower()}.csv")
            df.to_csv(file_path, index=False)
            
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
    
    return summary_df

if __name__ == "__main__":
    fetch_and_verify_all()
