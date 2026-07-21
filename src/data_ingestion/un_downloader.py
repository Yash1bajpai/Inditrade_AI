"""
IndiTrade AI - UN Comtrade HS 2-Digit Data Ingestion Module
Fetches bilateral trade flows (Imports 'M' & Exports 'X') between India (699)
and its top 20 trade partners using exact UN M49 numeric codes for years 2015-2024.

Features:
- Dual API Key Rotation & Quota Management (COMTRADE_API_KEY1, COMTRADE_API_KEY2).
- Strict 403 / 429 error handling: Logs exact error and STOPS immediately without synthetic substitution.
- 1.5s sleep interval between requests to honor API rate limits.
- Test mode: Fetches strictly 1 partner and 1 year first for user approval before full 400-iteration loop.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
import comtradeapicall

load_dotenv()

def get_api_keys():
    keys = []
    k1 = os.getenv("COMTRADE_API_KEY1")
    k2 = os.getenv("COMTRADE_API_KEY2")
    k_single = os.getenv("COMTRADE_API_KEY")

    if k1 and k1 != "your_comtrade_api_key_here":
        keys.append(k1.strip())
    if k2 and k2 != "your_comtrade_api_key_here" and k2.strip() not in keys:
        keys.append(k2.strip())
    if k_single and k_single != "your_comtrade_api_key_here" and k_single.strip() not in keys:
        keys.append(k_single.strip())

    if not keys:
        raise ValueError("[ERROR] No valid UN Comtrade API keys found in .env file.")
    return keys

REPORTER_INDIA = "699"

TOP_20_PARTNERS = {
    "842": "USA",
    "156": "China",
    "784": "United Arab Emirates",
    "682": "Saudi Arabia",
    "368": "Iraq",
    "702": "Singapore",
    "344": "Hong Kong",
    "360": "Indonesia",
    "410": "South Korea",
    "036": "Australia",
    "276": "Germany",
    "392": "Japan",
    "756": "Switzerland",
    "458": "Malaysia",
    "826": "United Kingdom",
    "643": "Russian Federation",
    "528": "Netherlands",
    "056": "Belgium",
    "250": "France",
    "704": "Vietnam"
}

OUTPUT_PARQUET = os.path.join("data", "raw", "un_comtrade", "india_trade_hs2_2015_2024.parquet")

class ComtradeFetcher:
    def __init__(self):
        self.keys = get_api_keys()
        self.current_key_idx = 0
        self.exhausted_keys = set()
        print(f"[INIT] Loaded {len(self.keys)} UN Comtrade API Key(s) for rotation/redundancy.")

    def get_active_key(self):
        if len(self.exhausted_keys) >= len(self.keys):
            return None
        return self.keys[self.current_key_idx]

    def rotate_key(self, reason="Rate/Quota limit hit"):
        old_idx = self.current_key_idx
        self.exhausted_keys.add(old_idx)
        print(f"\n[KEY ROTATION] Key #{old_idx+1} hit issue ({reason}). Rotating to next key...")

        for idx in range(len(self.keys)):
            if idx not in self.exhausted_keys:
                self.current_key_idx = idx
                print(f"[KEY ROTATION] Switched to Key #{self.current_key_idx+1}.")
                return self.keys[self.current_key_idx]

        print("\n[CRITICAL ERROR / STOPPING] All available UN Comtrade API keys are exhausted or rate limited (403/429)!")
        return None

    def fetch_slice(self, partner_code, year, flow_code):
        """
        Fetches a single slice: Reporter=699 (India), Partner=partner_code, Year=year, Flow=flow_code, HS 2-digit (AG2).
        """
        key = self.get_active_key()
        if not key:
            return None, "ALL_KEYS_EXHAUSTED"

        try:

            df = comtradeapicall.getFinalData(
                subscription_key=key,
                typeCode="C",
                freqCode="A",
                clCode="HS",
                period=str(year),
                reporterCode=REPORTER_INDIA,
                cmdCode="AG2",
                flowCode=flow_code,
                partnerCode=str(partner_code),
                partner2Code=None,
                customsCode=None,
                motCode=None
            )

            if isinstance(df, pd.DataFrame):
                return df, "SUCCESS"
            else:
                return pd.DataFrame(), "EMPTY_OR_UNFORMATTED"

        except Exception as e:
            err_str = str(e)
            if "403" in err_str or "429" in err_str or "Quota" in err_str or "Rate limit" in err_str:
                print(f"\n[API LIMIT DETECTED] Error: {err_str}")
                new_key = self.rotate_key(reason=err_str)
                if new_key:

                    return self.fetch_slice(partner_code, year, flow_code)
                else:
                    return None, f"403/429_EXHAUSTED: {err_str}"
            else:
                print(f"[FETCH ERROR] Partner {partner_code} ({TOP_20_PARTNERS.get(str(partner_code), '')}) | Year {year} | Flow {flow_code} -> {err_str}")
                return pd.DataFrame(), f"ERROR: {err_str}"

def run_test_fetch(partner_code="842", year="2023", flow_code="M"):
    """
    Test fetch for strictly 1 partner and 1 year as requested before full 400-iteration loop.
    Prints exact df.shape and verification details for user approval.
    """
    print(f"\n=== UN COMTRADE TEST FETCH (Strictly 1 Partner, 1 Year) ===")
    partner_name = TOP_20_PARTNERS.get(str(partner_code), "Unknown")
    print(f"Target: India (699) <-> {partner_name} (M49: {partner_code}) | Year: {year} | Flow: {flow_code} | HS 2-Digit")

    fetcher = ComtradeFetcher()
    df, status = fetcher.fetch_slice(partner_code=partner_code, year=year, flow_code=flow_code)

    if df is None:
        print(f"\n[STOPPED ON ERROR] Test fetch stopped due to API exhaustion/error: {status}")
        return None

    print(f"\n[TEST FETCH RESULT] Status: {status}")
    print(f"[df.shape] EXACT SHAPE: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

    if not df.empty:
        print("\n--- SAMPLE DATA (First 5 Rows, Key Columns) ---")
        display_cols = [c for c in ["period", "reporterCode", "reporterDesc", "partnerCode", "partnerDesc", "flowCode", "flowDesc", "cmdCode", "cmdDesc", "primaryValue"] if c in df.columns]
        if display_cols:
            print(df[display_cols].head(5).to_string(index=False))
        else:
            print(df.head(5).to_string(index=False))
    else:
        print("[NOTICE] Returned dataframe is empty.")

    return df

def run_full_loop():
    """
    Runs the full 400-iteration loop:
    20 partners * 10 years (2015-2024) * 2 flows ('M', 'X') = 400 iterations.
    Sleeps 1.5s between calls to strictly respect API rate limits.
    Stops immediately on 403/429 without synthetic data substitution.
    """
    print("\n=== STARTING FULL 400-ITERATION UN COMTRADE FETCH LOOP ===")
    fetcher = ComtradeFetcher()

    all_dfs = []
    iteration = 0
    total_iterations = len(TOP_20_PARTNERS) * 10 * 2
    stopped_early = False

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

    for partner_code, partner_name in TOP_20_PARTNERS.items():
        if stopped_early:
            break
        for year in range(2015, 2025):
            if stopped_early:
                break
            for flow in ["M", "X"]:
                iteration += 1
                flow_label = "Import (M)" if flow == "M" else "Export (X)"
                print(f"[{iteration:03d}/{total_iterations}] Fetching India <-> {partner_name:<20} ({partner_code}) | {year} | {flow_label}...", end=" ", flush=True)

                df, status = fetcher.fetch_slice(partner_code=partner_code, year=year, flow_code=flow)

                if df is None:

                    print(f"\n[CRITICAL STOP] API Rate/Quota Exceeded ({status}). Stopping full loop cleanly.")
                    stopped_early = True
                    break

                row_count = len(df)
                print(f"Rows: {row_count:<4} | Status: {status}")

                if not df.empty:
                    all_dfs.append(df)

                time.sleep(1.5)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"\n=== FULL FETCH SAVED ===")
        print(f"File: {OUTPUT_PARQUET}")
        print(f"Total Combined Rows: {len(final_df)} | Columns: {len(final_df.columns)}")
    else:
        print("\n[WARNING] No data was fetched across the iterations.")

if __name__ == "__main__":

    run_test_fetch(partner_code="842", year="2023", flow_code="M")

