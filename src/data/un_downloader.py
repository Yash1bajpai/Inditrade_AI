"""
UN Comtrade Data Downloader for IndiTrade AI.

Fetches Indian export/import trade data (2005-2024) at HS 2-digit commodity level.
Complies with UN Comtrade Oct 2022 pricing restructuring:
- Free tier limit: 500 API calls per day
- Max records per call: 100,000
- Uses paginated queries by year and HS 2-digit commodity chapters.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv

# Try importing comtradeapicall, fallback to direct requests if needed
try:
    import comtradeapicall
    HAS_COMTRADE_LIB = True
except ImportError:
    HAS_COMTRADE_LIB = False
    import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("UNComtradeDownloader")

# Load environment variables
load_dotenv()
COMTRADE_API_KEY = os.getenv("COMTRADE_API_KEY", "")

# Constants for UN Comtrade M49 / HS Codes
INDIA_M49_CODE = "356"       # India reporter code in UN Comtrade
WORLD_M49_CODE = "0"         # World / All partners
DEFAULT_START_YEAR = 2005
DEFAULT_END_YEAR = 2024

# Top HS 2-digit Commodity Chapters critical for Indian economy
TOP_HS2_COMMODITIES = [
    "27",  # Mineral fuels, mineral oils and products of their distillation (Oil/Gas)
    "71",  # Natural or cultured pearls, precious or semi-precious stones, coins (Gold/Gems)
    "85",  # Electrical machinery and equipment and parts thereof (Electronics)
    "84",  # Nuclear reactors, boilers, machinery and mechanical appliances
    "30",  # Pharmaceutical products
    "29",  # Organic chemicals
    "10",  # Cereals (Wheat/Rice)
    "72",  # Iron and steel
    "87",  # Vehicles other than railway or tramway rolling-stock
    "39",  # Plastics and articles thereof
]


class UNComtradeDownloader:
    """Downloader wrapper handling API limits, pagination, and Parquet storage."""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "data/raw/un_comtrade"):
        self.api_key = api_key or COMTRADE_API_KEY
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key or self.api_key == "your_comtrade_free_api_key_here":
            logger.warning("No valid COMTRADE_API_KEY found in .env. Using preview/free tier limits (max 500 records/call).")

    def download_year_data(self, year: int, cmd_codes: List[str] = TOP_HS2_COMMODITIES) -> pd.DataFrame:
        """
        Download trade flows for India for a specific year across specified HS chapters.
        To stay well within 500 calls/day and 100k records/call limits, we query by year.
        """
        logger.info(f"Fetching trade data for India (Year: {year})...")
        cmd_string = ",".join(cmd_codes)
        
        if HAS_COMTRADE_LIB and self.api_key and self.api_key != "your_comtrade_free_api_key_here":
            try:
                # Use official comtradeapicall library
                df = comtradeapicall.getFinalData(
                    subscription_key=self.api_key,
                    typeCode="C",          # Commodities
                    freqCode="A",          # Annual (use 'M' for monthly if needed)
                    clCode="HS",           # Harmonized System
                    period=str(year),
                    reporterCode=INDIA_M49_CODE,
                    cmdCode=cmd_string,
                    flowCode="M,X",        # Imports (M) and Exports (X)
                    partnerCode=WORLD_M49_CODE
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.error(f"comtradeapicall failed for {year}: {e}. Falling back to REST API.")

        # Fallback to direct REST requests (works with or without library)
        return self._fetch_via_rest(year, cmd_string)

    def _fetch_via_rest(self, year: int, cmd_string: str) -> pd.DataFrame:
        """Direct REST API fallback with proper rate limiting and error handling."""
        base_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
        params = {
            "reporterCode": INDIA_M49_CODE,
            "period": str(year),
            "cmdCode": cmd_string,
            "flowCode": "M,X",
            "partnerCode": WORLD_M49_CODE
        }
        headers = {}
        if self.api_key and self.api_key != "your_comtrade_free_api_key_here":
            headers["Ocp-Apim-Subscription-Key"] = self.api_key
        else:
            # Use public preview endpoint if no key
            base_url = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("data", [])
            if not records:
                logger.warning(f"No records returned for year {year}.")
                return pd.DataFrame()
                
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"REST API error for year {year}: {e}")
            return pd.DataFrame()

    def run_pipeline(self, start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR) -> str:
        """
        Executes full download loop across years, merges data, and saves to Parquet.
        Returns path to saved Parquet file.
        """
        all_dfs = []
        total_years = end_year - start_year + 1
        
        logger.info(f"Starting UN Comtrade download pipeline ({start_year}-{end_year})...")
        
        for idx, year in enumerate(range(start_year, end_year + 1), 1):
            logger.info(f"Progress: [{idx}/{total_years}] Processing year {year}...")
            df_year = self.download_year_data(year)
            
            if not df_year.empty:
                all_dfs.append(df_year)
                logger.info(f"Year {year}: Retrieved {len(df_year)} trade records.")
            else:
                logger.warning(f"Year {year}: Empty dataset.")
                
            # Rate limiting: Sleep 1.5 seconds between calls to respect UN servers
            time.sleep(1.5)
            
        if not all_dfs:
            logger.error("Pipeline failed: No data retrieved across all years.")
            return ""
            
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save to Parquet (10x smaller and faster than CSV)
        output_file = self.output_dir / f"india_trade_hs2_{start_year}_{end_year}.parquet"
        merged_df.to_parquet(output_file, index=False, engine="pyarrow")
        
        logger.info(f"SUCCESS: Saved {len(merged_df)} total trade records to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    downloader = UNComtradeDownloader()
    output_path = downloader.run_pipeline(start_year=2015, end_year=2023)  # Sample test range
    if output_path:
        print(f"\n[+] Pipeline completed successfully. Output: {output_path}")
