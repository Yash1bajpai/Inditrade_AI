"""
Kaggle kernel: rebuild IndiTrade AI trade features with the 2025 data refresh.

Runs the Phase 2 feature engineering pipeline (src/feature_engineering/
trade_features.py) on Kaggle per the heavy-compute rule. Input: the refreshed
raw Comtrade parquet (2015-2025) + daily macro CSVs from the GitHub data-v1
release. Output: trade_features_2015_2025.parquet + verification proof,
zipped for download.
"""
import os
import subprocess
import sys

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"


def run(cmd, **kwargs):
    print(f"\n$ {cmd}\n{'-' * 70}", flush=True)
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"!! command failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    return result


run("pip install -q 'pandas>=2.0' 'numpy<2.0' pyarrow 2>&1 | tail -2")

if not os.path.exists(REPO):
    run(f"git clone --depth 1 https://github.com/Yash1bajpai/Inditrade_AI {REPO}")
os.chdir(REPO)
os.makedirs("data/raw/un_comtrade", exist_ok=True)
os.makedirs("data/raw/forex_macro", exist_ok=True)

BASE = "https://github.com/Yash1bajpai/Inditrade_AI/releases/download/data-v1"
run(f"curl -fsL --retry 3 {BASE}/india_trade_hs2_2015_2024.parquet -o data/raw/un_comtrade/india_trade_hs2_2015_2024.parquet")
for f in ["usdinr", "eurinr", "gbpinr", "jpyinr", "cnyinr",
          "brent_crude", "gold_futures", "nifty_50", "sensex"]:
    run(f"curl -fsL --retry 3 {BASE}/{f}.csv -o data/raw/forex_macro/{f}.csv")

# Input proof before building
run("python -c \"import pandas as pd; df=pd.read_parquet('data/raw/un_comtrade/india_trade_hs2_2015_2024.parquet'); print('RAW INPUT: rows', len(df), '| years', sorted(df['period'].astype(int).unique().tolist()))\"")

# Phase 2 feature engineering (aggregation, macro merge, lags, policy flag incl. 2025)
run("python src/feature_engineering/trade_features.py", stdout=sys.stdout, stderr=sys.stderr)

# Output verification proof
run("python - <<'EOF'\n"
    "import pandas as pd\n"
    "df = pd.read_parquet('data/processed/trade_features.parquet')\n"
    "print('FEATURES: shape', df.shape)\n"
    "print('years:', sorted(df['period'].astype(int).unique().tolist()))\n"
    "print('2025 rows:', (df['period'].astype(int)==2025).sum())\n"
    "print('policy years:', sorted(df.loc[df['policy_event_flag']==1,'period'].astype(int).unique().tolist()))\n"
    "d=df[df['period'].astype(int)==2025]\n"
    "print('2025 lag_1y nulls:', d['primaryValue_lag_1y'].isna().sum(), '/', len(d))\n"
    "print('2025 sample (partner, cmd, flow, value, lag1):')\n"
    "print(d[['partnerDesc','cmdCode','flowCode','primaryValue','primaryValue_lag_1y']].head(5).to_string(index=False))\n"
    "EOF", stdout=sys.stdout, stderr=sys.stderr)

run("cp data/processed/trade_features.parquet trade_features_2015_2025.parquet "
    "&& zip -q trade_features_2025_refresh.zip trade_features_2015_2025.parquet "
    "&& ls -la trade_features_2025_refresh.zip")

print("\n=== KERNEL COMPLETE: trade_features_2025_refresh.zip ready ===", flush=True)
