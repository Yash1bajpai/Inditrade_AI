import pandas as pd
from src.backend.api.forecast import PARTNER_MAP, CMD_MAP

df = pd.read_parquet('data/processed/trade_features.parquet')
print('Columns:', df.columns.tolist())

# Check if there is flow data
flow_cols = [c for c in df.columns if 'flow' in c.lower()]
if flow_cols:
    for c in flow_cols:
        print(f'Unique values in {c}:', df[c].unique())
else:
    print('No flow column found.')

# Get top 2 commodities per country
grouped = df.groupby(['partnerCode', 'cmdCode'])['primaryValue'].sum().reset_index()
grouped['partnerName'] = grouped['partnerCode'].apply(lambda x: PARTNER_MAP.get(str(x).split('.')[0], str(x)))
grouped['cmdName'] = grouped['cmdCode'].apply(lambda x: CMD_MAP.get(str(x).split('.')[0].zfill(2), str(x)))

top_cmds = grouped.sort_values(['partnerName', 'primaryValue'], ascending=[True, False]).groupby('partnerName').head(3)

print('\nTop Commodities per Country:')
for name, group in top_cmds.groupby('partnerName'):
    cmds = ', '.join([f"{row['cmdName']} (${row['primaryValue']/1e9:.1f}B)" for _, row in group.iterrows()])
    print(f'{name}: {cmds}')
