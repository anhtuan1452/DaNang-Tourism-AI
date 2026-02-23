import pandas as pd
import numpy as np
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv'))
df['month'] = pd.to_datetime(df['month'])

# Same aggregation as the ensemble script
excluded_locations = ['d6974493']
df = df[~df['locationId'].isin(excluded_locations)].copy()
global_df = df.groupby('month').agg({'review_count':'sum','avg_sentiment':'mean','rainfall_mm':'mean','holiday_count':'sum'}).reset_index().sort_values('month')
global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
global_df = global_df[global_df['month'] < '2026-02-01'].copy()
global_df.reset_index(drop=True, inplace=True)

split = int(len(global_df) * 0.8) - 1
print(f"Total rows: {len(global_df)}")
print(f"Train ends at index {split}: {global_df.iloc[split]['month'].date()}")
print(f"Test starts at index {split+1}: {global_df.iloc[split+1]['month'].date()}")
print()

print("=== ACTUAL TEST DATA (what the model saw during evaluation) ===")
print(global_df.iloc[split+1:][['month','review_count','rainfall_mm','holiday_count']].to_string())
print()

print("=== SEASONAL BOOTSTRAP LOOKUP (what future months will use as aux features) ===")
start = global_df['month'].max() + pd.DateOffset(months=1)
for step in range(12):
    future_month = start + pd.DateOffset(months=step)
    same_month_last_year = future_month - pd.DateOffset(years=1)
    match = global_df[global_df['month'] == same_month_last_year]
    if len(match) > 0:
        row = match.iloc[0]
        print(f"Forecast {future_month.strftime('%Y-%m')} -> uses aux from {same_month_last_year.strftime('%Y-%m')}: "
              f"review={row['review_count']:.0f}  rainfall={row['rainfall_mm']:.1f}  holiday={row['holiday_count']:.0f}  sentiment={row['avg_sentiment']:.3f}")
    else:
        print(f"Forecast {future_month.strftime('%Y-%m')} -> NO MATCH — frozen (fallback to last known)")

print()
print("=== LAST 12 MONTHS (lookback window the model starts from) ===")
print(global_df.tail(12)[['month','review_count','rainfall_mm','holiday_count','avg_sentiment']].to_string())
