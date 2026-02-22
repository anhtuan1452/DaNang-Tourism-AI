import pandas as pd
import os

raw_file = 'e:\\Ky 1 nam 4\\NCKH\\Forecasting_Project\\data\\raw\\absa_deepseek_results_merged_backup.csv'
raw_file2 = 'e:\\Ky 1 nam 4\\NCKH\\Forecasting_Project\\data\\raw\\absa_deepseek_results_backup.csv'

print("Loading raw files...")
dfs = []
for f in [raw_file, raw_file2]:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f, usecols=['locationId', 'hotelName'], dtype=str))

df = pd.concat(dfs, ignore_index=True)
df['hotelName'] = df['hotelName'].str.split('Unclaimed').str[0].str.split('Someone from this business manages').str[0].str.split('If you own this business').str[0].str.strip()

# Find unique mappings
mappings = df.groupby('hotelName')['locationId'].unique().reset_index()

output_file = 'e:\\Ky 1 nam 4\\NCKH\\Forecasting_Project\\scripts\\duplicates_report.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("--- Locations with Multiple IDs ---\n")
    duplicates = mappings[mappings['locationId'].apply(len) > 1]
    if len(duplicates) > 0:
        for idx, row in duplicates.iterrows():
            f.write(f"Name: {row['hotelName']} -> IDs: {row['locationId']}\n")
    else:
        f.write("No exact name matches with multiple IDs found.\n")

    f.write("\n--- Checking 'Dacotour' or similar (case-insensitive) ---\n")
    daco_mask = df['hotelName'].str.lower().str.contains('daco', na=False)
    if daco_mask.any():
        daco_df = df[daco_mask].drop_duplicates(subset=['locationId', 'hotelName'])
        f.write(daco_df.to_string() + "\n")
    else:
        f.write("No location containing 'daco' found.\n")
        
    f.write("\n--- Review Count per location ID (Top 10) ---\n")
    f.write(df['locationId'].value_counts().head(10).to_string() + "\n")

    f.write("\n--- Review Count per location Name (Top 10) ---\n")
    f.write(df['hotelName'].value_counts().head(10).to_string() + "\n")

print(f"Report saved to {output_file}")
