import pandas as pd
import os

def merge_datasets(absa_path, weather_path, holiday_path, output_path):
    print("Loading datasets...")
    # Load ABSA daily features
    df_absa = pd.read_csv(absa_path)
    df_absa['date'] = pd.to_datetime(df_absa['date'])
    
    # Load Weather data
    df_weather = pd.read_csv(weather_path)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    # Load Holiday data
    df_holiday = pd.read_csv(holiday_path)
    # Holiday dates are in DD/MM/YYYY format based on preview
    df_holiday['date'] = pd.to_datetime(df_holiday['date'], format='%d/%m/%Y', errors='coerce')
    
    # Remove duplicates from holidays if there are multiple events on the same day
    # We will just keep the first event name, but we mainly care about `is_holiday`
    df_holiday = df_holiday.drop_duplicates(subset=['date'], keep='first')
    
    # Merge datasets
    print("Merging datasets on 'date'...")
    # Base is df_absa because it has the continuous date range we created
    df_merged = pd.merge(df_absa, df_weather, on='date', how='left')
    df_merged = pd.merge(df_merged, df_holiday, on='date', how='left')
    
    # Feature Engineering for Holiday
    print("Processing holiday and weather features...")
    df_merged['is_holiday'] = df_merged['event'].notna().astype(int)
    
    # Fill missing values for weather
    # For rainfall and rainy_day, missing means 0
    weather_fill_0 = ['rainfall_mm', 'rainy_day']
    for col in weather_fill_0:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
    
    # For temperatures, interpolate or forward fill
    weather_fill_interp = ['temp_mean', 'temp_min', 'temp_max', 'humidity', 'sunshine_hours']
    for col in weather_fill_interp:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].interpolate(method='linear').ffill().bfill()
    
    # Fill remaining text columns
    df_merged['event'] = df_merged['event'].fillna('None')
    
    # Drop columns that are entirely missing (e.g., humidity, sunshine_hours)
    cols_before = df_merged.columns.tolist()
    df_merged = df_merged.dropna(axis=1, how='all')
    dropped_cols = set(cols_before) - set(df_merged.columns)
    if dropped_cols:
        print(f"Dropped completely empty columns: {dropped_cols}")
    
    print(f"Saving finalized dataset to: {output_path}")
    df_merged.to_csv(output_path, index=False)
    
    print("\n--- Summary of Final Merged Data ---")
    print(df_merged.head())
    print(f"\nTotal rows (days): {len(df_merged)}")
    print("Columns available for Forecasting:")
    print(list(df_merged.columns))
    
    # Check for missing values
    missing_counts = df_merged.isnull().sum()
    if missing_counts.sum() > 0:
        print("\nWarning: Missing values still exist in the following columns:")
        print(missing_counts[missing_counts > 0])
    else:
        print("\nAll missing values handled successfully. Dataset is clean!")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    root_dir = os.path.dirname(project_dir) # e:\Ky 1 nam 4\NCKH
    
    absa_file = os.path.join(project_dir, 'data', 'processed', 'daily_absa_features.csv')
    weather_file = os.path.join(root_dir, 'weather_danang_daily.csv')
    holiday_file = os.path.join(root_dir, 'Holiday', 'all_holidays_2010_2026.csv')
    
    output_file = os.path.join(project_dir, 'data', 'processed', 'final_multivariate_dataset.csv')
    
    merge_datasets(absa_file, weather_file, holiday_file, output_file)
