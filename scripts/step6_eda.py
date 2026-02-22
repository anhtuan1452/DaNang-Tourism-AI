import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def perform_eda(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    ensure_dir(output_dir)
    print("Output directory ready.")
    
    # 1. Global Trend & Seasonality
    print("Generating Global Trend & Seasonality Plot...")
    global_monthly = df.groupby('month')['review_count'].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=global_monthly, x='month', y='review_count', marker='o')
    plt.title('Global Monthly Review Count Trend (Da Nang Tourism)')
    plt.xlabel('Month')
    plt.ylabel('Total Reviews')
    plt.grid(True)
    plt.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2021-12-31'), color='red', alpha=0.1, label='COVID-19 Impact')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_global_trend.png'))
    plt.close()
    
    # 2. Correlation Matrix
    print("Generating Correlation Matrix...")
    # Select numeric columns relevant for modeling
    corr_cols = [
        'review_count', 'rating', 'avg_sentiment', 'dom_count', 'intl_count',
        'temp_mean', 'rainfall_mm', 'holiday_count'
    ]
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation between Core Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_correlation_matrix.png'))
    plt.close()
    
    # 3. Top Locations Trend Analysis
    print("Generating Top Locations Trend...")
    
    # Attempt to load location names from the original dataset included in data/raw
    location_mapping = {}
    try:
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), 'raw')
        raw_files = [
            os.path.join(raw_dir, 'absa_deepseek_results_merged_backup.csv'),
            os.path.join(raw_dir, 'absa_deepseek_results_backup.csv')
        ]
        
        raw_dfs = []
        for f in raw_files:
            if os.path.exists(f):
                raw_dfs.append(pd.read_csv(f, usecols=['locationId', 'hotelName'], dtype=str))
                
        if raw_dfs:
            raw_df = pd.concat(raw_dfs, ignore_index=True)
            # Clean up TripAdvisor boilerplate text from names
            raw_df['hotelName'] = raw_df['hotelName'].str.split('Unclaimed').str[0].str.split('Someone from this business manages').str[0].str.split('If you own this business').str[0].str.strip()
            location_mapping = raw_df.drop_duplicates(subset=['locationId']).set_index('locationId')['hotelName'].to_dict()
    except Exception as e:
        print(f"Could not load location names: {e}")
        
    top_locations = df.groupby('locationId')['review_count'].sum().nlargest(5).index
    df_top = df[df['locationId'].isin(top_locations)].copy()
    
    # Map IDs to Names
    if location_mapping:
        df_top['locationName'] = df_top['locationId'].map(lambda x: location_mapping.get(x, x))
        hue_col = 'locationName'
    else:
        hue_col = 'locationId'
        
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_top, x='month', y='review_count', hue=hue_col, marker='o')
    plt.title('Review Count Trend for Top 5 Locations')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_top_locations_trend.png'))
    plt.close()
    
    # 4. Stationarity Check (ADF Test) on Global Review Count
    print("\n--- Stationarity Check (ADF Test) ---")
    # ADF Null hypothesis: Time series has a unit root (is non-stationary)
    timeseries = global_monthly['review_count'].values
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
        
    if result[1] <= 0.05:
        print("Conclusion: Reject the null hypothesis (Series is Stationary).")
    else:
        print("Conclusion: Fail to reject the null hypothesis (Series is Non-Stationary). Differencing may be required for traditional models like ARIMA/SARIMA.")
    
    with open(os.path.join(output_dir, '4_adf_test_results.txt'), 'w') as f:
        f.write("--- Stationarity Check (ADF Test) on Global Review Count ---\n")
        f.write(f"ADF Statistic: {result[0]:.4f}\n")
        f.write(f"p-value: {result[1]:.4f}\n")
        for key, value in result[4].items():
            f.write(f"{key}: {value:.4f}\n")
        
        if result[1] <= 0.05:
            f.write("\nConclusion: Series is Stationary (p <= 0.05).\n")
        else:
            f.write("\nConclusion: Series is Non-Stationary (p > 0.05). Differencing recommended for ARIMA.\n")

    print(f"\nEDA completed. Plots and logs saved to: {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    perform_eda(input_file, output_directory)
