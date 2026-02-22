import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_location_mapping(raw_dir):
    location_mapping = {}
    raw_files = [
        'absa_deepseek_results_merged_backup.csv',
        'absa_deepseek_results_backup.csv'
    ]
    for rf in raw_files:
        raw_path = os.path.join(raw_dir, rf)
        if os.path.exists(raw_path):
            try:
                raw_df = pd.read_csv(raw_path, usecols=['locationId', 'hotelName'], dtype=str)
                raw_df = raw_df.dropna(subset=['locationId', 'hotelName'])
                for _, row in raw_df.drop_duplicates(subset=['locationId']).iterrows():
                    loc_id = row['locationId']
                    if loc_id not in location_mapping:
                        name = str(row['hotelName']).split('-')[0].split(',')[0].strip()
                        location_mapping[loc_id] = name
            except Exception as e:
                print(f"Error reading {rf}: {e}")
    return location_mapping

def run_interpretability(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # Exclude Dacotour leakage
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Seasonality Analysis: Domestic vs International
    print("Generating Seasonality Analysis...")
    # Extract the month number (1-12)
    df['month_num'] = df['month'].dt.month
    
    # Because of COVID, let's exclude 2020-2021 to get pure seasonality patterns
    df_normal = df[~((df['month'].dt.year >= 2020) & (df['month'].dt.year <= 2021))].copy()
    
    # Group by month_num
    seasonality = df_normal.groupby('month_num').agg({
        'dom_count': 'sum',
        'intl_count': 'sum'
    }).reset_index()
    
    # Normalize to see the shape (percentage of total year)
    dom_total = seasonality['dom_count'].sum()
    intl_total = seasonality['intl_count'].sum()
    seasonality['dom_pct'] = (seasonality['dom_count'] / dom_total) * 100
    seasonality['intl_pct'] = (seasonality['intl_count'] / intl_total) * 100
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(seasonality['month_num']))
    
    plt.bar(x - width/2, seasonality['dom_pct'], width, label='Domestic Tourist Seasonality', color='royalblue')
    plt.bar(x + width/2, seasonality['intl_pct'], width, label='International Tourist Seasonality', color='crimson')
    
    plt.title('Seasonal Patterns: Domestic vs International Tourists in Da Nang')
    plt.xlabel('Month')
    plt.ylabel('% of Total Yearly Reviews')
    plt.xticks(x, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    season_plot_path = os.path.join(output_dir, '9_seasonality_dom_vs_intl.png')
    plt.savefig(season_plot_path)
    plt.close()
    print(f"Saved Seasonality Plot to {season_plot_path}")
    
    # 2. Case Studies (Deep dive into top 3 locations)
    print("Generating Case Studies for Top Locations...")
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), 'data', 'raw')
    location_mapping = get_location_mapping(raw_dir)
    
    # Identify top 3 locations by total reviews (excluding Dacotour)
    top_locs = df.groupby('locationId')['review_count'].sum().sort_values(ascending=False).head(3).index.tolist()
    
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, loc_id in enumerate(top_locs):
        loc_data = df[df['locationId'] == loc_id].sort_values('month')
        name = location_mapping.get(loc_id, f"Location {loc_id}")
        plt.plot(loc_data['month'], loc_data['review_count'], label=name, color=colors[i], linewidth=2)
        
    plt.title('Tourist Volume Trends for Top 3 Destinations in Da Nang')
    plt.xlabel('Time')
    plt.ylabel('Monthly Reviews')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    case_plot_path = os.path.join(output_dir, '10_top_locations_case_studies.png')
    plt.savefig(case_plot_path)
    plt.close()
    print(f"Saved Case Studies Plot to {case_plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_interpretability(input_file, output_directory)
