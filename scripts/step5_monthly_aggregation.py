import pandas as pd
import numpy as np
import os

def generate_monthly_features(absa_paths, output_path):
    print("Loading ABSA sentiment datasets...")
    
    # 1. Read and combine multiple CSVs
    dfs = []
    for path in absa_paths:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: File {path} not found.")
            
    if not dfs:
        print("Error: No valid input files found.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    
    # 2. Extract Location ID and standardize date
    df['createdDate'] = pd.to_datetime(df['createdDate'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['createdDate', 'locationId'])
    
    # Filter for > 2017
    df = df[df['createdDate'] >= pd.to_datetime('2017-01-01')]
    
    # Extract Month/YYYY format for grouping
    df['month'] = df['createdDate'].dt.to_period('M')
    
    # 3. Categorize Domestic vs International
    # Assuming 'language' column: 'vi' is domestic, else international
    if 'language' in df.columns:
        df['is_domestic'] = (df['language'] == 'vi').astype(int)
        df['is_intl'] = (df['language'] != 'vi').astype(int)
    else:
        # Fallback if language is missing
        df['is_domestic'] = 0
        df['is_intl'] = 1
        
    # 4. Process Sentiment Scores
    aspects = ['service', 'staff', 'quality', 'facility', 'cleanliness', 'price', 'ambiance', 'food']
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    score_cols = []
    
    for aspect in aspects:
        sentiment_col = f'deepseek_aspect_{aspect}_sentiment'
        confidence_col = f'deepseek_aspect_{aspect}_confidence'
        score_col = f'{aspect}_score'
        
        if sentiment_col in df.columns and confidence_col in df.columns:
            mapped_sentiment = df[sentiment_col].map(sentiment_mapping).fillna(0)
            conf = df[confidence_col].fillna(0)
            df[score_col] = mapped_sentiment * conf
            score_cols.append(score_col)
            
    # Calculate global average sentiment per review
    if score_cols:
        df['avg_sentiment'] = df[score_cols].mean(axis=1)
    else:
        df['avg_sentiment'] = 0
        
    # Domestic vs International Sentiment
    df['dom_sentiment'] = df['avg_sentiment'] * df['is_domestic']
    df['intl_sentiment'] = df['avg_sentiment'] * df['is_intl']
    
    # 5. Aggregate by Location and Month
    print("Aggregating data by 'locationId' and 'month'...")
    
    agg_dict = {
        'id': 'count',               # review_count
        'rating': 'mean',            # avg_rating
        'avg_sentiment': 'mean',
        'is_domestic': 'sum',        # dom_count
        'is_intl': 'sum',            # intl_count
    }
    
    # Add aspect means
    for col in score_cols:
        agg_dict[col] = 'mean'
        
    monthly_df = df.groupby(['locationId', 'month']).agg(agg_dict).reset_index()
    monthly_df = monthly_df.rename(columns={'id': 'review_count', 'is_domestic': 'dom_count', 'is_intl': 'intl_count'})
    
    # Calculate dom/intl sentiment averages
    # We group again specifically for the subset
    dom_sent = df[df['is_domestic'] == 1].groupby(['locationId', 'month'])['avg_sentiment'].mean().reset_index()
    dom_sent = dom_sent.rename(columns={'avg_sentiment': 'dom_sentiment'})
    
    intl_sent = df[df['is_intl'] == 1].groupby(['locationId', 'month'])['avg_sentiment'].mean().reset_index()
    intl_sent = intl_sent.rename(columns={'avg_sentiment': 'intl_sentiment'})
    
    monthly_df = pd.merge(monthly_df, dom_sent, on=['locationId', 'month'], how='left')
    monthly_df = pd.merge(monthly_df, intl_sent, on=['locationId', 'month'], how='left')
    
    # Fill NaN sent scores with 0
    monthly_df['dom_sentiment'] = monthly_df['dom_sentiment'].fillna(0)
    monthly_df['intl_sentiment'] = monthly_df['intl_sentiment'].fillna(0)
    
    # 6. Join Weather & Calendar Data
    print("Joining Weather data...")
    weather_df = pd.read_csv(os.path.join(project_dir, '..', 'weather_danang_monthly.csv'))
    weather_df['month'] = pd.to_datetime(weather_df['month']).dt.to_period('M')
    
    # Drop empty columns from weather
    weather_df = weather_df.dropna(axis=1, how='all')
    
    # Merge with monthly_df
    monthly_df = pd.merge(monthly_df, weather_df, on='month', how='left')
    
    print("Joining Holiday data...")
    holiday_df = pd.read_csv(os.path.join(project_dir, '..', 'Holiday', 'all_holidays_2010_2026.csv'))
    holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%d/%m/%Y', errors='coerce')
    holiday_df = holiday_df.dropna(subset=['date'])
    # Count number of holidays per month
    holiday_df['month'] = holiday_df['date'].dt.to_period('M')
    monthly_holidays = holiday_df.groupby('month').size().reset_index(name='holiday_count')
    
    monthly_df = pd.merge(monthly_df, monthly_holidays, on='month', how='left')
    monthly_df['holiday_count'] = monthly_df['holiday_count'].fillna(0)
    
    # Fill remaining weather NaNs with ffill/bfill to maintain structure
    weather_cols = [c for c in weather_df.columns if c != 'month']
    monthly_df[weather_cols] = monthly_df[weather_cols].interpolate(method='linear').ffill().bfill()
    
    # 7. (Optional) Smoothing by Moving Average (k=3) for Target Variables
    print("Applying 3-month moving average smoothing for noisy series...")
    # Sort by location and time to ensure sequential rolling
    monthly_df = monthly_df.sort_values(['locationId', 'month']).reset_index(drop=True)
    
    for loc in monthly_df['locationId'].unique():
        loc_mask = monthly_df['locationId'] == loc
        
        # Smooth review_count and average sentiment
        # min_periods=1 ensures that the first 2 months are still calculated
        monthly_df.loc[loc_mask, 'review_count_smoothed'] = monthly_df.loc[loc_mask, 'review_count'].rolling(window=3, min_periods=1).mean()
        monthly_df.loc[loc_mask, 'avg_sentiment_smoothed'] = monthly_df.loc[loc_mask, 'avg_sentiment'].rolling(window=3, min_periods=1).mean()
    
    # Convert 'month' Period back to string or timestamp for saving
    monthly_df['month'] = monthly_df['month'].dt.to_timestamp()
    
    print(f"Saving finalized monthly features to: {output_path}")
    monthly_df.to_csv(output_path, index=False)
    
    print("\n--- Summary of Monthly Data ---")
    print(monthly_df.head(10))
    print(f"\nTotal rows (location-months): {len(monthly_df)}")
    print(f"Total unique locations: {monthly_df['locationId'].nunique()}")
    print("Final List of Features:")
    print(list(monthly_df.columns))
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    absa_files = [
        os.path.join(project_dir, 'data', 'raw', 'absa_deepseek_results_merged_backup.csv'),
        os.path.join(project_dir, 'data', 'raw', 'absa_deepseek_results_backup.csv')
    ]
    output_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    
    generate_monthly_features(absa_files, output_file)
