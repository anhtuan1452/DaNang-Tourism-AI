import pandas as pd
import numpy as np
import os

def preprocess_absa_data(input_paths, output_path):
    print(f"Loading data from: {input_paths}")
    
    # Read and combine multiple CSVs
    dfs = []
    for path in input_paths:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: File {path} not found.")
            
    if not dfs:
        print("No valid input files found.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    
    # Define aspect columns based on the dataset
    aspects = ['service', 'staff', 'quality', 'facility', 'cleanliness', 'price', 'ambiance', 'food']
    
    # 1. Map sentiments to numerical values (-1, 0, 1)
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    
    print("Converting sentiment labels to numerical scores and combining with confidence...")
    for aspect in aspects:
        sentiment_col = f'deepseek_aspect_{aspect}_sentiment'
        confidence_col = f'deepseek_aspect_{aspect}_confidence'
        score_col = f'{aspect}_score'
        
        if sentiment_col in df.columns and confidence_col in df.columns:
            # Map sentiment, fill NaN with 0 (neutral/no mention)
            mapped_sentiment = df[sentiment_col].map(sentiment_mapping).fillna(0)
            # Fill NaN confidence with 0
            conf = df[confidence_col].fillna(0)
            
            # Weighted score: Sentiment * Confidence 
            # Example: positive (1) * 0.9 confidence = 0.9 score
            df[score_col] = mapped_sentiment * conf
            
    # 2. Date conversion
    # Assume createdDate is in format dd/mm/yyyy based on sample data
    df['createdDate'] = pd.to_datetime(df['createdDate'], format='%d/%m/%Y', errors='coerce')
    
    # Drop rows without a valid date
    df = df.dropna(subset=['createdDate'])
    
    # Filter dates from 01/01/2017 onwards
    start_date = pd.to_datetime('2017-01-01')
    df = df[df['createdDate'] >= start_date]
    
    # Sort by date
    df = df.sort_values('createdDate')
    
    # 3. Aggregate by Date (Time-Series format)
    print("Aggregating scores by date...")
    
    # We want to calculate the daily average score for each aspect
    agg_dict = {f'{aspect}_score': 'mean' for aspect in aspects if f'{aspect}_score' in df.columns}
    
    # We can also add some other useful features:
    # - Daily review count
    # - Average daily rating
    agg_dict.update({
        'id': 'count',       # Total reviews per day
        'rating': 'mean'     # Average rating per day
    })
    
    daily_df = df.groupby('createdDate').agg(agg_dict).reset_index()
    
    # Rename columns for clarity
    daily_df = daily_df.rename(columns={'id': 'review_count', 'rating': 'avg_rating'})
    
    # Handle dates with 0 reviews by reindexing the date range (filling missing dates)
    # This is crucial for Time Series forecasting models!
    full_date_range = pd.date_range(start=daily_df['createdDate'].min(), end=daily_df['createdDate'].max(), freq='D')
    daily_df = daily_df.set_index('createdDate').reindex(full_date_range).reset_index()
    daily_df = daily_df.rename(columns={'index': 'date'})
    
    # Fill missing values for the newly added dates
    # For review_count, missing means 0
    daily_df['review_count'] = daily_df['review_count'].fillna(0)
    # For scores and rating, we can use forward-fill or backward-fill, or fill with 0
    # Forward-fill is generally safe for time-series features like sentiment momentum
    daily_df = daily_df.ffill().fillna(0) # Forward fill, then fill remaining NaNs at the beginning with 0
    
    # Save the processed data
    print(f"Saving processed time-series data to: {output_path}")
    daily_df.to_csv(output_path, index=False)
    
    print("\n--- Summary of Processed Data ---")
    print(daily_df.head())
    print(f"\nTotal rows (days): {len(daily_df)}")
    print("Feature columns available for Forecasting:")
    print(daily_df.columns.tolist())

if __name__ == "__main__":
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_files = [
        os.path.join(project_dir, 'data', 'raw', 'absa_deepseek_results_merged_backup.csv'),
        os.path.join(project_dir, 'data', 'raw', 'absa_deepseek_results_backup.csv')
    ]
    output_file = os.path.join(project_dir, 'data', 'processed', 'daily_absa_features.csv')
    
    preprocess_absa_data(input_files, output_file)
