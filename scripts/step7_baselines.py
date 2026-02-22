import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_idx = y_true != 0
    if not np.any(non_zero_idx):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def run_baselines(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # --- ABLATION EXPERIMENT: Exclude Anomalous Locations ---
    excluded_locations = ['d6974493'] # Dacotour
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Evaluating baselines on Global aggregated data for simplicity...")
    # Aggregate globally
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index()
    
    global_df = global_df.sort_values('month')
    
    # Exclude COVID-19 period (2020 and 2021)
    global_df = global_df[~((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021))].copy()
    
    # Time-based split: Train (2017-2019, 2022), Test (2023-2024)
    train = global_df[global_df['month'].dt.year <= 2022].copy()
    test = global_df[global_df['month'].dt.year > 2022].copy()
    
    print(f"Train samples: {len(train)}, Test samples: {len(test)}")
    
    results = {}
    
    # 1. Naive Baseline (Last Value T-1)
    print("Running Naive Baseline...")
    # The true naive forecast is predicting the immediately preceding month's value
    # We create a combined series to easily shift by 1
    combined_counts = pd.concat([train['review_count'], test['review_count']])
    shifted_counts = combined_counts.shift(1)
    predictions_naive = shifted_counts.loc[test.index].values
    results['Naive'] = get_metrics(test['review_count'], predictions_naive)
    
    # 2. Seasonal Naive Baseline (Same month last year)
    print("Running Seasonal Naive Baseline...")
    last_train_val = train['review_count'].iloc[-1]
    predictions_snaive = []
    for test_date in test['month']:
        # Look back 12 months
        past_date = test_date - pd.DateOffset(months=12)
        match = global_df[global_df['month'] == past_date]
        if not match.empty:
            predictions_snaive.append(match['review_count'].values[0])
        else:
            predictions_snaive.append(last_train_val)
    results['Seasonal_Naive'] = get_metrics(test['review_count'], predictions_snaive)
    
    # 3. Prophet Model
    print("Running Prophet Model...")
    # Prepare data for Prophet (requires 'ds' and 'y')
    prophet_train = train.rename(columns={'month': 'ds', 'review_count': 'y'})
    prophet_test = test.rename(columns={'month': 'ds', 'review_count': 'y'})
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.add_country_holidays(country_name='VN')
    
    # Add regressors
    model.add_regressor('avg_sentiment')
    model.add_regressor('rainfall_mm')
    
    model.fit(prophet_train)
    
    # Predict on test
    forecast = model.predict(prophet_test[['ds', 'avg_sentiment', 'rainfall_mm']])
    predictions_prophet = forecast['yhat'].values
    
    results['Prophet'] = get_metrics(test['review_count'], predictions_prophet)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(train['month'], train['review_count'], label='Train Actuals', color='black')
    plt.plot(test['month'], test['review_count'], label='Test Actuals', color='blue', marker='o')
    plt.plot(test['month'], predictions_naive, label='Naive Forecast', linestyle='--')
    plt.plot(test['month'], predictions_snaive, label='Seasonal Naive Forecast', linestyle='--')
    plt.plot(test['month'], predictions_prophet, label='Prophet Forecast', linestyle='-', color='red', marker='x')
    
    # Fill between Prophet uncertainty intervals if we want (requires predicting on full range or joining)
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='Prophet Uncertainty')

    plt.title('Global Review Count Forecast: Baselines vs Prophet')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '5_baseline_forecasts.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save Metrics
    metrics_df = pd.DataFrame(results).T
    print("\n--- Baseline Metrics (Test Set 2023-2024) ---")
    print(metrics_df)
    
    metrics_path = os.path.join(output_dir, 'baseline_metrics.csv')
    metrics_df.to_csv(metrics_path)
    
    print(f"\nSaved plots to {plot_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_baselines(input_file, output_directory)
