import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# --- METRICS ---
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
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
    print(f"Loading data from {input_path} for Baseline Comparison...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # 1. Exclude anomaly point
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Global Aggregation
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    # Filter out the incomplete month of February 2026
    global_df = global_df[global_df['month'] < '2026-02-01'].copy()
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_col = 'review_count'
    
    # Custom Dynamic Split: Standard 80/20 train/test split.
    train_idx = int(len(global_df) * 0.8) - 1
    
    train_df = global_df.iloc[:train_idx+1].copy()
    test_df = global_df.iloc[train_idx+1:].copy()
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Exogenous features for ML models
    X_train = train_df[['avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']].values
    X_test = test_df[['avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']].values
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    predictions = {'Actuals': y_test}
    
    # --- 1. Naive (Seasonal) ---
    print("\n>> Training Model: Seasonal Naive")
    # T=12 months seasonality
    naive_preds = train_df[target_col].iloc[-12:].values
    # If test set is > 12 months, we repeat
    reps = int(np.ceil(len(y_test) / 12))
    naive_preds = np.tile(naive_preds, reps)[:len(y_test)]
    predictions['Seasonal Naive'] = naive_preds
    results['Seasonal Naive'] = get_metrics(y_test, naive_preds)
    
    # --- 2. ARIMA ---
    print("\n>> Training Model: ARIMA")
    try:
        model_arima = ARIMA(y_train, exog=X_train_scaled, order=(5, 1, 0))
        model_arima_fit = model_arima.fit()
        arima_preds = model_arima_fit.forecast(steps=len(y_test), exog=X_test_scaled)
        predictions['ARIMA'] = arima_preds
        results['ARIMA'] = get_metrics(y_test, arima_preds)
    except Exception as e:
        print(f"ARIMA failed: {e}")
        
    # --- 3. Prophet ---
    print("\n>> Training Model: Prophet")
    df_prophet = train_df[['month', target_col, 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']].copy()
    df_prophet = df_prophet.rename(columns={'month': 'ds', target_col: 'y'})
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.add_regressor('avg_sentiment')
    m.add_regressor('rainfall_mm')
    m.add_regressor('holiday_count')
    m.add_regressor('is_covid')
    m.fit(df_prophet)
    
    df_prophet_test = test_df[['month', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']].copy()
    df_prophet_test = df_prophet_test.rename(columns={'month': 'ds'})
    prophet_forecast = m.predict(df_prophet_test)
    prophet_preds = prophet_forecast['yhat'].values
    predictions['Prophet'] = prophet_preds
    results['Prophet'] = get_metrics(y_test, prophet_preds)
    
    # --- 4. Random Forest ---
    print("\n>> Training Model: Random Forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    predictions['Random Forest'] = rf_preds
    results['Random Forest'] = get_metrics(y_test, rf_preds)
    
    # --- 5. Support Vector Regression (SVR) ---
    print("\n>> Training Model: SVR")
    svr = SVR(kernel='rbf', C=1000, gamma=0.1)
    svr.fit(X_train_scaled, y_train)
    svr_preds = svr.predict(X_test_scaled)
    predictions['SVR'] = svr_preds
    results['SVR'] = get_metrics(y_test, svr_preds)
    
    # --- Advanced Ensemble Load ---
    print("\n>> Fetching Advanced Ensemble Results (Step 13)")
    ensemble_csv = os.path.join(output_dir, 'ensemble_test_predictions.csv')
    if os.path.exists(ensemble_csv):
        ens_df = pd.read_csv(ensemble_csv)
        ens_preds = ens_df['ensemble_pred'].values
        # Double check alignment
        if len(ens_preds) == len(y_test):
            predictions['Proposed Advanced Ensemble'] = ens_preds
            results['Proposed Advanced Ensemble'] = get_metrics(y_test, ens_preds)
        else:
            print("Warning: Length mismatch in ensemble predictions. Skipping ensemble plotting.")
    else:
        print("Warning: ensemble_test_predictions.csv not found. Please run step13_advanced_ensemble.py first.")

    # --- Save Metrics ---
    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df[['MAE', 'RMSE', 'MAPE']]
    metrics_df = metrics_df.sort_values(by='MAPE')
    
    csv_path = os.path.join(output_dir, 'baseline_comparison_metrics.csv')
    metrics_df.to_csv(csv_path)
    print("\n--- COMPARISON METRICS ---")
    print(metrics_df)
    print(f"\nSaved metrics to {csv_path}")
    
    # --- Plotting ---
    test_dates = test_df['month'].values
    plt.figure(figsize=(16, 8))
    
    plt.plot(train_df['month'], train_df[target_col], label='Training Actuals', color='black', alpha=0.3)
    plt.plot(test_dates, y_test, label='Test Actuals', color='black', linewidth=2, marker='o')
    
    colors = ['gray', 'orange', 'green', 'purple', 'brown']
    models_to_plot = ['Seasonal Naive', 'ARIMA', 'Prophet', 'Random Forest', 'SVR']
    
    for i, model in enumerate(models_to_plot):
        if model in predictions:
            plt.plot(test_dates, predictions[model], label=f'{model} (MAPE: {results[model]["MAPE"]:.1f}%)', color=colors[i], linestyle='--', alpha=0.7)
            
    if 'Proposed Advanced Ensemble' in predictions:
        ens_mape = results['Proposed Advanced Ensemble']['MAPE']
        plt.plot(test_dates, predictions['Proposed Advanced Ensemble'], 
                 label=f'Proposed Advanced Ensemble (MAPE: {ens_mape:.1f}%)', 
                 color='red', linewidth=3, marker='*', markersize=10)
    
    plt.title('Performance Comparison: Traditional & ML Baselines vs Proposed Advanced Ensemble')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'step26_baseline_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_baselines(input_file, output_directory)
