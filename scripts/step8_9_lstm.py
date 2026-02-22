import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Metrics
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

def create_sequences(data, target_col_idx, lookback=12, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
    return np.array(X), np.array(y)

def run_lstm(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Global Aggregation (matching baselines)
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index()
    global_df = global_df.sort_values('month')
    
    # Drop COVID years (2020-2021)
    global_df = global_df[~((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021))].copy()
    global_df.reset_index(drop=True, inplace=True)
    
    # Define features
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count']
    data = global_df[features].values
    target_idx = features.index('review_count')
    
    # 2. Train-Test Split (Time-based: 2017-2019,2022 vs 2023-2024)
    # We need to find the split index
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    # 3. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit only on train to prevent data leakage
    scaler.fit(data[:train_idx+1])
    scaled_data = scaler.transform(data)
    
    # 4. Create sequences (Sliding Window)
    lookback = 12
    horizon = 1
    X, y = create_sequences(scaled_data, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    # Split sequences into train and test
    # A sequence belongs to train if its target 'y' falls in the train period
    X_train = X[:train_idx - lookback + 1]
    y_train = y[:train_idx - lookback + 1]
    
    X_test = X[train_idx - lookback + 1:]
    y_test = y[train_idx - lookback + 1:]
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # 5. Build LSTM Model
    print("Building and Training LSTM Model...")
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(lookback, len(features)), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(horizon)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_split=0.2, # 20% of train data for validation
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Evaluation
    print("Evaluating Model on Test Set...")
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actuals
    # Create a dummy array matching the scaler's expected shape to inverse transform the target column
    dummy_input = np.zeros((len(predictions_scaled), len(features)))
    dummy_input[:, target_idx] = predictions_scaled.flatten()
    predictions_real = scaler.inverse_transform(dummy_input)[:, target_idx]
    
    dummy_input[:, target_idx] = y_test.flatten()
    y_test_real = scaler.inverse_transform(dummy_input)[:, target_idx]
    
    lstm_metrics = get_metrics(y_test_real, predictions_real)
    print("\n--- LSTM Metrics (Test Set 2023-2024) ---")
    print(lstm_metrics)
    
    # Compare with Baseline
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        baseline_metrics.loc['LSTM'] = lstm_metrics
        baseline_metrics.to_csv(baseline_path)
        print("\nUpdated Metrics File:")
        print(baseline_metrics)
    
    # 7. Plotting
    test_dates = global_df['month'].iloc[-len(y_test_real):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actuals (Global)', color='black', alpha=0.5)
    plt.plot(test_dates, y_test_real, label='Test Actuals', color='blue', marker='o')
    plt.plot(test_dates, predictions_real, label='LSTM Forecast', color='green', marker='x', linestyle='--')
    
    plt.title('Global Review Count Forecast: Actuals vs LSTM')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '6_lstm_forecasts.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved LSTM plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_lstm(input_file, output_directory)
