import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# --- MIXED LSTM MODEL ---
class MixedSTLLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        super(MixedSTLLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def run_mixed_stl_lstm(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Global Aggregation 
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    global_df.set_index('month', inplace=True)
    
    # ---------------------------------------------------------
    # SCIENTIFIC DECOMPOSITION (STL): Trend + Seasonality
    # ---------------------------------------------------------
    print("Performing STL Decomposition for Feature Engineering...")
    stl = STL(global_df['review_count'], period=12, robust=True)
    res = stl.fit()
    
    global_df['trend'] = res.trend
    global_df['seasonal'] = res.seasonal
    
    global_df['is_covid'] = ((global_df.index.year >= 2020) & (global_df.index.year <= 2021)).astype(int)
    
    # Features for the Mixed Model
    features = ['review_count', 'trend', 'seasonal', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('review_count')
    
    # Train-Test Split Date
    split_date = pd.to_datetime('2022-12-31')
    train_idx = np.where(global_df.index <= split_date)[0][-1]
    
    data = global_df[features].values
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data[:train_idx+1])
    scaled_data = scaler.transform(data)
    
    lookback = 12
    horizon = 1
    X, y = create_sequences(scaled_data, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    X_train_np = X[:train_idx - lookback + 1]
    y_train_np = y[:train_idx - lookback + 1]
    X_test_np = X[train_idx - lookback + 1:]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print("Training Mixed STL-LSTM Model...")
    torch.manual_seed(42)
    model = MixedSTLLSTM(input_size=len(features), hidden_size=64, num_layers=1, dropout=0.0).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for float_epoch in range(80):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()
        
    dummy_input = np.zeros((len(preds_scaled), len(features)))
    dummy_input[:, target_idx] = preds_scaled.flatten()
    final_preds = scaler.inverse_transform(dummy_input)[:, target_idx]
    
    # Clip to ensure no negative reviews
    final_preds = np.maximum(0, final_preds)
    
    test_dates = global_df.index[train_idx+1:]
    actuals = global_df['review_count'].values[train_idx+1:]
    
    metrics = get_metrics(actuals, final_preds)
    print(f"\n[SUCCESS] END-TO-END MIXED STL-LSTM METRICS: {metrics}")
    
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        baseline_metrics.loc['End-to-End Mixed STL-LSTM'] = metrics
        baseline_metrics.to_csv(baseline_path)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(global_df.index, global_df['review_count'], label='Actual Data (Full Ecosystem)', color='black', alpha=0.5)
    
    plt.plot(test_dates, actuals, label='Test Actuals (2023-2024)', color='blue', marker='o')
    plt.plot(test_dates, final_preds, label='Mixed STL-LSTM Forecast', color='purple', marker='X', markersize=8)
    
    plt.title('Mixed STL-LSTM Forecast: End-to-End Learning from Decomposed Signals')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '23_mixed_stl_lstm_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved Mixed STL-LSTM plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_mixed_stl_lstm(input_file, output_directory)
