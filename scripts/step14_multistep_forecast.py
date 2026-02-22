import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

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

# Multi-Step Sequence Creator
def create_multistep_sequences(data, target_col_idx, lookback=12, horizon=3):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
    return np.array(X), np.array(y)

# --- MODELS ---
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=3, dropout=0.2):
        super(MultiStepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size) # output_size = horizon
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def run_multistep_forecast(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global_df = df.groupby('month').agg({
        'review_count': 'sum', 'avg_sentiment': 'mean',
        'rainfall_mm': 'mean', 'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # We will run experiments for Horizon = 3 and Horizon = 6 using LSTM
    horizons = [1, 3, 6]
    results = {}
    
    lookback = 12
    
    for h in horizons:
        print(f"\n--- Running Experiment for Horizon = {h} months ---")
        df_lstm = global_df.copy()
        data_lstm = df_lstm[features].values
        target_idx = features.index('review_count')
        
        scaler_lstm = MinMaxScaler(feature_range=(0, 1))
        scaler_lstm.fit(data_lstm[:train_idx+1])
        scaled_data_lstm = scaler_lstm.transform(data_lstm)
        
        X_l, y_l = create_multistep_sequences(scaled_data_lstm, target_col_idx=target_idx, lookback=lookback, horizon=h)
        
        # Test sequences should start entirely after train_idx
        # Train sequence needs to ensure y doesn't bleed into test
        X_train_l = torch.FloatTensor(X_l[:train_idx - lookback - h + 2]).to(device)
        y_train_l = torch.FloatTensor(y_l[:train_idx - lookback - h + 2]).to(device)
        X_test_l = torch.FloatTensor(X_l[train_idx - lookback - h + 2:]).to(device)
        y_test_l = y_l[train_idx - lookback - h + 2:]
        
        train_loader_l = DataLoader(TensorDataset(X_train_l, y_train_l), batch_size=8, shuffle=True)
        
        torch.manual_seed(42)
        model_lstm = MultiStepLSTM(input_size=len(features), output_size=h).to(device)
        optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience = 20
        early_stop = 0
        
        for epoch in range(200):
            model_lstm.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader_l:
                optimizer_lstm.zero_grad()
                loss = criterion(model_lstm(batch_X), batch_y)
                loss.backward()
                optimizer_lstm.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader_l)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = model_lstm.state_dict()
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= patience:
                    break
                    
        model_lstm.load_state_dict(best_model)
        model_lstm.eval()
        with torch.no_grad():
            preds_scaled_lstm = model_lstm(X_test_l).cpu().numpy()
            
        # Inverse transform the predictions array of shape (N, horizon)
        # We need to flatten or iterate
        preds_real = np.zeros_like(preds_scaled_lstm)
        actuals_real = np.zeros_like(y_test_l)
        
        for i in range(h):
            dummy_pred = np.zeros((len(preds_scaled_lstm), len(features)))
            dummy_pred[:, target_idx] = preds_scaled_lstm[:, i]
            preds_real[:, i] = scaler_lstm.inverse_transform(dummy_pred)[:, target_idx]
            
            dummy_act = np.zeros((len(y_test_l), len(features)))
            dummy_act[:, target_idx] = y_test_l[:, i]
            actuals_real[:, i] = scaler_lstm.inverse_transform(dummy_act)[:, target_idx]
            
        # Overall Metric across all steps
        flat_preds = preds_real.flatten()
        flat_actuals = actuals_real.flatten()
        
        metrics = get_metrics(flat_actuals, flat_preds)
        results[h] = metrics
        print(f"Metrics for Horizon={h}: {metrics}")
        
    print("\n--- Summary of Horizon Impact ---")
    summary_df = pd.DataFrame(results).T
    summary_df.index.name = 'Horizon (Months)'
    print(summary_df)
    
    # Save chart
    plt.figure(figsize=(10, 6))
    summary_df['MAPE'].plot(kind='bar', color='orange', alpha=0.8)
    plt.title('Predictive Error (MAPE) vs Forecasting Horizon')
    plt.ylabel('MAPE (%)')
    plt.xlabel('Horizon (Months Ahead)')
    plt.xticks(rotation=0)
    
    # Annotate bars
    for i, v in enumerate(summary_df['MAPE']):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    h_plot_path = os.path.join(output_dir, '12_horizon_impact.png')
    plt.savefig(h_plot_path)
    plt.close()
    print(f"Saved Horizon Impact plot to {h_plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    run_multistep_forecast(input_file, output_directory)
