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

def create_sequences(data, target_col_idx, lookback=12, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
    return np.array(X), np.array(y)

# --- MODELS ---
class TourismLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        super(TourismLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, horizon=1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)
        
    def forward(self, src):
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = output[:, -1, :] 
        prediction = self.fc(output)
        return prediction

def run_deep_ensemble(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # Ablation
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
    }).reset_index()
    global_df = global_df.sort_values('month')
    
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    # --- 1. Train LSTM (Raw Targets) ---
    print("\n--- 1. Training LSTM ---")
    df_lstm = global_df.copy()
    data_lstm = df_lstm[features].values
    target_idx = features.index('review_count')
    
    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    scaler_lstm.fit(data_lstm[:train_idx+1])
    scaled_data_lstm = scaler_lstm.transform(data_lstm)
    
    lookback = 12
    horizon = 1
    X_l, y_l = create_sequences(scaled_data_lstm, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    X_train_l = torch.FloatTensor(X_l[:train_idx - lookback + 1]).to(device)
    y_train_l = torch.FloatTensor(y_l[:train_idx - lookback + 1]).to(device)
    X_test_l = torch.FloatTensor(X_l[train_idx - lookback + 1:]).to(device)
    y_test_l = y_l[train_idx - lookback + 1:] # numpy
    
    train_dataset_l = TensorDataset(X_train_l, y_train_l)
    train_loader_l = DataLoader(train_dataset_l, batch_size=8, shuffle=True)
    
    torch.manual_seed(42)
    model_lstm = TourismLSTM(input_size=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.005)
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 20
    model_lstm.train()
    
    for epoch in range(200):
        epoch_loss = 0
        for batch_X, batch_y in train_loader_l:
            optimizer_lstm.zero_grad()
            outputs = model_lstm(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_lstm.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader_l)
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            best_model_state_lstm = model_lstm.state_dict()
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            break
            
    model_lstm.load_state_dict(best_model_state_lstm)
    model_lstm.eval()
    with torch.no_grad():
        preds_scaled_lstm = model_lstm(X_test_l).cpu().numpy()
        
    dummy_input_lstm = np.zeros((len(preds_scaled_lstm), len(features)))
    dummy_input_lstm[:, target_idx] = preds_scaled_lstm.flatten()
    lstm_preds = scaler_lstm.inverse_transform(dummy_input_lstm)[:, target_idx]
    
    
    # --- 2. Train Transformer (Log1p Targets) ---
    print("\n--- 2. Training Transformer ---")
    df_tft = global_df.copy()
    df_tft['review_count'] = np.log1p(df_tft['review_count'])
    data_tft = df_tft[features].values
    
    scaler_tft = MinMaxScaler(feature_range=(0, 1))
    scaler_tft.fit(data_tft[:train_idx+1])
    scaled_data_tft = scaler_tft.transform(data_tft)
    
    X_t, y_t = create_sequences(scaled_data_tft, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    X_train_t = torch.FloatTensor(X_t[:train_idx - lookback + 1]).to(device)
    y_train_t = torch.FloatTensor(y_t[:train_idx - lookback + 1]).to(device)
    X_test_t = torch.FloatTensor(X_t[train_idx - lookback + 1:]).to(device)
    
    train_dataset_t = TensorDataset(X_train_t, y_train_t)
    train_loader_t = DataLoader(train_dataset_t, batch_size=8, shuffle=True)
    
    torch.manual_seed(42)
    model_tft = TimeSeriesTransformer(num_features=len(features)).to(device)
    optimizer_tft = torch.optim.Adam(model_tft.parameters(), lr=0.001)
    
    best_loss = float('inf')
    early_stop_counter = 0
    model_tft.train()
    
    for epoch in range(200):
        epoch_loss = 0
        for batch_X, batch_y in train_loader_t:
            optimizer_tft.zero_grad()
            outputs = model_tft(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_tft.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader_t)
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            best_model_state_tft = model_tft.state_dict()
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            break
            
    model_tft.load_state_dict(best_model_state_tft)
    model_tft.eval()
    with torch.no_grad():
        preds_scaled_tft = model_tft(X_test_t).cpu().numpy()
        
    dummy_input_tft = np.zeros((len(preds_scaled_tft), len(features)))
    dummy_input_tft[:, target_idx] = preds_scaled_tft.flatten()
    preds_log = scaler_tft.inverse_transform(dummy_input_tft)[:, target_idx]
    tft_preds = np.expm1(preds_log)
    
    
    # --- 3. Deep Ensemble Blending ---
    print("\n--- 3. Evaluating Deep Ensemble (50% LSTM + 50% Transformer) ---")
    
    # Simple average
    ensemble_preds = (lstm_preds + tft_preds) / 2.0
    
    # Actual y
    actuals = global_df['review_count'].values[train_idx+1:]
    
    lstm_metrics = get_metrics(actuals, lstm_preds)
    tft_metrics = get_metrics(actuals, tft_preds)
    ensemble_metrics = get_metrics(actuals, ensemble_preds)
    
    print(f"\nLSTM Metrics: {lstm_metrics}")
    print(f"Transformer Metrics: {tft_metrics}")
    print(f"Deep Ensemble Metrics: {ensemble_metrics}")
    
    # Save to metrics file
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        # Drop old ensemble if exists
        try:
            baseline_metrics = baseline_metrics.drop(index='Ensemble (Prophet+TFT)')
        except:
            pass
            
        baseline_metrics.loc['Deep Ensemble (LSTM+TFT)'] = ensemble_metrics
        baseline_metrics.to_csv(baseline_path)
        print("\nUpdated Metrics File:")
        print(baseline_metrics)
        
    # Plotting
    test_dates = global_df['month'].iloc[-len(actuals):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actuals (Global)', color='black', alpha=0.5)
    plt.plot(test_dates, actuals, label='Test Actuals', color='blue', marker='o')
    plt.plot(test_dates, ensemble_preds, label='Deep Ensemble Forecast', color='red', marker='*', markersize=10, linestyle='-')
    plt.plot(test_dates, lstm_preds, label='LSTM Only', color='green', linestyle=':', alpha=0.6)
    plt.plot(test_dates, tft_preds, label='Transformer Only', color='purple', linestyle='--', alpha=0.6)
    
    plt.title('Global Review Count Forecast: Actuals vs Deep Ensemble (LSTM + Transformer)')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '11_ensemble_forecasts.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved Full Ensemble plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    run_deep_ensemble(input_file, output_directory)
