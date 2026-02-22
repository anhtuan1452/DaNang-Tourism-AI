import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
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

# --- THE MIGHTY JOINT ARCHITECTURE ---
class JointLSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_lstm=64, d_model=32, nhead=4, num_layers_tft=2, output_size=1, dropout=0.2):
        super(JointLSTMTransformer, self).__init__()
        
        # Branch 1: LSTM (Excellent for Sequential Memory & Trend)
        self.lstm = nn.LSTM(input_size, hidden_lstm, num_layers=2, batch_first=True, dropout=dropout)
        
        # Branch 2: Transformer (Excellent for Attention & Shocks)
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_tft)
        
        # Fusion Multi-Layer Perceptron (Meta-Learner)
        # It takes the deeply extracted features from BOTH branches and fuses them
        self.fc1 = nn.Linear(hidden_lstm + d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        # 1. LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_final_state = lstm_out[:, -1, :] # (batch, hidden_lstm)
        
        # 2. Transformer processing
        x_tft = self.input_projection(x)
        x_tft = self.pos_encoder(x_tft)
        tft_out = self.transformer(x_tft)
        tft_final_state = tft_out[:, -1, :] # (batch, d_model)
        
        # 3. Deep Fusion
        combined = torch.cat((lstm_final_state, tft_final_state), dim=1) # (batch, hidden_lstm + d_model)
        
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def run_joint_mae_optimized(input_path, output_dir):
    print(f"Loading FULL ECOSYSTEM data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # Keeping 3301 rows strictly (excluding one known pure outlier to prevent model explosion)
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Global Aggregation (Preserving all Hotel, Restaurant, and Attraction Data)
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    # Target Denoising 
    global_df['target_smoothed'] = global_df['review_count'].rolling(window=2, min_periods=1).mean()
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['target_smoothed', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('target_smoothed')
    
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    # ---------------------------------------------------------------------------------
    # SCIENTIFIC STRATEGY: Log1p Transformation on targets to kill massive variance
    # This specifically forces the deep neural network to optimize MAE instead of RMSE.
    # ---------------------------------------------------------------------------------
    data = global_df[features].values.copy()
    data[:, target_idx] = np.log1p(data[:, target_idx]) # Transform target to Log scale
    
    scaler = MinMaxScaler(feature_range=(0, 1))
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    print("Training Joint LSTM-Transformer Network (Optimized for MAE via Huber Loss)...")
    torch.manual_seed(42)
    
    model = JointLSTMTransformer(input_size=len(features), hidden_lstm=64, d_model=32, nhead=4, dropout=0.2).to(device)
    
    # SCIENTIFIC STRATEGY: SmoothL1Loss (Huber Loss)
    # L1 Loss optimizes purely for MAE. SmoothL1 is L1 Loss that is rounded near zero to prevent exploding gradients.
    criterion = nn.SmoothL1Loss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4) # AdamW for better generalization
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 20
    
    model.train()
    for epoch in range(150):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break
                
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()
        
    dummy_input = np.zeros((len(preds_scaled), len(features)))
    dummy_input[:, target_idx] = preds_scaled.flatten()
    
    # Inversely Transform
    preds_log = scaler.inverse_transform(dummy_input)[:, target_idx]
    # Reverse Log1p
    preds_real = np.expm1(preds_log)
    
    # Validate against actual ecosystem raw values
    actuals_raw = global_df['review_count'].values[train_idx+1:]
    
    metrics = get_metrics(actuals_raw, preds_real)
    print(f"\n[SUCCESS] JOINT LSTM-TRANSFORMER METRICS: {metrics}")
    
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        baseline_metrics.loc['Joint End-to-End LSTM-Transformer (Full Data)'] = metrics
        baseline_metrics.to_csv(baseline_path)
    
    test_dates = global_df['month'].iloc[-len(actuals_raw):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actual Full Ecosystem Reviews', color='black', alpha=0.4)
    plt.plot(test_dates, actuals_raw, label='Test Actuals (2023-2024)', color='blue', marker='o')
    plt.plot(test_dates, preds_real, label='Joint LSTM-Transformer Forecast', color='red', marker='*')
    
    plt.title('Joint End-to-End Deep Architecture (Optimized via Huber Loss)')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '18_joint_lstm_transformer_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved Joint Model plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_joint_mae_optimized(input_file, output_directory)
