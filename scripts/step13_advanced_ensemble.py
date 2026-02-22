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
from sklearn.linear_model import Ridge

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

def run_advanced_ensemble(input_path, output_dir):
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
    
    # ---------------------------------------------------------
    # 1. Train LSTM & get Train/Test Preds
    # ---------------------------------------------------------
    df_lstm = global_df.copy()
    data_lstm = df_lstm[features].values
    target_idx = features.index('review_count')
    
    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    scaler_lstm.fit(data_lstm[:train_idx+1])
    scaled_data_lstm = scaler_lstm.transform(data_lstm)
    
    lookback, horizon = 12, 1
    X_l, y_l = create_sequences(scaled_data_lstm, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    X_train_l = torch.FloatTensor(X_l[:train_idx - lookback + 1]).to(device)
    y_train_l = torch.FloatTensor(y_l[:train_idx - lookback + 1]).to(device)
    X_test_l = torch.FloatTensor(X_l[train_idx - lookback + 1:]).to(device)
    
    train_loader_l = DataLoader(TensorDataset(X_train_l, y_train_l), batch_size=8, shuffle=True)
    
    torch.manual_seed(42)
    model_lstm = TourismLSTM(input_size=len(features)).to(device)
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        model_lstm.train()
        for batch_X, batch_y in train_loader_l:
            optimizer_lstm.zero_grad()
            loss = criterion(model_lstm(batch_X), batch_y)
            loss.backward()
            optimizer_lstm.step()
            
    model_lstm.eval()
    with torch.no_grad():
        preds_train_scaled_lstm = model_lstm(X_train_l).cpu().numpy()
        preds_test_scaled_lstm = model_lstm(X_test_l).cpu().numpy()
        
    def inverse_scale_lstm(preds):
        dummy = np.zeros((len(preds), len(features)))
        dummy[:, target_idx] = preds.flatten()
        return scaler_lstm.inverse_transform(dummy)[:, target_idx]
        
    lstm_train_preds = inverse_scale_lstm(preds_train_scaled_lstm)
    lstm_test_preds = inverse_scale_lstm(preds_test_scaled_lstm)
    
    # ---------------------------------------------------------
    # 2. Train Transformer & get Train/Test Preds
    # ---------------------------------------------------------
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
    
    train_loader_t = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)
    
    torch.manual_seed(42)
    model_tft = TimeSeriesTransformer(num_features=len(features)).to(device)
    optimizer_tft = torch.optim.Adam(model_tft.parameters(), lr=0.001)
    
    for epoch in range(150):
        model_tft.train()
        for batch_X, batch_y in train_loader_t:
            optimizer_tft.zero_grad()
            loss = criterion(model_tft(batch_X), batch_y)
            loss.backward()
            optimizer_tft.step()
            
    model_tft.eval()
    with torch.no_grad():
        preds_train_scaled_tft = model_tft(X_train_t).cpu().numpy()
        preds_test_scaled_tft = model_tft(X_test_t).cpu().numpy()
        
    def inverse_scale_tft(preds):
        dummy = np.zeros((len(preds), len(features)))
        dummy[:, target_idx] = preds.flatten()
        return np.expm1(scaler_tft.inverse_transform(dummy)[:, target_idx])
        
    tft_train_preds = inverse_scale_tft(preds_train_scaled_tft)
    tft_test_preds = inverse_scale_tft(preds_test_scaled_tft)
    
    # ---------------------------------------------------------
    # 3. ADVANCED ENSEMBLE TECHNIQUES
    # ---------------------------------------------------------
    train_actuals = global_df['review_count'].values[lookback : train_idx+1]
    test_actuals = global_df['review_count'].values[train_idx+1:]
    
    # Technique A: Inverse-Error Weighting (Variance Based)
    # Calculate RMSE on Training set
    lstm_train_rmse = np.sqrt(mean_squared_error(train_actuals, lstm_train_preds))
    tft_train_rmse = np.sqrt(mean_squared_error(train_actuals, tft_train_preds))
    
    # Give higher weight to the model with lower error
    w_lstm = tft_train_rmse / (lstm_train_rmse + tft_train_rmse)
    w_tft = lstm_train_rmse / (lstm_train_rmse + tft_train_rmse)
    
    inv_err_preds = (w_lstm * lstm_test_preds) + (w_tft * tft_test_preds)
    
    # Technique B: Meta-Learner Stacking (Ridge Regression)
    # Train Meta Learner on the training predictions
    stack_X_train = np.column_stack((lstm_train_preds, tft_train_preds))
    stack_y_train = train_actuals
    
    meta_learner = Ridge(alpha=1.0, fit_intercept=True)
    meta_learner.fit(stack_X_train, stack_y_train)
    
    stack_X_test = np.column_stack((lstm_test_preds, tft_test_preds))
    meta_preds = meta_learner.predict(stack_X_test)
    
    print("\n--- Advanced Ensemble Results (Test Set) ---")
    print(f"LSTM Training RMSE: {lstm_train_rmse:.2f} | TFT Training RMSE: {tft_train_rmse:.2f}")
    print(f"Calculated Weights -> LSTM: {w_lstm:.2f}, TFT: {w_tft:.2f}")
    
    print(f"\nSimple Avg Ensemble (50/50):   MAE={mean_absolute_error(test_actuals, (lstm_test_preds+tft_test_preds)/2):.2f}")
    print(f"Inverse-Error Ensemble:        MAE={mean_absolute_error(test_actuals, inv_err_preds):.2f}")
    print(f"Meta-Learner Stacking (Ridge): MAE={mean_absolute_error(test_actuals, meta_preds):.2f}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    run_advanced_ensemble(input_file, output_directory)
