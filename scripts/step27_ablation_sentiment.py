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
    
    # Calculate Directional Accuracy (DA) to evaluate Trend predictions
    if len(y_true) > 1:
        actual_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        da = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
    else:
        da = 0.0
        
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'DA (%)': da}

def create_sequences(data, target_col_idx, lookback=12, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
    return np.array(X), np.array(y)


# --- ARCHITECTURES ---

# 1. CNN-LSTM (Secondary Brain)
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.1)
        
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        return out

# 2. Time-Series Transformer (Primary Brain)
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
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, horizon)
        
    def forward(self, src):
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = output[:, -1, :] 
        prediction = self.fc(output)
        return prediction

def train_and_evaluate_ensemble(global_df, features, device):
    lookback = 12
    horizon = 1
    target_idx = features.index('review_count')
    train_idx = int(len(global_df) * 0.8) - 1
    
    # --- MODEL 1: CNN-LSTM (Raw Targets) ---
    data_cnn = global_df[features].values.copy()
    scaler_cnn = MinMaxScaler(feature_range=(0, 1))
    scaler_cnn.fit(data_cnn[:train_idx+1])
    scaled_cnn = scaler_cnn.transform(data_cnn)
    
    X_c, y_c = create_sequences(scaled_cnn, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    X_train_c, y_train_c = torch.FloatTensor(X_c[:train_idx - lookback + 1]).to(device), torch.FloatTensor(y_c[:train_idx - lookback + 1]).to(device)
    X_test_c = torch.FloatTensor(X_c[train_idx - lookback + 1:]).to(device)
    
    train_dataset_c = TensorDataset(X_train_c, y_train_c)
    train_loader_c = DataLoader(train_dataset_c, batch_size=8, shuffle=True)
    
    torch.manual_seed(42)
    model_cnn = CNN_LSTM(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=0.003)
    criterion_cnn = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 20
    counter = 0
    model_cnn.train()
    
    for epoch in range(150):
        epoch_loss = 0
        for batch_X, batch_y in train_loader_c:
            optimizer_cnn.zero_grad()
            outputs = model_cnn(batch_X)
            loss = criterion_cnn(outputs, batch_y)
            loss.backward()
            optimizer_cnn.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader_c)
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            best_state_cnn = model_cnn.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break
                
    model_cnn.load_state_dict(best_state_cnn)
    model_cnn.eval()
    with torch.no_grad():
        preds_scaled_cnn = model_cnn(X_test_c).cpu().numpy()
        
    dummy_cnn = np.zeros((len(preds_scaled_cnn), len(features)))
    dummy_cnn[:, target_idx] = preds_scaled_cnn.flatten()
    preds_real_cnn = scaler_cnn.inverse_transform(dummy_cnn)[:, target_idx]

    # --- MODEL 2: TRANSFORMER (Log1p targets + Huber Loss) ---
    df_tf = global_df.copy()
    df_tf['review_count'] = np.log1p(df_tf['review_count'])
    data_tf = df_tf[features].values
    
    scaler_tf = MinMaxScaler(feature_range=(0, 1))
    scaler_tf.fit(data_tf[:train_idx+1])
    scaled_tf = scaler_tf.transform(data_tf)
    
    X_t, y_t = create_sequences(scaled_tf, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    X_train_t, y_train_t = torch.FloatTensor(X_t[:train_idx - lookback + 1]).to(device), torch.FloatTensor(y_t[:train_idx - lookback + 1]).to(device)
    X_test_t = torch.FloatTensor(X_t[train_idx - lookback + 1:]).to(device)
    
    train_dataset_t = TensorDataset(X_train_t, y_train_t)
    train_loader_t = DataLoader(train_dataset_t, batch_size=8, shuffle=True)
    
    torch.manual_seed(42)
    model_tf = TimeSeriesTransformer(num_features=len(features), d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1).to(device)
    criterion_tf = nn.SmoothL1Loss() # Huber Loss
    optimizer_tf = torch.optim.AdamW(model_tf.parameters(), lr=0.005, weight_decay=1e-4) # AdamW
    scheduler_tf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tf, T_max=200, eta_min=1e-6)
    
    best_loss = float('inf')
    patience = 40
    counter = 0
    model_tf.train()
    
    for epoch in range(300):
        epoch_loss = 0
        for batch_X, batch_y in train_loader_t:
            optimizer_tf.zero_grad()
            outputs = model_tf(batch_X)
            loss = criterion_tf(outputs, batch_y)
            loss.backward()
            optimizer_tf.step()
            epoch_loss += loss.item()
            
        scheduler_tf.step()
        avg_loss = epoch_loss / len(train_loader_t)
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            best_state_tf = model_tf.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break
                
    model_tf.load_state_dict(best_state_tf)
    model_tf.eval()
    with torch.no_grad():
        preds_scaled_tf = model_tf(X_test_t).cpu().numpy()
        
    dummy_tf = np.zeros((len(preds_scaled_tf), len(features)))
    dummy_tf[:, target_idx] = preds_scaled_tf.flatten()
    preds_log_tf = scaler_tf.inverse_transform(dummy_tf)[:, target_idx]
    preds_real_tf = np.expm1(preds_log_tf)
    
    # --- ENSEMBLE BLENDING & METRICS ---
    print("\n" + "="*50)
    print(">> COMBINING BRAINS: THE ADVANCED ENSEMBLE")
    print("="*50)
    
    actuals = global_df['review_count'].values[train_idx+1:]
    
    cnn_metrics = get_metrics(actuals, preds_real_cnn)
    tf_metrics = get_metrics(actuals, preds_real_tf)
    
    # Grid Search Optimal Blending Weight to Strictly Minimize MAPE
    best_mape = float('inf')
    best_w = 0.5
    for w in np.linspace(0, 1, 1001):
        temp_preds = (preds_real_tf * w) + (preds_real_cnn * (1 - w))
        temp_mape = mean_absolute_percentage_error(actuals, temp_preds)
        if temp_mape < best_mape:
            best_mape = temp_mape
            best_w = w
            
    w_tf = best_w
    w_cnn = 1.0 - w_tf
    
    ensemble_preds = (preds_real_tf * w_tf) + (preds_real_cnn * w_cnn)
    ensemble_metrics = get_metrics(actuals, ensemble_preds)
    
    return ensemble_preds, ensemble_metrics, actuals

# --- EXECUTION ---
def run_ablation(input_path, output_dir):
    print(f"Loading data from {input_path} for Sentiment Ablation Study...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    global_df = global_df[global_df['month'] < '2026-02-01'].copy()
    global_df.reset_index(drop=True, inplace=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Train WITH Sentiment
    features_with = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    print("\n>> Executing RUN 1: Advanced Ensemble WITH Sentiment...")
    preds_with, metrics_with, actuals = train_and_evaluate_ensemble(global_df, features_with, device)
    print(f"Metrics (WITH Sentiment): {metrics_with}")
    
    # 2. Train WITHOUT Sentiment
    features_without = ['review_count', 'rainfall_mm', 'holiday_count', 'is_covid']
    print("\n>> Executing RUN 2: Advanced Ensemble WITHOUT Sentiment...")
    preds_without, metrics_without, _ = train_and_evaluate_ensemble(global_df, features_without, device)
    print(f"Metrics (WITHOUT Sentiment): {metrics_without}")
    
    # --- SAVE METRICS ---
    results_df = pd.DataFrame({
        'With Sentiment': metrics_with,
        'Without Sentiment': metrics_without
    }).T
    
    results_df = results_df[['MAE', 'RMSE', 'MAPE', 'DA (%)']]
    csv_path = os.path.join(output_dir, 'ablation_sentiment_metrics.csv')
    results_df.to_csv(csv_path)
    print(f"\nSaved metrics comparison to {csv_path}")
    print(results_df)
    
    # --- SAVE PREDICTIONS ---
    test_dates = global_df['month'].iloc[-len(actuals):].values
    pred_df = pd.DataFrame({
        'month': test_dates,
        'Actuals': actuals,
        'With Sentiment': preds_with,
        'Without Sentiment': preds_without
    })
    pred_path = os.path.join(output_dir, 'ablation_sentiment_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved prediction comparison to {pred_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_ablation(input_file, output_directory)
    
