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

class JointLSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_lstm=64, d_model=32, nhead=4, num_layers_tft=2, output_size=1, dropout=0.2):
        super(JointLSTMTransformer, self).__init__()
        
        # To enable Monte Carlo Dropout for Uncertainty estimation,
        # we must ensure dropout layers remain active during inference.
        self.lstm = nn.LSTM(input_size, hidden_lstm, num_layers=2, batch_first=True, dropout=dropout)
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_tft)
        
        self.fc1 = nn.Linear(hidden_lstm + d_model, 64)
        self.relu = nn.ReLU()
        # Explicit Dropout layer that we can force to stay active
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_final_state = lstm_out[:, -1, :]
        
        x_tft = self.input_projection(x)
        x_tft = self.pos_encoder(x_tft)
        tft_out = self.transformer(x_tft)
        tft_final_state = tft_out[:, -1, :]
        
        combined = torch.cat((lstm_final_state, tft_final_state), dim=1)
        
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out

def run_mc_dropout_intervals(input_path, output_dir):
    print(f"Loading FULL ECOSYSTEM data from {input_path}...")
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
    
    global_df['target_smoothed'] = global_df['review_count'].rolling(window=2, min_periods=1).mean()
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['target_smoothed', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('target_smoothed')
    
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    # Log1p Transformation
    data = global_df[features].values.copy()
    data[:, target_idx] = np.log1p(data[:, target_idx])
    
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
    
    print("Training Joint Network...")
    torch.manual_seed(42)
    
    model = JointLSTMTransformer(input_size=len(features), hidden_lstm=64, d_model=32, nhead=4, dropout=0.2).to(device)
    criterion = nn.SmoothL1Loss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
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
    
    # -------------------------------------------------------------
    # MONTE CARLO DROPOUT: Generate Prediction Intervals
    # -------------------------------------------------------------
    print("Running Monte Carlo Dropout Inference (100 forward passes) to estimate Uncertainty Boundaries...")
    # Keep the model in TRAIN mode to leave dropout active during inference!
    model.train() 
    
    num_mc_samples = 100
    mc_predictions = []
    
    with torch.no_grad():
        for _ in range(num_mc_samples):
            # Each forward pass drops different neurons yielding slight variations
            preds_scaled_sample = model(X_test).cpu().numpy()
            
            # Inverse Transform
            dummy_input = np.zeros((len(preds_scaled_sample), len(features)))
            dummy_input[:, target_idx] = preds_scaled_sample.flatten()
            preds_log = scaler.inverse_transform(dummy_input)[:, target_idx]
            preds_real_sample = np.expm1(preds_log)
            
            mc_predictions.append(preds_real_sample)
            
    mc_predictions = np.array(mc_predictions) # (100, test_len)
    
    # Calculate Mean and Percentiles
    mean_preds = np.mean(mc_predictions, axis=0)
    lower_bound_95 = np.percentile(mc_predictions, 2.5, axis=0) # 2.5th percentile
    upper_bound_95 = np.percentile(mc_predictions, 97.5, axis=0) # 97.5th percentile
    
    actuals_raw = global_df['review_count'].values[train_idx+1:]
    test_dates = global_df['month'].iloc[-len(actuals_raw):].values
    
    metrics = get_metrics(actuals_raw, mean_preds)
    print(f"\n[SUCCESS] MC-DROPOUT MEAN METRICS: {metrics}")
    
    # Plotting Sequence with Intervals
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actual Full Ecosystem Reviews', color='black', alpha=0.4)
    plt.plot(test_dates, actuals_raw, label='Test Actuals (2023-2024)', color='blue', marker='o')
    plt.plot(test_dates, mean_preds, label='Deep Learning Forecast (MC Mean)', color='red', marker='*')
    
    # Fill between the 95% Confidence Bounds
    plt.fill_between(test_dates, lower_bound_95, upper_bound_95, color='red', alpha=0.2, label='95% Prediction Interval (Uncertainty)')
    
    plt.title('Deep Learning Forecast with Monte Carlo Dropout Prediction Intervals')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '20_deep_learning_prediction_intervals.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved Prediction Interval plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_mc_dropout_intervals(input_file, output_directory)
