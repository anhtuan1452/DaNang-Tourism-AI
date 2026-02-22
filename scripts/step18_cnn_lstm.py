import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# --- ADVANCED CNN-LSTM ARCHITECTURE ---
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        
        # 1D Convolution to extract features and smooth out noise in the sequence
        # input shape: (batch_size, input_size, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.1)
        
        # LSTM layer processes the sequence of feature maps extracted by CNN
        # input shape: (batch_size, seq_len, 64)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        # x is (batch, seq_len, features). PyTorch Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        # Permute back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM block
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take the last time step output
        
        # Dense block
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out

def run_cnn_lstm(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 100% Scientific alignment: Keeping all Ecosystem Data
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    # Target Denoising for stability
    global_df['target_smoothed'] = global_df['review_count'].rolling(window=2, min_periods=1).mean()
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['target_smoothed', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('target_smoothed')
    
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    data = global_df[features].values
    
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
    
    print("Building and Training ConvLSTM Model...")
    torch.manual_seed(42)
    
    model = CNN_LSTM(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
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
    preds_real = scaler.inverse_transform(dummy_input)[:, target_idx]
    
    # Evaluate against RAW UNSMOOTHED Actuals
    actuals_raw = global_df['review_count'].values[train_idx+1:]
    
    metrics = get_metrics(actuals_raw, preds_real)
    print(f"\n[SUCCESS] CNN-LSTM METRICS: {metrics}")
    
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        baseline_metrics.loc['CNN-LSTM (Full Data)'] = metrics
        baseline_metrics.to_csv(baseline_path)
    
    test_dates = global_df['month'].iloc[-len(actuals_raw):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actual Full Ecosystem Reviews', color='black', alpha=0.4)
    plt.plot(test_dates, actuals_raw, label='Test Actuals (2023-2024)', color='blue', marker='o')
    plt.plot(test_dates, preds_real, label='Deep CNN-LSTM Forecast', color='red', marker='*')
    
    plt.title('CNN-LSTM Forecast (Optimized Science-keeping Approach)')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '16_cnn_lstm_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved CNN-LSTM plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_cnn_lstm(input_file, output_directory)
