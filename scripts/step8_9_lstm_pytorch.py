import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

# PyTorch LSTM Model
class TourismLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
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
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def run_lstm(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # --- ABLATION EXPERIMENT: Exclude Anomalous Locations ---
    # d6974493 is Dacotour. It has massive review spikes in late 2025/2026
    # which leaks future data into the timeline and heavily distorts moving averages / seasonality
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    print(f"Excluded {len(excluded_locations)} anomalous locations.")
    
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
    
    # Add COVID-19 feature flag instead of dropping the data
    # 2020 and 2021 are considered COVID years
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    # Define features
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    data = global_df[features].values
    target_idx = features.index('review_count')
    
    # 2. Train-Test Split (Time-based: 2017-2022 vs 2023-2024)
    split_date = pd.to_datetime('2022-12-31')
    train_idx = global_df[global_df['month'] <= split_date].index[-1]
    
    # 3. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_idx+1])
    scaled_data = scaler.transform(data)
    
    # 4. Create sequences (Sliding Window)
    lookback = 12
    horizon = 1
    X, y = create_sequences(scaled_data, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    # Split sequences
    X_train_np = X[:train_idx - lookback + 1]
    y_train_np = y[:train_idx - lookback + 1]
    
    X_test_np = X[train_idx - lookback + 1:]
    y_test_np = y[train_idx - lookback + 1:]
    
    print(f"X_train shape: {X_train_np.shape}, y_train shape: {y_train_np.shape}")
    print(f"X_test shape: {X_test_np.shape}, y_test shape: {y_test_np.shape}")
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    y_test = torch.FloatTensor(y_test_np).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 5. Build LSTM Model
    print("Building and Training LSTM Model (PyTorch)...")
    torch.manual_seed(42)
    
    input_size = len(features)
    hidden_size = 64
    num_layers = 1
    output_size = horizon
    
    model = TourismLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Training Loop with simple early stopping
    epochs = 200
    patience = 20
    best_loss = float('inf')
    early_stop_counter = 0
    
    model.train()
    for epoch in range(epochs):
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
            # Save best weights (simple approach)
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
            
    # Load best weights
    model.load_state_dict(best_model_state)
    
    # 6. Evaluation
    print("Evaluating Model on Test Set...")
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test).cpu().numpy()
        
    dummy_input = np.zeros((len(predictions_scaled), len(features)))
    dummy_input[:, target_idx] = predictions_scaled.flatten()
    predictions_real = scaler.inverse_transform(dummy_input)[:, target_idx]
    
    dummy_input[:, target_idx] = y_test_np.flatten()
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
    
    print(f"\nSaved LSTM plot with COVID Flag to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_lstm(input_file, output_directory)
