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

# --- ADVANCED ATTENTION MECHANISM ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        # Bidirectional brings hidden_size * 2
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_length, hidden_size * 2)
        attn_weights = self.attention(lstm_out) # (batch, seq, 1)
        attn_weights = self.softmax(attn_weights)
        # Multiply weights by lstm_out and sum across the sequence length
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_size * 2)
        return context_vector, attn_weights

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM to learn from both Past->Future and Future->Past context
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout)
                              
        self.attention = SelfAttention(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        # Apply Attention to focus on the most important time steps (e.g. months with holidays/shocks)
        attn_out, attn_weights = self.attention(lstm_out)
        
        out = self.fc1(attn_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def run_bilstm_attention(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # Keeping the original full dataset (Exclude only the extreme anomaly Dacotour)
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Global Aggregation of the FULL ECOSYSTEM (3301 rows converted to 127 macro-months)
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    
    # SCIENTIFIC TRICK: Apply Target Denoising to smooth out random daily noise, preserving real trend
    global_df['target_smoothed'] = global_df['review_count'].rolling(window=2, min_periods=1).mean()
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['target_smoothed', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('target_smoothed')
    
    # Train-Test Split (2017-2022 vs 2023-2024 cutoff)
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
    
    # Build BiLSTM-Attention Model
    print("Building and Training BiLSTM-Attention Model...")
    torch.manual_seed(42)
    
    model = BiLSTMAttention(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5) # Added weight decay for regularization
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 25
    
    model.train()
    for epoch in range(250):
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
    
    # Evaluate against RAW UNSMOOTHED Actuals to be 100% Scientific and Fair
    actuals_raw = global_df['review_count'].values[train_idx+1:]
    
    metrics = get_metrics(actuals_raw, preds_real)
    print(f"\n[SUCCESS] BiLSTM-ATTENTION METRICS: {metrics}")
    
    # Save to metrics file
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        try:
            baseline_metrics = baseline_metrics.drop(index='BiLSTM-Attention (Full Data)')
        except:
            pass
        baseline_metrics.loc['BiLSTM-Attention (Full Ecosystem + Denoising)'] = metrics
        baseline_metrics.to_csv(baseline_path)
    
    test_dates = global_df['month'].iloc[-len(actuals_raw):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actual Reviews (Full Ecosystem)', color='black', alpha=0.4)
    plt.plot(global_df['month'], global_df['target_smoothed'], label='Denoised Signal (Input)', color='teal', linestyle=':', linewidth=2)
    plt.plot(test_dates, actuals_raw, label='Test Actuals (2023-2024)', color='blue', marker='o')
    plt.plot(test_dates, preds_real, label='BiLSTM-Attention Forecast', color='red', marker='*')
    
    plt.title('BiLSTM with Attention Mechanism Forecast on Entire Ecosystem')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '15_bilstm_attention_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved BiLSTM-Attention plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_bilstm_attention(input_file, output_directory)
