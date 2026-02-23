import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pickle
import matplotlib.pyplot as plt

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
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def create_sequences(data, target_col_idx, lookback=12, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
    return np.array(X), np.array(y)


# --- ARCHITECTURES ---

# 1. Simple Deep LSTM
class Simple_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(Simple_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 2. CNN-LSTM
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


# 3. BiLSTM-Attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = self.softmax(attn_weights)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) 
        return context_vector, attn_weights

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTMAttention, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = SelfAttention(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_out, attn_weights = self.attention(lstm_out)
        
        out = self.fc1(attn_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# 4. Time-Series Transformer
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


# --- ARENA EXECUTION ---
def run_post_covid_arena(input_path, output_dir, models_dir):
    print(f"Loading data from {input_path} for Post-COVID Arena...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # 1. Post-Covid Filter
    df = df[df['month'] >= '2022-01-01'].copy()
    
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 2. Global Aggregation
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index().sort_values('month')
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count']
    
    # Smoothing Target (Log1p allows mapping highly volatile spikes downward)
    global_df['review_count'] = np.log1p(global_df['review_count'])
    
    data = global_df[features].values
    target_idx = features.index('review_count')
    
    # Tiny test set for Post-Covid (last 6 months)
    split_idx = len(global_df) - 6
    train_idx = split_idx - 1
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_idx+1])
    scaled_data = scaler.transform(data)
    
    scaler_path = os.path.join(models_dir, 'scaler_post_covid_hyper.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    lookback = 12
    horizon = 1
    X, y = create_sequences(scaled_data, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    X_train_np = X[:train_idx - lookback + 1]
    y_train_np = y[:train_idx - lookback + 1]
    X_test_np = X[train_idx - lookback + 1:]
    y_test_np = y[train_idx - lookback + 1:]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    y_test = torch.FloatTensor(y_test_np).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Models to evaluate
    num_features = len(features)
    models_dict = {
        'TimeSeriesTransformer': TimeSeriesTransformer(num_features=num_features, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1),
        'Simple_LSTM': Simple_LSTM(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2),
        'CNN_LSTM': CNN_LSTM(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2),
        'BiLSTMAttention': BiLSTMAttention(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.3)
    }
    
    criterion = nn.SmoothL1Loss() # Huber Loss
    epochs = 400
    patience = 50
    
    arena_results = {}
    best_overall_mape = float('inf')
    best_overall_model_name = ""
    best_overall_model_state = None
    
    print("\n" + "="*50)
    print("BEGINNING POST-COVID HYPER ARENA")
    print("="*50)

    for name, model in models_dict.items():
        print(f"\n>> Training [ {name} ] ...")
        torch.manual_seed(42)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        
        best_loss = float('inf')
        early_stop_counter = 0
        best_state = None
        
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
                
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stop_counter = 0
                best_state = model.state_dict()
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                break
                
        # Inference
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test).cpu().numpy()
            
        dummy_input = np.zeros((len(preds_scaled), len(features)))
        dummy_input[:, target_idx] = preds_scaled.flatten()
        preds_log = scaler.inverse_transform(dummy_input)[:, target_idx]
        preds_real = np.expm1(preds_log)
        
        dummy_test = np.zeros((len(y_test_np), len(features)))
        dummy_test[:, target_idx] = y_test_np.flatten()
        y_test_log = scaler.inverse_transform(dummy_test)[:, target_idx]
        y_test_real = np.expm1(y_test_log)
        
        metrics = get_metrics(y_test_real, preds_real)
        arena_results[name] = metrics
        print(f"   [ {name} ] MAPE: {metrics['MAPE']:.2f}% | MAE: {metrics['MAE']:.2f}")
        
        if metrics['MAPE'] < best_overall_mape:
            best_overall_mape = metrics['MAPE']
            best_overall_model_name = name
            import copy
            best_overall_model_state = copy.deepcopy(best_state)
            
    # Save Leaderboard
    df_metrics = pd.DataFrame.from_dict(arena_results, orient='index')
    df_metrics = df_metrics.sort_values(by='MAPE')
    
    csv_path = os.path.join(output_dir, 'post_covid_arena_metrics.csv')
    df_metrics.to_csv(csv_path)
    print("\n" + "="*50)
    print(f"ARENA COMPLETE. WINNER: {best_overall_model_name} (MAPE: {best_overall_mape:.2f}%)")
    print("Leaderboard Exported.")
    print("="*50)
    
    # Save the absolute best model
    best_model_path = os.path.join(models_dir, 'best_post_covid_model.pt')
    torch.save(best_overall_model_state, best_model_path)
    
    meta_path = os.path.join(models_dir, 'best_model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({"winning_architecture": best_overall_model_name}, f)
        
    print(f"Saved best model weights to: {best_model_path}")
    print(f"Saved model metadata to: {meta_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    models_directory = os.path.join(project_dir, 'models')
    
    run_post_covid_arena(input_file, output_directory, models_directory)
