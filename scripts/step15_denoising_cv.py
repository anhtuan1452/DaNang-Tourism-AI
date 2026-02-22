import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

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

def run_denoising_cv(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    excluded_locations = ['d6974493'] # Remove anomaly Dacotour
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
    
    # ---------------------------------------------------------
    # SCIENTIFIC TRICK 1: TARGET DENOISING (Moving Average)
    # Tripadvisor reviews are extremely "spiky" because of batch-uploads or crawling artifacts.
    # Applying a small Rolling Average (k=2) to the Target preserves the real macro-trend 
    # while eliminating the micro-noise that confuses Neural Networks.
    # ---------------------------------------------------------
    global_df['target_smoothed'] = global_df['review_count'].rolling(window=2, min_periods=1).mean()
    
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    features = ['target_smoothed', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('target_smoothed')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # SCIENTIFIC TRICK 2: TIME-SERIES CROSS VALIDATION
    # Instead of one static split (2017-2022 vs 2023-2024), we use Expanding Window CV.
    # This proves the model is robust across all time periods, not just lucky on 2023.
    # ---------------------------------------------------------
    data = global_df[features].values
    
    tscv = TimeSeriesSplit(n_splits=3, test_size=12) # 3 folds, predict 12 months ahead at each step
    
    fold = 1
    cv_scores = []
    
    for train_index, test_index in tscv.split(data):
        print(f"\n--- Cross Validation Fold {fold} ---")
        
        # Scaling based only on current train fold
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data[train_index])
        
        # Scale both train and test fold
        scaled_train = scaler.transform(data[train_index])
        
        # We need lookback from train to predict the first test element
        # So we create a combined chunk
        combined_chunk = np.vstack([data[train_index], data[test_index]])
        scaled_combined = scaler.transform(combined_chunk)
        
        lookback = 12
        horizon = 1
        
        # Sequences for Train
        X_train_fold, y_train_fold = create_sequences(scaled_train, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
        
        # Sequences for Test (Extracted from the end of combined)
        X_test_all, y_test_all = create_sequences(scaled_combined, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
        X_test_fold = X_test_all[-len(test_index):]
        y_test_fold = y_test_all[-len(test_index):]
        
        X_train_t = torch.FloatTensor(X_train_fold).to(device)
        y_train_t = torch.FloatTensor(y_train_fold).to(device)
        X_test_t = torch.FloatTensor(X_test_fold).to(device)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)
        
        # Build Model
        torch.manual_seed(42)
        model = TourismLSTM(input_size=len(features)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience = 20
        early_stop = 0
        
        model.train()
        for epoch in range(150):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = model.state_dict()
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= patience:
                    break
                    
        model.load_state_dict(best_model)
        model.eval()
        
        with torch.no_grad():
            preds_scaled = model(X_test_t).cpu().numpy()
            
        dummy_pred = np.zeros((len(preds_scaled), len(features)))
        dummy_pred[:, target_idx] = preds_scaled.flatten()
        preds_real = scaler.inverse_transform(dummy_pred)[:, target_idx]
        
        # Compare against the RAW unsmoothed actuals from global_df to be mathematically fair!
        actuals_raw = global_df.iloc[test_index]['review_count'].values
        
        metrics = get_metrics(actuals_raw, preds_real)
        print(f"Fold {fold} Metrics (Tested on {len(test_index)} months): {metrics}")
        cv_scores.append(metrics)
        fold += 1
        
    # Aggregate CV Results
    avg_mae = np.mean([x['MAE'] for x in cv_scores])
    avg_rmse = np.mean([x['RMSE'] for x in cv_scores])
    avg_mape = np.mean([x['MAPE'] for x in cv_scores])
    
    print(f"\n=== FINAL CROSS-VALIDATION SOTA METRICS ===")
    print(f"Robust Denoised LSTM -> Average MAE: {avg_mae:.2f} | Average RMSE: {avg_rmse:.2f} | Average MAPE: {avg_mape:.2f}%")
    
    # Save a final comparative visualization using the last fold
    last_test_dates = global_df['month'].iloc[-len(preds_real):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'].iloc[-36:], global_df['review_count'].iloc[-36:], label='Raw Review Highlights', color='black', alpha=0.3, linestyle=':')
    plt.plot(global_df['month'].iloc[-36:], global_df['target_smoothed'].iloc[-36:], label='Denoised Signal (Input to AI)', color='teal', linewidth=2)
    plt.plot(last_test_dates, preds_real, label=f'LSTM Forecast (Fold {fold-1})', color='crimson', marker='X', markersize=8)
    
    plt.title('Denoised LSTM Forecast with Time-Series Cross Validation')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cv_plot_path = os.path.join(output_dir, '13_denoising_cv.png')
    plt.savefig(cv_plot_path)
    plt.close()
    print(f"Saved Denoising CV plot to {cv_plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    run_denoising_cv(input_file, output_directory)
