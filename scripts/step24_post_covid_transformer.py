import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math
import pickle

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

# Simplified Time-Series Transformer (Vanilla)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
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
        
        # Linear projection to d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Flatten and predict
        self.fc = nn.Linear(d_model, horizon)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, num_features)
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(x)
        
        # Take the output of the sequence to make prediction (can use last token or mean)
        output = output[:, -1, :] # Last sequence element
        
        prediction = self.fc(output)
        return prediction

def run_transformer(input_path, output_dir, models_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # --- POST-COVID FILTERING ---
    df = df[df['month'] >= '2022-01-01'].copy()
    print(f"Filtered to Post-COVID era. {len(df)} records from {df['month'].min().date()} to {df['month'].max().date()}")

    # --- EXCLUDE ANOMALOUS LOCATIONS ---
    excluded_locations = ['d6974493'] # Dacotour
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 1. Global Aggregation
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index()
    global_df = global_df.sort_values('month')
    
    global_df.reset_index(drop=True, inplace=True)
    
    # Define features
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count']
    
    # --- TARGET TRANSFORMATION ---
    # Log-transform the target to heavily penalize over/under forecasting spikes 
    # and improve MAPE symmetry.
    global_df['review_count'] = np.log1p(global_df['review_count'])
    
    data = global_df[features].values
    target_idx = features.index('review_count')
    
    # 2. Train-Test Split (Since data is small POST-2022, we use last 6 months for testing to maximize training size)
    # Available data is typically Jan 2022 -> Max (mid 2024).
    split_idx = len(global_df) - 6
    train_idx = split_idx - 1 # Index of the last training sample
    
    # 3. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_idx+1])
    scaled_data = scaler.transform(data)
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, 'scaler_post_covid.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # 4. Create sequences (Sliding Window)
    lookback = 12
    horizon = 1
    X, y = create_sequences(scaled_data, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    # Note: If train_idx - lookback + 1 <= 0, we don't have enough data to train with lookback=12 and test=6.
    # We may need to ensure we have enough points. 2022 to mid 2024 = 30 months roughly.
    # 30 - 12 (lookback) - 1 (horizon) = 17 sequences.
    
    X_train_np = X[:train_idx - lookback + 1]
    y_train_np = y[:train_idx - lookback + 1]
    
    X_test_np = X[train_idx - lookback + 1:]
    y_test_np = y[train_idx - lookback + 1:]
    
    print(f"X_train shape: {X_train_np.shape}, y_train shape: {y_train_np.shape}")
    print(f"X_test shape: {X_test_np.shape}, y_test shape: {y_test_np.shape}")
    
    if len(X_train_np) <= 0:
        raise ValueError("Not enough post-COVID data to train with lookback=12. Check dataset date ranges.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    y_test = torch.FloatTensor(y_test_np).to(device)
    
    # Use smaller batch size for small dataset
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 5. Build SUPER-OPTIMIZED Transformer Model
    print("Building and Training MAE-Optimized Transformer Model (PyTorch)...")
    torch.manual_seed(42)
    
    # Increased dim_feedforward for better feature representations
    model = TimeSeriesTransformer(num_features=len(features), d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=horizon).to(device)
    
    # SUPER OPTIMIZATION 1: Huber Loss (Smooth L1) forces the model to hyper-focus on MAPE/MAE
    criterion = nn.SmoothL1Loss() 
    
    # SUPER OPTIMIZATION 2: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # SUPER OPTIMIZATION 3: Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    
    # Training Loop with early stopping
    epochs = 400
    patience = 50
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
            
        scheduler.step() # Advance the LR scheduler
        
        avg_loss = epoch_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
            
        if (epoch+1) % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
            
    # Load best weights
    model.load_state_dict(best_model_state)
    
    # Save the model
    model_path = os.path.join(models_dir, 'transformer_post_covid.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 6. Evaluation
    print("Evaluating Super-Optimized Transformer Model on Test Set...")
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test).cpu().numpy()
        
    dummy_input = np.zeros((len(predictions_scaled), len(features)))
    dummy_input[:, target_idx] = predictions_scaled.flatten()
    # Inverse scale, then inverse log
    predictions_log = scaler.inverse_transform(dummy_input)[:, target_idx]
    predictions_real = np.expm1(predictions_log)
    
    dummy_input[:, target_idx] = y_test_np.flatten()
    y_test_log = scaler.inverse_transform(dummy_input)[:, target_idx]
    y_test_real = np.expm1(y_test_log)
    
    transformer_metrics = get_metrics(y_test_real, predictions_real)
    print("\n--- Post-COVID Transformer Metrics (Test Set) ---")
    print(transformer_metrics)
    
    # Compare with Baseline
    baseline_path = os.path.join(output_dir, 'post_covid_metrics.csv')
    df_metrics = pd.DataFrame([transformer_metrics], index=['Post-COVID Transformer'])
    df_metrics.to_csv(baseline_path)
    print("\nSaved Metrics File:")
    print(df_metrics)
    
    # 7. Plotting
    actual_real_counts = np.expm1(global_df['review_count'].values)
    test_dates = global_df['month'].iloc[-len(y_test_real):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], actual_real_counts, label='Actuals (Global)', color='black', alpha=0.5)
    plt.plot(test_dates, y_test_real, label='Test Actuals', color='blue', marker='o')
    plt.plot(test_dates, predictions_real, label='Transformer Forecast', color='purple', marker='^', linestyle='--')
    
    plt.title('Post-COVID Global Review Count Forecast')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '24_post_covid_transformer_forecasts.png')
    plt.savefig(plot_path)
    plt.close()
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    models_directory = os.path.join(project_dir, 'models')
    
    run_transformer(input_file, output_directory, models_directory)
