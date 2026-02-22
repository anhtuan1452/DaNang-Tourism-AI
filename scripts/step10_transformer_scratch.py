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

def run_transformer(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # --- ABLATION EXPERIMENT: Exclude Anomalous Locations ---
    excluded_locations = ['d6974493'] # Dacotour
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Global Aggregation
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index()
    global_df = global_df.sort_values('month')
    
    # Add COVID-19 feature flag
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
    global_df.reset_index(drop=True, inplace=True)
    
    # Define features
    features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    
    # --- TARGET TRANSFORMATION ---
    # Log-transform the target to heavily penalize over/under forecasting spikes 
    # and improve MAPE symmetry.
    global_df['review_count'] = np.log1p(global_df['review_count'])
    
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
    
    X_train_np = X[:train_idx - lookback + 1]
    y_train_np = y[:train_idx - lookback + 1]
    
    X_test_np = X[train_idx - lookback + 1:]
    y_test_np = y[train_idx - lookback + 1:]
    
    print(f"X_train shape: {X_train_np.shape}, y_train shape: {y_train_np.shape}")
    print(f"X_test shape: {X_test_np.shape}, y_test shape: {y_test_np.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    y_test = torch.FloatTensor(y_test_np).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 5. Build Transformer Model
    print("Building and Training Vanilla Transformer Model (PyTorch)...")
    torch.manual_seed(42)
    
    model = TimeSeriesTransformer(num_features=len(features), d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, horizon=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop with early stopping
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
    print("Evaluating Transformer Model on Test Set...")
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
    print("\n--- Vanilla Transformer Metrics (Test Set 2023-2024) ---")
    print(transformer_metrics)
    
    # Compare with Baseline
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        baseline_metrics.loc['Transformer'] = transformer_metrics
        baseline_metrics.to_csv(baseline_path)
        print("\nUpdated Metrics File:")
        print(baseline_metrics)
    
    # 7. Plotting
    # Get actuals in real scale (not log) for plotting
    actual_real_counts = np.expm1(global_df['review_count'].values)
    test_dates = global_df['month'].iloc[-len(y_test_real):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], actual_real_counts, label='Actuals (Global)', color='black', alpha=0.5)
    plt.plot(test_dates, y_test_real, label='Test Actuals', color='blue', marker='o')
    plt.plot(test_dates, predictions_real, label='Transformer Forecast', color='purple', marker='^', linestyle='--')
    
    plt.title('Global Review Count Forecast: Actuals vs Transformer')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '7_transformer_forecasts.png')
    plt.savefig(plot_path)
    plt.close()
    
    # 8. Feature Importance (Permutation Method)
    print("\n--- Running Permutation Feature Importance ---")
    model.eval()
    baseline_rmse = transformer_metrics['RMSE']
    feature_importance = {}
    
    # We will perturb each feature in X_test and measure the increase in RMSE
    with torch.no_grad():
        for i, feature_name in enumerate(features):
            if feature_name in ['is_covid']: 
                continue # Skip static/flag features if desired, though we can still permute
                
            # Create a corrupted version of X_test
            X_test_corrupted = X_test_np.copy()
            
            # Shuffle the specific feature across the batch dimension
            # to break its relationship with the target
            np.random.shuffle(X_test_corrupted[:, :, i])
            
            X_test_corrupted_tensor = torch.FloatTensor(X_test_corrupted).to(device)
            corrupted_predictions_scaled = model(X_test_corrupted_tensor).cpu().numpy()
            
            dummy_input_corr = np.zeros((len(corrupted_predictions_scaled), len(features)))
            dummy_input_corr[:, target_idx] = corrupted_predictions_scaled.flatten()
            corrupted_predictions_log = scaler.inverse_transform(dummy_input_corr)[:, target_idx]
            corrupted_predictions_real = np.expm1(corrupted_predictions_log)
            
            corrupted_rmse = np.sqrt(mean_squared_error(y_test_real, corrupted_predictions_real))
            
            # The larger the increase in error, the more important the feature
            importance_score = corrupted_rmse - baseline_rmse
            feature_importance[feature_name] = max(0, importance_score) # floor at 0
            
    # Normalize to percentages
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        for k in feature_importance:
            feature_importance[k] = (feature_importance[k] / total_importance) * 100
            
    print("Feature Importance (% contribution to accuracy):")
    for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
        print(f"  {k}: {v:.2f}%")
        
    # Bar Chart for Feature Importance
    plt.figure(figsize=(10, 6))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=False)
    names = [x[0] for x in sorted_features]
    scores = [x[1] for x in sorted_features]
    
    plt.barh(names, scores, color='teal')
    plt.title('Transformer Feature Importance (Permutation)')
    plt.xlabel('Importance Score (%)')
    plt.tight_layout()
    fi_plot_path = os.path.join(output_dir, '8_feature_importance.png')
    plt.savefig(fi_plot_path)
    plt.close()
    
    print(f"Saved Feature Importance plot to {fi_plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_transformer(input_file, output_directory)
