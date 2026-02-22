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

def create_sequences_with_location(data, target_col_idx, lookback=12, horizon=1):
    X, y, locs = [], [], []
    
    # data is a DataFrame/Array sorted by Location and then by Time
    # Expected structured input: [location_idx, temporal_feature1, temporal_feature2...]
    locations = data[:, 0]
    temporal_features = data[:, 1:]
    
    unique_locs = np.unique(locations)
    
    for loc in unique_locs:
        loc_mask = (locations == loc)
        loc_temporal = temporal_features[loc_mask]
        
        # We need at least lookback + horizon data points per location
        if len(loc_temporal) < lookback + horizon:
            continue
            
        for i in range(len(loc_temporal) - lookback - horizon + 1):
            X.append(loc_temporal[i:(i + lookback), :])
            # The target is the original unshifted target_col_idx in the temporal slice
            y.append(loc_temporal[i + lookback : i + lookback + horizon, target_col_idx])
            locs.append(loc)
            
    return np.array(X), np.array(y), np.array(locs)

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

# --- ULTIMATE GLOBAL ARCHITECTURE: LOCATION EMBEDDING + JOINT LSTM-TRANSFORMER ---
class GlobalJointNetwork(nn.Module):
    def __init__(self, num_locations, embed_dim, input_size, hidden_lstm=64, d_model=32, nhead=4, num_layers_tft=2, output_size=1, dropout=0.2):
        super(GlobalJointNetwork, self).__init__()
        
        # 1. Location Entity Embedding
        self.location_embedding = nn.Embedding(num_embeddings=num_locations, embedding_dim=embed_dim)
        
        # 2. Branch 1: LSTM (Temporal memory + Spatial Embedding)
        # The input to LSTM will be temporal_features concatenated with spatial location embedding
        lstm_input_size = input_size + embed_dim
        self.lstm = nn.LSTM(lstm_input_size, hidden_lstm, num_layers=2, batch_first=True, dropout=dropout)
        
        # 3. Branch 2: Transformer (Attention over Temporal + Spatial Features)
        self.input_projection = nn.Linear(lstm_input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_tft)
        
        # 4. Deep Fusion MLP
        self.fc1 = nn.Linear(hidden_lstm + d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, temporal_x, location_id):
        # temporal_x: (batch, seq_len, input_size)
        # location_id: (batch) -> embedding becomes (batch, embed_dim)
        
        loc_emb = self.location_embedding(location_id) # (batch, embed_dim)
        
        # We need to copy this static location embedding across the entire sequence length 
        # so the LSTM and Transformer know "Which location am I looking at?" at EVERY timestep.
        seq_len = temporal_x.size(1)
        loc_emb_expanded = loc_emb.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq_len, embed_dim)
        
        # Combine temporal and spatial data
        combined_x = torch.cat((temporal_x, loc_emb_expanded), dim=2)
        
        # Branch 1: LSTM processing
        lstm_out, _ = self.lstm(combined_x)
        lstm_final_state = lstm_out[:, -1, :]
        
        # Branch 2: Transformer processing
        x_tft = self.input_projection(combined_x)
        x_tft = self.pos_encoder(x_tft)
        tft_out = self.transformer(x_tft)
        tft_final_state = tft_out[:, -1, :]
        
        # Branch 3: Deep Fusion
        combined_encoded = torch.cat((lstm_final_state, tft_final_state), dim=1)
        
        out = self.fc1(combined_encoded)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def run_global_embedding_model(input_path, output_dir):
    print(f"Loading FULL ECOSYSTEM location-level data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    # Exclude global outlier
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------------------------------------------------------------
    # NON-AGGREGATED STRATEGY: We do NOT groupby month. We keep location panels.
    # -------------------------------------------------------------------------
    
    # 1. Create integer IDs for Locations 
    unique_locations = df['locationId'].unique()
    loc_to_idx = {loc: idx for idx, loc in enumerate(unique_locations)}
    df['loc_idx'] = df['locationId'].map(loc_to_idx)
    num_locations = len(unique_locations)
    print(f"Total Unique Locations for Embedding: {num_locations}")
    
    # 2. Sort by Location and Date to ensure sequence continuity
    df = df.sort_values(by=['loc_idx', 'month']).reset_index(drop=True)
    
    # 3. Target Denoising per Location
    df['target_smoothed'] = df.groupby('loc_idx')['review_count'].rolling(window=2, min_periods=1).mean().reset_index(drop=True)
    df['is_covid'] = ((df['month'].dt.year >= 2020) & (df['month'].dt.year <= 2021)).astype(int)
    
    features = ['target_smoothed', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
    target_idx = features.index('target_smoothed')
    
    # 4. Log Transformation on Target
    df['target_smoothed'] = np.log1p(df['target_smoothed'])
    
    # 5. Train / Test Split
    split_date = pd.to_datetime('2022-12-31')
    train_df = df[df['month'] <= split_date].copy()
    test_df = df[df['month'] > split_date].copy()
    
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    
    # 6. Global Scaler fitted on Train only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[features])
    
    # Apply scaling
    train_features_scaled = scaler.transform(train_df[features])
    test_features_scaled = scaler.transform(test_df[features])
    
    # Combine back with Location IDs for Sequence Generator
    train_matrix = np.hstack((train_df['loc_idx'].values.reshape(-1, 1), train_features_scaled))
    test_matrix = np.hstack((test_df['loc_idx'].values.reshape(-1, 1), test_features_scaled))
    
    lookback = 12
    horizon = 1
    
    X_train_np, y_train_np, locs_train_np = create_sequences_with_location(train_matrix, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    X_test_np, y_test_np, locs_test_np = create_sequences_with_location(test_matrix, target_col_idx=target_idx, lookback=lookback, horizon=horizon)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.FloatTensor(X_train_np).to(device)
    y_train = torch.FloatTensor(y_train_np).to(device)
    locs_train = torch.LongTensor(locs_train_np).to(device)
    
    X_test = torch.FloatTensor(X_test_np).to(device)
    y_test = torch.FloatTensor(y_test_np).to(device)
    locs_test = torch.LongTensor(locs_test_np).to(device)
    
    train_dataset = TensorDataset(X_train, locs_train, y_train)
    # Using larger batch size because we have thousands of location-sequences now
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("\nTraining GLOBAL Joint Network with Location Embedding...")
    torch.manual_seed(42)
    
    embed_dim = 16 # Map 167 dimension location space into a Dense 16-D vector space
    model = GlobalJointNetwork(num_locations=num_locations, embed_dim=embed_dim, input_size=len(features), 
                               hidden_lstm=64, d_model=32, nhead=4, dropout=0.2).to(device)
    
    criterion = nn.SmoothL1Loss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4) 
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 20
    
    model.train()
    for epoch in range(150):
        epoch_loss = 0
        for batch_X, batch_locs, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, batch_locs)
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
    
    # Inferencing...
    with torch.no_grad():
        preds_scaled = model(X_test, locs_test).cpu().numpy()
        
    dummy_input = np.zeros((len(preds_scaled), len(features)))
    dummy_input[:, target_idx] = preds_scaled.flatten()
    
    preds_log = scaler.inverse_transform(dummy_input)[:, target_idx]
    preds_real = np.expm1(preds_log)
    
    # Calculate truth scaling backwards
    y_test_numpy = y_test.cpu().numpy()
    dummy_truth = np.zeros((len(y_test_numpy), len(features)))
    dummy_truth[:, target_idx] = y_test_numpy.flatten()
    actuals_log = scaler.inverse_transform(dummy_truth)[:, target_idx]
    actuals_real = np.expm1(actuals_log)

    metrics = get_metrics(actuals_real, preds_real)
    print(f"\n[SUCCESS] GLOBAL EMBEDDING LSTM-TRANSFORMER METRICS: {metrics}")
    
    # Plotting out a specific famous Location to verify Embedding powers
    target_loc_str = 'd2255351' # Ba Na Hills
    
    if target_loc_str in loc_to_idx:
        target_idx_val = loc_to_idx[target_loc_str]
        
        # Filter test set for Ba Na Hills predictions
        bana_mask = (locs_test_np == target_idx_val)
        bana_preds = preds_real[bana_mask]
        bana_actuals = actuals_real[bana_mask]
        
        test_months_bana = test_df[test_df['loc_idx'] == target_idx_val]['month'].values
        # Due to sliding window lookback, we drop the first 'lookback' months from dates
        test_months_bana = test_months_bana[lookback:]
        
        plt.figure(figsize=(10, 5))
        plt.plot(test_months_bana, bana_actuals, label='Actual Review Volume', color='black', marker='o')
        plt.plot(test_months_bana, bana_preds, label='Global Embedded Deep Learning Forecast', color='red', marker='*')
        plt.title(f'Global Model Individual View: Ba Na Hills Inference')
        plt.xlabel('Month')
        plt.ylabel('Review Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, '19_global_embedding_bana_hills.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"\nSaved Global Model specific-inference plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_global_embedding_model(input_file, output_directory)
