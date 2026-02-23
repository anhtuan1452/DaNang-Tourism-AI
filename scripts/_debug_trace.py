"""
Deep diagnostic: run the EXACT same models & scalers as the ensemble
but trace each prediction step to understand why values keep declining.
"""
import os, sys, pickle, torch, torch.nn as nn
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, 'scripts'))

# ---- Load models ----
models_dir = os.path.join(project_dir, 'models')
with open(os.path.join(models_dir, 'advanced_ensemble_scaler.pkl'), 'rb') as f:
    scaler_cnn = pickle.load(f)

# Load the saved models
from step13_advanced_ensemble import CNN_LSTM, TimeSeriesTransformer

features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
target_idx = 0

df = pd.read_csv(os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv'))
df['month'] = pd.to_datetime(df['month'])
excluded_locations = ['d6974493']
df = df[~df['locationId'].isin(excluded_locations)].copy()
global_df = df.groupby('month').agg({'review_count':'sum','avg_sentiment':'mean','rainfall_mm':'mean','holiday_count':'sum'}).reset_index().sort_values('month')
global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(int)
global_df = global_df[global_df['month'] < '2026-02-01'].copy()
global_df.reset_index(drop=True, inplace=True)

device = torch.device('cpu')
model_cnn = CNN_LSTM(input_size=5, hidden_size=64, num_layers=2, dropout=0.2).to(device)
model_cnn.load_state_dict(torch.load(os.path.join(models_dir, 'advanced_ensemble_cnn.pt'), map_location=device, weights_only=True))
model_cnn.eval()

model_tf = TimeSeriesTransformer(num_features=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1).to(device)
model_tf.load_state_dict(torch.load(os.path.join(models_dir, 'advanced_ensemble_tf.pt'), map_location=device, weights_only=True))
model_tf.eval()

last_sequence_real = global_df[features].values[-12:]
current_sequence = scaler_cnn.transform(last_sequence_real)

print("=== INITIAL SEQUENCE (last 12 months) ===")
for i, row in enumerate(last_sequence_real):
    print(f"  {global_df['month'].iloc[-12+i].strftime('%Y-%m')}: review={row[0]:.0f}  rain={row[2]:.1f}  holiday={row[3]:.0f}")

print("\n=== STEP-BY-STEP AUTOREGRESSIVE TRACE ===")
last_known_metrics = current_sequence[-1].copy()
start_future_month = global_df['month'].max() + pd.DateOffset(months=1)

w_tf, w_cnn = 0.35, 0.65  # approx weights

for step in range(12):
    seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_cnn_scaled = model_cnn(seq_tensor).cpu().numpy()[0, 0]
        pred_tf_scaled = model_tf(seq_tensor).cpu().numpy()[0, 0]
    
    dummy_c = last_known_metrics.copy(); dummy_c[target_idx] = pred_cnn_scaled
    real_c = scaler_cnn.inverse_transform([dummy_c])[0, target_idx]
    
    dummy_t = last_known_metrics.copy(); dummy_t[target_idx] = pred_tf_scaled
    real_t = np.expm1(scaler_cnn.inverse_transform([dummy_t])[0, target_idx])  # approximate
    
    blended = (real_t * w_tf) + (real_c * w_cnn)
    
    future_month = start_future_month + pd.DateOffset(months=step)
    same_last_year = future_month - pd.DateOffset(years=1)
    match = global_df[global_df['month'] == same_last_year]
    
    # Build seasonal row
    if len(match) > 0:
        seasonal_row = match[features].values[0].copy()
        ctx_review = seasonal_row[0]
        seasonal_row[target_idx] = blended
    else:
        ctx_review = "N/A"
        seasonal_row = scaler_cnn.inverse_transform([last_known_metrics])[0].copy()
        seasonal_row[target_idx] = blended
    
    next_scaled = scaler_cnn.transform([seasonal_row])[0]
    
    print(f"  {future_month.strftime('%Y-%m')}: CNN_raw={real_c:.1f}  TF_raw={real_t:.1f}  Blend={blended:.1f}  "
          f"| ctx_review_last_year={ctx_review}  rain_ctx={seasonal_row[2]:.1f}  scaled_rev_in={next_scaled[0]:.4f}")
    
    last_known_metrics = next_scaled.copy()
    current_sequence = np.vstack([current_sequence[1:], next_scaled])
