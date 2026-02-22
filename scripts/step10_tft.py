import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Custom Metrics
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

def run_tft(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['month'] = pd.to_datetime(df['month'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Global Aggregation (matching baselines and LSTM)
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'avg_sentiment': 'mean',
        'rainfall_mm': 'mean',
        'holiday_count': 'sum'
    }).reset_index()
    global_df = global_df.sort_values('month')
    
    # 2. Preparation for PyTorch Forecasting
    # TFT requires an integer time index
    global_df['time_idx'] = np.arange(len(global_df))
    # TFT requires a group ID (even for a single global series)
    global_df['group_id'] = "Global"
    
    # Needs to be float for target normalizer
    global_df['review_count'] = global_df['review_count'].astype(float)
    global_df['holiday_count'] = global_df['holiday_count'].astype(float)
    global_df['rainfall_mm'] = global_df['rainfall_mm'].astype(float)
    
    # Add COVID-19 feature flag (Known covariate)
    global_df['is_covid'] = ((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021)).astype(str)
    
    # Add explicit Month as a categorical feature to help learn seasonality
    global_df['month_cat'] = global_df['month'].dt.month.astype(str)
    
    # 3. Train-Test Split Index
    # We want to forecast the last 24 months (2023-2024)
    split_date = pd.to_datetime('2022-12-31')
    training_cutoff = global_df[global_df['month'] <= split_date]['time_idx'].max()
    
    print(f"Total time steps: {len(global_df)}, Training cutoff: {training_cutoff}")

    # 4. Define TimeSeriesDataSet
    max_prediction_length = 1 # 1-step ahead forecasting to match LSTM
    max_encoder_length = 12 # 12 months lookback
    
    training = TimeSeriesDataSet(
        global_df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="review_count",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["month_cat", "is_covid"],
        time_varying_known_reals=["time_idx", "holiday_count", "rainfall_mm"], # We assume we know holidays and weather (or use average)
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["review_count", "avg_sentiment"], # Past targets and sentiment
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Validation set (Last portion of training data)
    validation = TimeSeriesDataSet.from_dataset(training, global_df, predict=True, stop_randomization=True)
    
    # Prepare Dataloaders
    batch_size = 16
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    # 5. Define TFT Model
    print("Building Temporal Fusion Transformer (TFT)...")
    pl.seed_everything(42)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu", # Assume CPU, modify if GPU available
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        logger=False, # Disable tensorboard for simpler script execution
        enable_checkpointing=False,
        enable_progress_bar=False, # cleaner output
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # QuantileLoss has 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    
    # 6. Training Model
    print("Training TFT... This may take a moment.")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # 7. Evaluation & Forecasting
    print("Evaluating TFT Model on Test Set (2023-2024)...")
    
    # To predict the test set recursively 1-step ahead like LSTM, we can use the model's built-in predict function
    # Create dataset for predictions
    test_data = global_df[global_df.time_idx > training_cutoff - max_encoder_length]
    
    # Predict directly
    # pytorch-forecasting returns predictions in original scale automatically because of target_normalizer
    actuals = []
    predictions = []
    
    model = tft.eval()
    
    # Rolling prediction over the test set
    for i in range(len(global_df) - training_cutoff - 1):
        target_time_idx = training_cutoff + 1 + i
        
        # Take the slice ending before target_time_idx to act as encoder, plus the row for target_time_idx to provide known covariates
        current_data = global_df[(global_df.time_idx <= target_time_idx) & (global_df.time_idx >= target_time_idx - max_encoder_length)].copy()
        
        # We must mask the target value at target_time_idx for prediction
        # though Dataloader predict method handles this for prediction_length rows
        
        # create dataset
        try:
            pred_ds = TimeSeriesDataSet.from_dataset(training, current_data, predict=True, stop_randomization=True)
            pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            
            # Predict
            out = model.predict(pred_loader, mode="prediction", return_x=False)
            
            # Extract point prediction (middle quantile for QuantileLoss, normally 0.5)
            # PyTorch forecasting predict returns raw point estimates
            point_pred = out[0][0].item()
            
            actual = current_data[current_data.time_idx == target_time_idx]['review_count'].values[0]
            
            predictions.append(point_pred)
            actuals.append(actual)
        except Exception as e:
            print(f"Skipping time_idx {target_time_idx} due to error: {e}")

    tft_metrics = get_metrics(actuals, predictions)
    print("\n--- Temporal Fusion Transformer (TFT) Metrics (Test Set 2023-2024) ---")
    print(tft_metrics)
    
    # Compare with Baseline
    baseline_path = os.path.join(output_dir, 'baseline_metrics.csv')
    if os.path.exists(baseline_path):
        baseline_metrics = pd.read_csv(baseline_path, index_col=0)
        baseline_metrics.loc['TFT'] = tft_metrics
        baseline_metrics.to_csv(baseline_path)
        print("\nUpdated Metrics File:")
        print(baseline_metrics)
        
    # 8. Plotting
    test_dates = global_df['month'].iloc[-len(actuals):].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(global_df['month'], global_df['review_count'], label='Actuals (Global)', color='black', alpha=0.5)
    plt.plot(test_dates, actuals, label='Test Actuals', color='blue', marker='o')
    plt.plot(test_dates, predictions, label='TFT Forecast', color='orange', marker='s', linestyle='--')
    
    plt.title('Global Review Count Forecast: Actuals vs TFT')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, '7_tft_forecasts.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved TFT plot to {plot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    output_directory = os.path.join(project_dir, 'eda_outputs')
    
    run_tft(input_file, output_directory)
