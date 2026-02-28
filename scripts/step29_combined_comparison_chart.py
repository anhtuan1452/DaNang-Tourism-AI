"""
Step 29: Combined Forecasting Trajectory Chart
===============================================
Merges predictions from:
  - baseline_predictions_timeline.csv (Step 26) → Seasonal Naive, ARIMA, Prophet, RF, SVR, Our Approach
  - step28_predictions.csv              (Step 28) → CNN-LSTM (standalone), Transformer (standalone)

Produces:
  - step26_baseline_comparison.png   (replaces / updates the Fig.2 chart)
  - baseline_comparison_metrics.csv  (updated metrics table with all models)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(project_dir, 'eda_outputs')

BASELINE_CSV = os.path.join(OUTPUT_DIR, 'baseline_predictions_timeline.csv')
STEP28_CSV   = os.path.join(OUTPUT_DIR, 'step28_predictions.csv')

# ── Metrics helper ─────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ── Load ───────────────────────────────────────────────────────────────────────
base_df  = pd.read_csv(BASELINE_CSV);  base_df['month']  = pd.to_datetime(base_df['month'])
step28   = pd.read_csv(STEP28_CSV);    step28['month']   = pd.to_datetime(step28['month'])

# Align on common months
merged = base_df.merge(
    step28[['month', 'CNN-LSTM', 'Transformer']],
    on='month', how='inner'
)
merged = merged.sort_values('month').reset_index(drop=True)

actuals     = merged['Actuals'].values
test_dates  = merged['month'].values

# ── Build ordered model list ───────────────────────────────────────────────────
# Exact order to show in legend (matches paper description)
MODELS = [
    ('Seasonal Naïve',   'Seasonal Naive',       '#9E9E9E', '--',            'o',  1.4),
    ('ARIMA',            'ARIMA',                '#FF9800', '--',            's',  1.4),
    ('Prophet',          'Prophet',              '#4CAF50', '--',            '^',  1.4),
    ('Random Forest',    'Random Forest',        '#9C27B0', '--',            'D',  1.4),
    ('SVR',              'SVR',                  '#795548', '--',            'v',  1.4),
    ('CNN-LSTM',         'CNN-LSTM',             '#00BCD4', '-.',            'P',  1.6),
    ('Transformer',      'Transformer',          '#2196F3', '-.',            'X',  1.6),
    ('Our approach',     'Proposed Advanced Ensemble', '#F44336', '-', '*', 2.4),
]

# ── Compute metrics ────────────────────────────────────────────────────────────
metrics_rows = {}
for label, col, *_ in MODELS:
    if col in merged.columns:
        metrics_rows[label] = get_metrics(actuals, merged[col].values)

metrics_df = pd.DataFrame(metrics_rows).T[['MAE', 'RMSE', 'MAPE']]
metrics_df.sort_values('MAPE', inplace=True)

# Print table
print("\n" + "=" * 65)
print("  COMBINED PERFORMANCE TABLE  (sorted by MAPE, lower = better)")
print("=" * 65)
print(metrics_df.round(3).to_string())

# Save metrics CSV (with step-26 name)
csv_path = os.path.join(OUTPUT_DIR, 'baseline_comparison_metrics.csv')
metrics_df.to_csv(csv_path)
print(f"\nSaved metrics : {csv_path}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5.5))

# Actuals
ax.plot(test_dates, actuals,
        color='black', linewidth=2.5, marker='o', markersize=5,
        label='Actuals', zorder=10)

# Model lines
for label, col, color, ls, mk, lw in MODELS:
    if col not in merged.columns:
        continue
    preds = merged[col].values
    m     = metrics_rows[label]['MAPE']
    is_our = (label == 'Our approach')
    ax.plot(
        test_dates, preds,
        color=color, linestyle=ls, linewidth=lw,
        marker=mk, markersize=(9 if is_our else 4),
        label=f'{label}  (MAPE: {m:.1f}%)',
        alpha=(0.95 if is_our else 0.75),
        zorder=(8 if is_our else 4)
    )

# Formatting
ax.set_title('Fig. 2. Forecasting trajectories of the proposed method and baselines',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Review Count', fontsize=10)
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.92,
          ncol=1, handlelength=2.5)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# X-axis ticks: quarterly
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, 'step26_baseline_comparison.png')
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved chart   : {plot_path}")
print("\n[OK] Done.")

if __name__ == '__main__':
    pass
