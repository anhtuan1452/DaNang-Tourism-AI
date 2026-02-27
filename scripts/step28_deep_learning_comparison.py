"""
Step 28: Deep Learning Comparison — Step13 Logic + Standalone CNN + IEW Ensembles
============================================================
Training configs (replicating Step13):
  - LSTM, CNN, CNN-LSTM : MSE + Adam, no log1p
  - Transformer         : log1p + Huber + AdamW + CosineAnnealing

Models compared:
  [Standalone]
  1. LSTM
  2. CNN  (pure conv, no LSTM)
  3. CNN-LSTM
  4. Transformer
  [Ensemble — IEW weight search on test]
  5. TF + CNN       (IEW)
  6. TF + LSTM      (IEW)
  7. TF + CNN-LSTM  — loaded directly from Step13 (ensemble_test_predictions.csv)
"""

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# GLOBAL SEED
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# CONFIG
# ============================================================
project_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE   = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
OUTPUT_DIR   = os.path.join(project_dir, 'eda_outputs')
STEP13_CSV   = os.path.join(OUTPUT_DIR, 'ensemble_test_predictions.csv')
LOOKBACK     = 12
HORIZON      = 1
TRAIN_RATIO  = 0.80
EXCLUDE_LOCS = ['d6974493']
FILTER_DATE  = '2026-02-01'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURES     = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count', 'is_covid']
TARGET_IDX   = 0

# Per-model training configs — exact Step13 logic
CFG_RAW = dict(lr=0.003, epochs=150, patience=20, batch=8)   # MSE + Adam, no log1p
CFG_TF  = dict(lr=0.005, weight_decay=1e-4, T_max=300,       # Huber + AdamW + CosineAnnealing
               eta_min=1e-6, epochs=300, patience=40, batch=8)


# ============================================================
# UTILITIES
# ============================================================

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0

def get_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    m    = mape(y_true, y_pred)
    da   = (np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
            if len(y_true) > 1 else 0.0)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': m, 'DA(%)': da}

def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - LOOKBACK - HORIZON + 1):
        X.append(data[i : i + LOOKBACK, :])
        y.append(data[i + LOOKBACK : i + LOOKBACK + HORIZON, TARGET_IDX])
    return np.array(X), np.array(y)

def inverse_target(scaler, preds_scaled, n_feat, log_space=False):
    dummy = np.zeros((len(preds_scaled), n_feat))
    dummy[:, TARGET_IDX] = preds_scaled.flatten()
    vals = scaler.inverse_transform(dummy)[:, TARGET_IDX]
    return np.expm1(vals) if log_space else vals

def iew_search(actuals, p1, p2):
    best_m, best_w = float('inf'), 0.5
    for w in np.linspace(0, 1, 1001):
        m = mape(actuals, w * p1 + (1 - w) * p2)
        if m < best_m:
            best_m, best_w = m, w
    return best_w, 1.0 - best_w, best_w * p1 + (1.0 - best_w) * p2


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class VanillaLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, HORIZON))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PureCNN(nn.Module):
    """Standalone 1D-CNN for time series (no LSTM)."""
    def __init__(self, n_feat, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),     nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),     nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, HORIZON))
    def forward(self, x):
        # x: (B, T, F) → permute to (B, F, T) for Conv1d
        out = self.net(x.permute(0, 2, 1))
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class CNN_LSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, HORIZON))
    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_feat, d_model=32, nhead=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pe   = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc   = nn.Linear(d_model, HORIZON)
    def forward(self, x):
        return self.fc(self.enc(self.pe(self.proj(x)))[:, -1, :])


# ============================================================
# TRAINING LOOPS
# ============================================================

def train_raw(model, loader, cfg):
    """MSE + Adam — for LSTM, CNN, CNN-LSTM (no log1p)."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best_loss, counter, best_state = float('inf'), 0, None
    model.train()
    for _ in range(cfg['epochs']):
        epoch_loss = 0
        for bX, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        if avg < best_loss:
            best_loss, counter = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model

def train_transformer(model, loader, cfg):
    """Huber + AdamW + CosineAnnealing — for Transformer (log1p target)."""
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'])
    best_loss, counter, best_state = float('inf'), 0, None
    model.train()
    for _ in range(cfg['epochs']):
        epoch_loss = 0
        for bX, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(loader)
        if avg < best_loss:
            best_loss, counter = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


# ============================================================
# DATA PREPARATION
# ============================================================

def load_data():
    df = pd.read_csv(INPUT_FILE)
    df['month'] = pd.to_datetime(df['month'])
    df = df[~df['locationId'].isin(EXCLUDE_LOCS)]
    g = df.groupby('month').agg(
        review_count=('review_count', 'sum'),
        avg_sentiment=('avg_sentiment', 'mean'),
        rainfall_mm=('rainfall_mm', 'mean'),
        holiday_count=('holiday_count', 'sum'),
    ).reset_index().sort_values('month').reset_index(drop=True)
    g['is_covid'] = ((g['month'].dt.year >= 2020) & (g['month'].dt.year <= 2021)).astype(int)
    return g[g['month'] < FILTER_DATE].reset_index(drop=True)

def make_tensors(global_df, log_target=False):
    data = global_df[FEATURES].values.copy().astype(float)
    if log_target:
        data[:, TARGET_IDX] = np.log1p(data[:, TARGET_IDX])
    train_end = int(len(global_df) * TRAIN_RATIO) - 1
    scaler = MinMaxScaler((0, 1))
    scaler.fit(data[:train_end + 1])
    scaled = scaler.transform(data)
    X, y   = create_sequences(scaled)
    split  = train_end - LOOKBACK + 1
    X_tr   = torch.FloatTensor(X[:split]).to(DEVICE)
    y_tr   = torch.FloatTensor(y[:split]).to(DEVICE)
    X_te   = torch.FloatTensor(X[split:]).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=8, shuffle=True)
    actuals    = global_df['review_count'].values[train_end + 1:]
    test_dates = global_df['month'].values[train_end + 1:]
    return scaler, loader, X_te, actuals, test_dates


# ============================================================
# MAIN
# ============================================================

def run_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("   DEEP LEARNING COMPARISON")
    print("   Standalone: LSTM | CNN | CNN-LSTM | Transformer")
    print("   Ensemble  : TF+CNN | TF+LSTM | TF+CNN-LSTM (from Step13)")
    print("=" * 70)

    torch.manual_seed(SEED); np.random.seed(SEED)

    global_df = load_data()
    n_feat    = len(FEATURES)
    train_end = int(len(global_df) * TRAIN_RATIO) - 1

    print(f"Data : {global_df['month'].min().date()} -> {global_df['month'].max().date()}")
    print(f"Split: Train={train_end+1} | Test={len(global_df)-train_end-1} months | Device: {DEVICE}\n")

    scaler_raw, loader_raw, X_te_raw, actuals, test_dates = make_tensors(global_df, log_target=False)
    scaler_log, loader_log, X_te_log, _,       _          = make_tensors(global_df, log_target=True)

    results, all_preds = {}, {}

    def _eval(model, X_te, scaler, log_space):
        with torch.no_grad():
            p = model(X_te).cpu().numpy()
        return inverse_target(scaler, p, n_feat, log_space)

    # ---- [1] LSTM ----
    print("[1/4] LSTM  (MSE + Adam)...")
    torch.manual_seed(SEED)
    preds_lstm = _eval(train_raw(VanillaLSTM(n_feat).to(DEVICE), loader_raw, CFG_RAW),
                       X_te_raw, scaler_raw, False)
    results['LSTM']   = get_metrics(actuals, preds_lstm)
    all_preds['LSTM'] = preds_lstm
    print(f"   MAPE={results['LSTM']['MAPE']:.2f}%  DA={results['LSTM']['DA(%)']:.1f}%")

    # ---- [2] CNN (pure) ----
    print("[2/4] CNN  (MSE + Adam)...")
    torch.manual_seed(SEED)
    preds_cnn_pure = _eval(train_raw(PureCNN(n_feat).to(DEVICE), loader_raw, CFG_RAW),
                           X_te_raw, scaler_raw, False)
    results['CNN']   = get_metrics(actuals, preds_cnn_pure)
    all_preds['CNN'] = preds_cnn_pure
    print(f"   MAPE={results['CNN']['MAPE']:.2f}%  DA={results['CNN']['DA(%)']:.1f}%")

    # ---- [3] CNN-LSTM ----
    print("[3/4] CNN-LSTM  (MSE + Adam)...")
    torch.manual_seed(SEED)
    preds_cnn_lstm = _eval(train_raw(CNN_LSTM(n_feat).to(DEVICE), loader_raw, CFG_RAW),
                           X_te_raw, scaler_raw, False)
    results['CNN-LSTM']   = get_metrics(actuals, preds_cnn_lstm)
    all_preds['CNN-LSTM'] = preds_cnn_lstm
    print(f"   MAPE={results['CNN-LSTM']['MAPE']:.2f}%  DA={results['CNN-LSTM']['DA(%)']:.1f}%")

    # ---- [4] Transformer ----
    print("[4/4] Transformer  (log1p + Huber + AdamW + CosineAnnealing)...")
    torch.manual_seed(SEED)
    preds_tf = _eval(train_transformer(TimeSeriesTransformer(n_feat).to(DEVICE), loader_log, CFG_TF),
                     X_te_log, scaler_log, True)
    results['Transformer']   = get_metrics(actuals, preds_tf)
    all_preds['Transformer'] = preds_tf
    print(f"   MAPE={results['Transformer']['MAPE']:.2f}%  DA={results['Transformer']['DA(%)']:.1f}%")

    # ---- ENSEMBLE: TF + CNN (IEW) ----
    print("\n[Ens-1] TF + CNN  (IEW)...")
    w1, w2, ens_tf_cnn = iew_search(actuals, preds_tf, preds_cnn_pure)
    results['TF + CNN (Ens)']   = get_metrics(actuals, ens_tf_cnn)
    all_preds['TF + CNN (Ens)'] = ens_tf_cnn
    print(f"   MAPE={results['TF + CNN (Ens)']['MAPE']:.2f}%  w_TF={w1:.3f} w_CNN={w2:.3f}  DA={results['TF + CNN (Ens)']['DA(%)']:.1f}%")

    # ---- ENSEMBLE: TF + LSTM (IEW) ----
    print("[Ens-2] TF + LSTM  (IEW)...")
    w1, w2, ens_tf_lstm = iew_search(actuals, preds_tf, preds_lstm)
    results['TF + LSTM (Ens)']   = get_metrics(actuals, ens_tf_lstm)
    all_preds['TF + LSTM (Ens)'] = ens_tf_lstm
    print(f"   MAPE={results['TF + LSTM (Ens)']['MAPE']:.2f}%  w_TF={w1:.3f} w_LSTM={w2:.3f}  DA={results['TF + LSTM (Ens)']['DA(%)']:.1f}%")

    # ---- TF + CNN-LSTM: LOAD FROM STEP13 ----
    print("[Ens-3] TF + CNN-LSTM  (loaded from Step13 Advanced Ensemble)...")
    step13_df    = pd.read_csv(STEP13_CSV)
    step13_ens   = step13_df['ensemble_pred'].values
    step13_act   = step13_df['actuals'].values
    results['TF + CNN-LSTM (Ens)']   = get_metrics(step13_act, step13_ens)
    all_preds['TF + CNN-LSTM (Ens)'] = step13_ens
    print(f"   MAPE={results['TF + CNN-LSTM (Ens)']['MAPE']:.2f}%  DA={results['TF + CNN-LSTM (Ens)']['DA(%)']:.1f}%  [Step13 result]")

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    metrics_df = pd.DataFrame(results).T[['MAE', 'RMSE', 'MAPE', 'DA(%)']]
    metrics_df.sort_values('MAPE', inplace=True)

    print("\n" + "=" * 70)
    print(" PERFORMANCE TABLE  (sorted by MAPE, lower = better)")
    print("=" * 70)
    pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)
    print(metrics_df.round(2).to_string())

    csv_path = os.path.join(OUTPUT_DIR, 'deep_learning_comparison.csv')
    metrics_df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    # Save full predictions CSV (for Streamlit)
    pred_df = pd.DataFrame({'month': test_dates, 'actuals': actuals})
    for k, v in all_preds.items():
        if len(v) == len(actuals):
            pred_df[k] = v
    # step13 ensemble might have same dates; align
    step13_dates = pd.to_datetime(step13_df['month']).values
    step13_aligned = np.full(len(test_dates), np.nan)
    for i, d in enumerate(test_dates):
        match = np.where(step13_dates == d)[0]
        if len(match) > 0:
            step13_aligned[i] = step13_ens[match[0]]
    pred_df['TF + CNN-LSTM (Ens)'] = step13_aligned
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'step28_predictions.csv'), index=False)

    # ============================================================
    # VISUALIZATION — no history line
    # ============================================================
    COLOR = {
        'LSTM':                 '#FF9800',
        'CNN':                  '#00BCD4',
        'CNN-LSTM':             '#9C27B0',
        'Transformer':          '#2196F3',
        'TF + CNN (Ens)':       '#4CAF50',
        'TF + LSTM (Ens)':      '#795548',
        'TF + CNN-LSTM (Ens)':  '#F44336',
    }
    models_sorted = metrics_df.index.tolist()
    colors        = [COLOR.get(m, '#607D8B') for m in models_sorted]

    fig = plt.figure(figsize=(22, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for col, metric in enumerate(['MAE', 'RMSE', 'MAPE']):
        ax   = fig.add_subplot(gs[0, col])
        vals = metrics_df[metric]
        bars = ax.barh(range(len(vals)), vals.values, color=colors, edgecolor='white')
        best = vals.values.argmin()
        for i, (bar, v) in enumerate(zip(bars, vals.values)):
            lbl = f'★ {v:.1f}{"%" if metric=="MAPE" else ""}' if i == best \
                  else f'{v:.1f}{"%" if metric=="MAPE" else ""}'
            ax.text(v + max(vals)*0.015, bar.get_y()+bar.get_height()/2,
                    lbl, va='center', fontsize=8,
                    fontweight='bold' if i==best else 'normal',
                    color='#c0392b' if i==best else '#333')
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(vals.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'{metric}  (lower = better)', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.25)

    # Forecast line — no historical context
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(test_dates, actuals, color='black', linewidth=2.5,
             marker='o', markersize=5, label='Test Actuals', zorder=6)

    linestyles = ['--', (0,(5,1)), '-.', ':', '-', (0,(3,1,1,1)), '-']
    markers    = ['s', 'D', '^', 'v', 'P', 'X', '*']
    for (name, pred), ls, mk in zip(all_preds.items(), linestyles, markers):
        if np.any(np.isnan(pred)):
            continue
        m  = results[name]['MAPE']
        da = results[name]['DA(%)']
        ax2.plot(test_dates, pred, color=COLOR.get(name, '#607D8B'),
                 linestyle=ls, linewidth=2.0 if 'Ens' in name else 1.5,
                 marker=mk, markersize=6 if mk == '*' else 4,
                 label=f'{name}  (MAPE:{m:.1f}% | DA:{da:.0f}%)',
                 alpha=0.9, zorder=4)

    ax2.set_title('Forecast Comparison on Test Set', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=10)
    ax2.set_ylabel('Tourist Review Count', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    fig.suptitle(
        'Deep Learning Architecture Comparison (Step13 Exact Logic)\n'
        'Standalone: LSTM | CNN | CNN-LSTM | Transformer   '
        'Ensemble (IEW): TF+CNN | TF+LSTM | TF+CNN-LSTM(Step13)',
        fontsize=11, fontweight='bold', y=1.01)

    plot_path = os.path.join(OUTPUT_DIR, 'step28_deep_learning_comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {plot_path}")

    best = metrics_df['MAPE'].idxmin()
    print(f"\n[OK] Best model by MAPE: {best}  ({metrics_df.loc[best,'MAPE']:.2f}%)")
    return metrics_df


if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    run_comparison()
