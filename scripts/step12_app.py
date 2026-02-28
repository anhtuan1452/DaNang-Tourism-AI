import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import math
import pickle


# --- CONFIGURATION ---
st.set_page_config(
    page_title="Da Nang Tourism Forecasting AI",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit default toolbar + apply premium light background theme
st.markdown("""
    <style>
        /* ── Hide toolbar ── */
        [data-testid="stToolbar"]   { visibility: hidden; height: 0; }
        [data-testid="stDecoration"]{ display: none; }

        /* ── Header bar: match the gradient background ── */
        [data-testid="stHeader"] {
            background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
            border-bottom: 1px solid rgba(30,136,229,0.12);
        }
        /* Ensure the sidebar toggle button inside header stays visible */
        [data-testid="stHeader"] button,
        [data-testid="stSidebarCollapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
        }

        /* ── Main background: soft sky-blue → white gradient ── */
        .stApp {
            background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 50%, #e3f2fd 100%);
            background-attachment: fixed;
        }

        /* ── Sidebar: clean white-blue panel ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #e8f4fd 100%);
            border-right: 1px solid rgba(30,136,229,0.15);
            box-shadow: 2px 0 12px rgba(30,136,229,0.08);
        }

        /* ── Metric cards: glassmorphism light ── */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(30, 136, 229, 0.2);
            border-radius: 12px;
            padding: 12px 16px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 16px rgba(30,136,229,0.08);
        }
        [data-testid="stMetricLabel"] { color: #1565c0 !important; font-size: 0.85rem; font-weight: 600; }
        [data-testid="stMetricValue"] { color: #0d47a1 !important; }

        /* ── Tab bar ── */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.6);
            border-radius: 10px;
            padding: 4px;
            box-shadow: 0 2px 8px rgba(30,136,229,0.1);
        }
        .stTabs [data-baseweb="tab"] { color: #1565c0; font-weight: 500; }
        .stTabs [aria-selected="true"] {
            background: rgba(30,136,229,0.15) !important;
            color: #0d47a1 !important;
            border-radius: 6px;
        }

        /* ── Buttons ── */
        .stButton>button {
            background: linear-gradient(135deg, #1976d2, #1565c0);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 3px 10px rgba(21,101,192,0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1e88e5, #1976d2);
            box-shadow: 0 4px 14px rgba(21,101,192,0.4);
        }

        /* ── Divider ── */
        hr { border-color: rgba(30,136,229,0.2) !important; }
    </style>
""", unsafe_allow_html=True)


# --- MODEL DEFINITIONS FOR INFERENCE ---
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

# --- CACHING & DATA LOADING ---
@st.cache_data
def get_location_names_clean(raw_dir):
    location_mapping = {}
    raw_files = [
        'absa_deepseek_results_merged_backup.csv',
        'absa_deepseek_results_backup.csv'
    ]
    for rf in raw_files:
        raw_path = os.path.join(raw_dir, rf)
        if os.path.exists(raw_path):
            try:
                raw_df = pd.read_csv(raw_path, usecols=['locationId', 'hotelName'], dtype=str)
                raw_df = raw_df.dropna(subset=['locationId', 'hotelName'])
                for _, row in raw_df.drop_duplicates(subset=['locationId']).iterrows():
                    loc_id = row['locationId']
                    if loc_id not in location_mapping:
                        name = str(row['hotelName']).split('-')[0].split(',')[0].strip()
                        name = name.replace("Someone from this business manages the listing.", "").strip()
                        name = name.split("UnclaimedIf you own this business")[0].strip()
                        location_mapping[loc_id] = f"{name} ({loc_id})"
            except Exception as e:
                pass
    return location_mapping

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'processed', 'monthly_location_features.csv')
    raw_dir = os.path.join(project_dir, 'data', 'raw')
    
    df = pd.read_csv(input_file)
    df['month'] = pd.to_datetime(df['month'])
    
    # Exclude anomalies
    excluded_locations = ['d6974493']
    df = df[~df['locationId'].isin(excluded_locations)].copy()
    
    loc_map = get_location_names_clean(raw_dir)
    df['location_name'] = df['locationId'].map(lambda x: loc_map.get(x, f"Location {x}"))
    
    # Scale the original sentiment [-1, 1] up to a standard [1, 5] star rating scale.
    df['scaled_sentiment'] = ((df['avg_sentiment'] + 1) / 2) * 4 + 1
    
    # Calculate a weighted sentiment per row using the true volume of reviews
    df['total_sentiment'] = df['scaled_sentiment'] * df['review_count']
    
    # Global aggregation
    global_df = df.groupby('month').agg({
        'review_count': 'sum',
        'dom_count': 'sum',
        'intl_count': 'sum',
        'total_sentiment': 'sum', # Total weighted sentiment
        'rainfall_mm': 'mean',
        'holiday_count': 'sum',
        'temp_mean': 'mean'
    }).reset_index()
    
    # Calculate the true global weighted average sentiment
    global_df['avg_sentiment'] = np.where(global_df['review_count'] > 0, 
                                          global_df['total_sentiment'] / global_df['review_count'], 
                                          0)
    
    # Overwrite the location-level avg_sentiment for consistency on other tabs
    df['avg_sentiment'] = df['scaled_sentiment']
    
    # Drop temporary variables
    df.drop(columns=['scaled_sentiment', 'total_sentiment'], inplace=True, errors='ignore')
    global_df.drop(columns=['total_sentiment'], inplace=True, errors='ignore')
    
    return df, global_df, loc_map

# Load datasets
try:
    df, global_df, loc_map = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure the data pipeline steps 1-5 have been run.")
    st.stop()


# --- STREAMLIT UI ---
st.title("🏖️ Forecasting Tourism Trends/Demand of Entertainment Attractions in Da Nang Using Online Reviews and Deep Time-Series Models ")
st.markdown("""
Welcome to the interactive Time-Series Forecasting Dashboard for Da Nang's Tourism Sector! 
This application leverages Deep Learning (Transformers/LSTM) to predict future tourist volumes based on historical user-generated reviews, sentiment, weather, and holidays.
""")

# Sidebar Navigation
mode = st.sidebar.selectbox(
    "Select Page",
    [
        "1. 📊 Overview & Data",
        "2. 🌤️ Seasonality Analysis",
        "3. 🏨 Location Deep Dive",
        "4. 🧠 Models & Results",
        "5. 🔮 Future Forecasting",
        "6. ⚙️ AI Pipeline & Source Code",
    ]
)



if mode == "1. 📊 Overview & Data":
    st.header("📊 Da Nang Tourism — Global Overview")
    st.write("Aggregated tourist activity across all Da Nang locations — from historical trends to raw data sources.")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Logged", f"{global_df['review_count'].sum():,.0f}")
    col2.metric("Total Domestic", f"{global_df['dom_count'].sum():,.0f}")
    col3.metric("Total International", f"{global_df['intl_count'].sum():,.0f}")
    col4.metric("Average Sentiment", f"{global_df['avg_sentiment'].mean():.2f} / 5.0")
    
    # Global Trend Chart
    st.subheader("Historical Timeline")
    fig = px.line(global_df, x='month', y='review_count', title="Total Historical Tourist Proxy (Review Count)")
    fig.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="red", opacity=0.2, layer="below", line_width=0, annotation_text="COVID-19 Period")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Tip:** View the Model Leaderboard and detailed forecast charts on page **4. 🧠 Models & Results**.")





elif mode == "3. 🏨 Location Deep Dive":
    st.header("🏨 Location Deep Dive")
    
    # Sort locations by total reviews so top attractions appear first
    location_list_sorted = df.groupby('location_name')['review_count'].sum().sort_values(ascending=False).index.tolist()
    
    selected_loc = st.selectbox("Choose a Tourist Destination:", location_list_sorted)
    
    loc_data = df[df['location_name'] == selected_loc].sort_values('month')
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Lifetime Reviews", f"{loc_data['review_count'].sum():.0f}")
    c2.metric("Peak Month Volume", f"{loc_data['review_count'].max():.0f}")
    c3.metric("Latest Sentiment", f"{loc_data.iloc[-1]['avg_sentiment']:.2f} / 5.0" if len(loc_data)>0 else "N/A")
    
    st.subheader(f"Historical Activity: {selected_loc}")
    fig = px.line(loc_data, x='month', y='review_count', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Weather impact on this location")
    fig2 = px.scatter(loc_data, x='rainfall_mm', y='review_count', color='avg_sentiment', 
                     title="Does Rainfall reduce tourists here?", trendline="ols")
    st.plotly_chart(fig2, use_container_width=True)


elif mode == "2. 🌤️ Seasonality Analysis":
    st.header("🌤️ Seasonality & Tourist Behavior")
    st.write("Excluding COVID years (2020-2021) to find true organic seasonality.")
    
    # Exclude covid
    df_normal = global_df[~((global_df['month'].dt.year >= 2020) & (global_df['month'].dt.year <= 2021))].copy()
    df_normal['month_num'] = df_normal['month'].dt.month
    
    seasonality = df_normal.groupby('month_num').agg({
        'dom_count': 'sum',
        'intl_count': 'sum'
    }).reset_index()
    
    dom_total = seasonality['dom_count'].sum()
    intl_total = seasonality['intl_count'].sum()
    seasonality['dom_pct'] = (seasonality['dom_count'] / dom_total) * 100
    seasonality['intl_pct'] = (seasonality['intl_count'] / intl_total) * 100
    
    # Melt for Plotly
    seas_melt = pd.melt(seasonality, id_vars=['month_num'], value_vars=['dom_pct', 'intl_pct'], 
                        var_name='Cohort', value_name='Percentage')
    seas_melt['Cohort'] = seas_melt['Cohort'].map({'dom_pct': 'Domestic', 'intl_pct': 'International'})
    seas_melt['MonthName'] = seas_melt['month_num'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                                                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
    
    fig = px.bar(seas_melt, x='MonthName', y='Percentage', color='Cohort', barmode='group',
                title="Yearly Distribution of Tourists", color_discrete_sequence=['royalblue', 'crimson'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight:** Domestic tourists heavily favor Summer (June-July). International travelers have a much flatter distribution, but actually peak during the Winter (Dec-Jan) to escape the cold in their home countries.")
    
    st.subheader("AI Feature Importance")
    st.markdown("""
    Based on our Transformer Permutation Analysis, the factors that dictate the fluctuations in Da Nang tourism are:
    1. **Historical Momentum (Review Count T-12)**: 76.7%
    2. **Rainfall (mm)**: 23.3%
    3. Sentiment / Holidays: Negligible on a macro-monthly scale.
    """)

elif mode == "4. 🧠 Models & Results":
    st.header("🧠 Deep Learning Models & Evaluation Results")
    st.write("Full leaderboard of all models evaluated on the 2017–2026 dataset, with interactive forecast visualizations.")
    
    # --- Leaderboard ---
    st.subheader("🏆 Model Leaderboard (MAPE Optimized)")
    try:
        metrics_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eda_outputs', 'baseline_metrics.csv')
        if os.path.exists(metrics_csv):
            metrics_df = pd.read_csv(metrics_csv, index_col=0)
            metrics_df = metrics_df.sort_values(by='MAPE')
            best_model_name = metrics_df.index[0]
            best_mape = metrics_df.iloc[0]['MAPE']
            st.markdown(f"*Evaluated **{len(metrics_df)} models** on the full tourism ecosystem. Best model: **{best_model_name}** with MAPE of **{best_mape:.2f}%**.*")
            st.dataframe(metrics_df.style.highlight_min(subset=['MAPE', 'MAE'], color='lightgreen', axis=0), use_container_width=True)
        else:
            st.warning("Metrics file not found. Run step13_advanced_ensemble.py first.")
    except Exception as e:
        st.error(f"Could not load leaderboard: {e}")

    st.divider()
    
    # --- Architecture Visualizations (moved from page 1) ---
    st.subheader("📈 Forecast Visualizations by Architecture")
    tab_dl, tab0, tab_base, tab_ablation, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔬 Deep Learning Comparison", "Baseline Comparison", "Advanced Ensemble", "Ablation: Impact of Sentiment", 
        "Transformer (Pure)", "Joint LSTM-Transformer", "Prediction Uncertainty (MC Dropout)", 
        "STL-LSTM Hybrid", "CNN-LSTM Hybrid", "BiLSTM-Attention", "Mixed STL-LSTM"
    ])
    eda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eda_outputs')
    
    with tab_dl:
        st.markdown("""
        **Deep Learning Architecture Comparison** — Fair evaluation of all architectures on unified 80/20 train/test split.
        Standalone models use MSE+Adam (LSTM/CNN/CNN-LSTM) or log1p+Huber+AdamW (Transformer).
        TF + CNN-LSTM Ensemble result loaded directly from **Step 13 Advanced Ensemble** (optimal IEW weights).
        """)
        dl_metrics_csv = os.path.join(eda_path, 'deep_learning_comparison.csv')
        dl_preds_csv   = os.path.join(eda_path, 'step28_predictions.csv')

        if os.path.exists(dl_metrics_csv):
            dl_df = pd.read_csv(dl_metrics_csv, index_col=0).sort_values('MAPE')
            best_dl = dl_df.index[0]
            st.success(f"Best model: **{best_dl}** (MAPE {dl_df.loc[best_dl,'MAPE']:.2f}%)")
            st.dataframe(
                dl_df.style.highlight_min(subset=['MAPE','MAE','RMSE'], color='lightgreen', axis=0)
                           .highlight_max(subset=['DA(%)'], color='lightblue', axis=0),
                use_container_width=True
            )
        else:
            st.warning("Run `step28_deep_learning_comparison.py` to generate results.")

        if os.path.exists(dl_preds_csv):
            dp = pd.read_csv(dl_preds_csv)
            dp['month'] = pd.to_datetime(dp['month'])
            model_cols = [c for c in dp.columns if c not in ('month', 'actuals')]

            COLOR_MAP = {
                'LSTM':                '#FF9800',
                'CNN':                 '#00BCD4',
                'CNN-LSTM':            '#9C27B0',
                'Transformer':         '#2196F3',
                'TF + CNN (Ens)':      '#4CAF50',
                'TF + LSTM (Ens)':     '#795548',
                'TF + CNN-LSTM (Ens)': '#F44336',
            }

            fig_dl = go.Figure()
            fig_dl.add_trace(go.Scatter(
                x=dp['month'], y=dp['actuals'],
                mode='lines+markers', name='Test Actuals',
                line=dict(color='black', width=3), marker=dict(size=7)
            ))
            dash_cycle = ['dash','dot','dashdot','longdash','longdashdot','solid','dash']
            for i, col in enumerate(model_cols):
                if dp[col].notna().any():
                    mape_val = ""
                    if os.path.exists(dl_metrics_csv):
                        try:
                            mape_val = f" ({pd.read_csv(dl_metrics_csv, index_col=0).loc[col,'MAPE']:.1f}%)"
                        except: pass
                    fig_dl.add_trace(go.Scatter(
                        x=dp['month'], y=dp[col],
                        mode='lines+markers',
                        name=f"{col}{mape_val}",
                        line=dict(color=COLOR_MAP.get(col,'#607D8B'), dash=dash_cycle[i % len(dash_cycle)],
                                  width=2.5 if 'Ens' in col else 1.8),
                        marker=dict(size=6 if 'Ens' not in col else 8,
                                    symbol='star' if col=='TF + CNN-LSTM (Ens)' else 'circle'),
                        opacity=0.95
                    ))
            fig_dl.update_layout(
                title='Deep Learning Forecast Comparison — Test Set Only',
                xaxis_title='Month', yaxis_title='Tourist Review Count',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
                hovermode='x unified'
            )
            st.plotly_chart(fig_dl, use_container_width=True)
        else:
            st.info("Run `step28_deep_learning_comparison.py` to generate predictions.")

    with tab_base:
        st.markdown("""
        **FULL MODEL COMPARISON** — Combines traditional baselines, standalone deep learning models, and Our proposed approach on a unified test set (80/20 split).
        - **Baselines (dashed):** Seasonal Naïve, ARIMA, Prophet, Random Forest, SVR
        - **DL Standalone (dash-dot):** CNN-LSTM, Transformer
        - **Our approach (solid red ★):** TF + CNN-LSTM Advanced Ensemble
        """)

        # ── Load metrics ──────────────────────────────────────────────────────
        metrics_csv = os.path.join(eda_path, 'baseline_comparison_metrics.csv')
        if os.path.exists(metrics_csv):
            m_df = pd.read_csv(metrics_csv, index_col=0)
            st.dataframe(
                m_df.style.highlight_min(subset=[c for c in ['MAPE','MAE','RMSE'] if c in m_df.columns],
                                         color='lightgreen', axis=0),
                use_container_width=True
            )

        # ── Load & merge prediction CSVs ──────────────────────────────────────
        pred_csv   = os.path.join(eda_path, 'baseline_predictions_timeline.csv')
        step28_csv = os.path.join(eda_path, 'step28_predictions.csv')

        if os.path.exists(pred_csv):
            pred_df = pd.read_csv(pred_csv)
            pred_df['month'] = pd.to_datetime(pred_df['month'])
            if 'Proposed Advanced Ensemble' in pred_df.columns:
                pred_df.rename(columns={'Proposed Advanced Ensemble': 'Our approach'}, inplace=True)

            # Merge CNN-LSTM & Transformer from step28
            if os.path.exists(step28_csv):
                s28 = pd.read_csv(step28_csv)[['month', 'CNN-LSTM', 'Transformer']]
                s28['month'] = pd.to_datetime(s28['month'])
                pred_df = pred_df.merge(s28, on='month', how='inner')

            # ── Color & style config ────────────────────────────────────────
            COLOR_MAP = {
                'Actuals':        ('black',   'solid',     4,  'circle'),
                'Seasonal Naïve': ('#9E9E9E', 'dash',      1.5,'circle'),
                'Seasonal Naive': ('#9E9E9E', 'dash',      1.5,'circle'),
                'ARIMA':          ('#FF9800', 'dash',      1.5,'square'),
                'Prophet':        ('#4CAF50', 'dash',      1.5,'triangle-up'),
                'Random Forest':  ('#9C27B0', 'dash',      1.5,'diamond'),
                'SVR':            ('#795548', 'dash',      1.5,'triangle-down'),
                'CNN-LSTM':       ('#00BCD4', 'dashdot',   2,  'pentagon'),
                'Transformer':    ('#2196F3', 'dashdot',   2,  'cross'),
                'Our approach':   ('#F44336', 'solid',     3.5,'star'),
            }

            # Ordered model display
            DISPLAY_ORDER = [
                'Actuals',
                'Seasonal Naive', 'ARIMA', 'Prophet', 'Random Forest', 'SVR',
                'CNN-LSTM', 'Transformer',
                'Our approach',
            ]

            fig_base = go.Figure()
            for col in DISPLAY_ORDER:
                if col not in pred_df.columns:
                    continue
                clr, dsh, wid, sym = COLOR_MAP.get(col, ('#607D8B','dash',1.5,'circle'))
                is_our = (col == 'Our approach')
                fig_base.add_trace(go.Scatter(
                    x=pred_df['month'],
                    y=pred_df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=clr, dash=dsh, width=wid),
                    marker=dict(size=(10 if is_our else 5), symbol=sym),
                    opacity=(1.0 if col in ('Actuals','Our approach') else 0.75),
                ))

            fig_base.update_layout(
                title="Fig. 2. Forecasting trajectories of the proposed method and baselines",
                xaxis_title="Month",
                yaxis_title="Review Count",
                legend_title="Model/Actuals",
                legend=dict(orientation='v', x=1.01, y=1, font=dict(size=11)),
                hovermode='x unified',
                height=480,
            )
            st.plotly_chart(fig_base, use_container_width=True)

        else:
            img = os.path.join(eda_path, 'step26_baseline_comparison.png')
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.warning("Plot not yet generated. Run step26_compare_baselines.py.")
            
    with tab0:
        st.markdown("**Advanced Multi-Model Ensemble:** Fuses Transformer + CNN-LSTM (60/40 IEW) across the full 2017-2026 dataset.")
        test_pred_csv = os.path.join(eda_path, 'ensemble_test_predictions.csv')
        if os.path.exists(test_pred_csv):
            tp = pd.read_csv(test_pred_csv); tp['month'] = pd.to_datetime(tp['month'])
            train_cut = global_df[global_df['month'] < tp['month'].min()]
            fig_ens = px.line(train_cut, x='month', y='review_count', title='All-Time Tourism Forecast: Advanced Multi-Model Ensemble', labels={'review_count': 'Review Count'})
            fig_ens.update_traces(line=dict(color='gray', width=1.5), opacity=0.5, name='Historical (Train)', showlegend=True)
            fig_ens.add_scatter(x=tp['month'], y=tp['actuals'], mode='lines+markers', name='Test Actuals', line=dict(color='royalblue', width=2), marker=dict(size=6))
            fig_ens.add_scatter(x=tp['month'], y=tp['ensemble_pred'], mode='lines+markers', name='Ensemble Forecast', line=dict(color='red', width=2.5), marker=dict(size=8, symbol='star'))
            fig_ens.add_scatter(x=tp['month'], y=tp['cnn_pred'], mode='lines', name='CNN-LSTM Only', line=dict(color='purple', dash='dot', width=1.5), opacity=0.7)
            fig_ens.add_scatter(x=tp['month'], y=tp['transformer_pred'], mode='lines', name='Transformer Only', line=dict(color='green', dash='dash', width=1.5), opacity=0.7)
            fig_ens.update_layout(xaxis_range=['2022-01-01', tp['month'].max().strftime('%Y-%m-%d')], legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig_ens, use_container_width=True)
        else:
            st.info("Run `step13_advanced_ensemble.py` to generate the predictions CSV.")
                
    with tab_ablation:
        st.markdown("**SENTIMENT ABLATION STUDY:** Measures the exact impact of extracting semantic sentiment from unstructured reviews on the final Advanced Ensemble.")
        
        ablation_metrics_csv = os.path.join(eda_path, 'ablation_sentiment_metrics.csv')
        ablation_preds_csv = os.path.join(eda_path, 'ablation_sentiment_predictions.csv')
        
        if os.path.exists(ablation_metrics_csv) and os.path.exists(ablation_preds_csv):
            # Show Metrics Comparison
            st.markdown("### 📊 Metrics Comparison")
            metrics_df = pd.read_csv(ablation_metrics_csv, index_col=0)
            st.dataframe(metrics_df.style.highlight_min(subset=['MAPE', 'MAE', 'RMSE'], color='lightgreen', axis=0)
                                         .highlight_max(subset=['DA (%)'], color='lightgreen', axis=0), 
                         use_container_width=True)
            
            # Key Insights
            with_sentiment_mape = metrics_df.loc['With Sentiment', 'MAPE']
            without_sentiment_mape = metrics_df.loc['Without Sentiment', 'MAPE']
            improvement = without_sentiment_mape - with_sentiment_mape
            
            st.success(f"🔥 **Insight:** Including NLP Sentiment Analysis drops the MAPE by **{improvement:.2f}%** on the Advanced Ensemble. It provides the hidden micro-level intent needed to accurately track sudden volume spikes.")
            
            # Plot Timeline
            pred_df = pd.read_csv(ablation_preds_csv)
            pred_df['month'] = pd.to_datetime(pred_df['month'])
            
            melted_df = pred_df.melt(id_vars=['month'], var_name='Model', value_name='Forecast')
            
            fig_abla = px.line(melted_df, x='month', y='Forecast', color='Model', 
                               title="Ablation Timeline: Real-world Tracking With vs Without Sentiment",
                               markers=True)
                               
            fig_abla.update_traces(selector=dict(name='Actuals'), line=dict(color='black', width=3))
            fig_abla.update_traces(selector=dict(name='With Sentiment'), line=dict(color='green', width=3, dash='solid'), marker=dict(size=10, symbol='star'))
            fig_abla.update_traces(selector=dict(name='Without Sentiment'), line=dict(color='red', width=2, dash='dash'))
            
            st.plotly_chart(fig_abla, use_container_width=True)
            
        else:
            st.warning("Ablation study files not found. Run step27_ablation_sentiment.py.")
            
    with tab1:
        st.markdown("**Transformer (Pure):** Standalone attention mechanism dynamically finding complex non-linear patterns with no recurrence.")
        test_pred_csv = os.path.join(eda_path, 'ensemble_test_predictions.csv')
        if os.path.exists(test_pred_csv):
            tp = pd.read_csv(test_pred_csv); tp['month'] = pd.to_datetime(tp['month'])
            train_cut = global_df[global_df['month'] < tp['month'].min()]
            fig_tf = px.line(train_cut, x='month', y='review_count', title='Actuals vs Transformer Forecast', labels={'review_count': 'Review Count'})
            fig_tf.update_traces(line=dict(color='gray', width=1.5), opacity=0.5, name='Actuals (Global)', showlegend=True)
            fig_tf.add_scatter(x=tp['month'], y=tp['actuals'], mode='lines+markers', name='Test Actuals', line=dict(color='royalblue', width=2), marker=dict(size=6))
            fig_tf.add_scatter(x=tp['month'], y=tp['transformer_pred'], mode='lines+markers', name='Transformer Forecast', line=dict(color='crimson', dash='dash', width=2), marker=dict(size=6, symbol='triangle-up'))
            st.plotly_chart(fig_tf, use_container_width=True)
        else:
            st.info("Run `step13_advanced_ensemble.py` first.")
    with tab2:
        st.markdown("**Joint End-to-End Network:** Fuses LSTM (Trend) and Transformer (Attention) with Huber Loss to minimize MAPE.")
        img = os.path.join(eda_path, '18_joint_lstm_transformer_forecast.png')
        st.image(img, use_container_width=True) if os.path.exists(img) else st.warning("Plot not yet generated. Run step13_ensemble.py.")
    with tab3:
        st.markdown("**MC Dropout 95% Confidence Intervals:** Neural Network uncertainty estimation showing where the forecast could fluctuate.")
        img = os.path.join(eda_path, '20_deep_learning_prediction_intervals.png')
        st.image(img, use_container_width=True) if os.path.exists(img) else st.warning("Plot not yet generated. Run step22_mc_dropout_intervals.py.")
    with tab4:
        st.markdown("**STL-LSTM Hybrid:** Statistical decomposition (STL) removes Trend/Seasonality before passing residuals to LSTM.")
        img = os.path.join(eda_path, '17_stl_lstm_hybrid_forecast.png')
        st.image(img, use_container_width=True) if os.path.exists(img) else st.warning("Plot not yet generated. Run step19_stl_lstm.py.")
    with tab5:
        st.markdown("**1D CNN-LSTM:** Convolutional layers filter noise first; LSTM then learns temporal patterns.")
        test_pred_csv = os.path.join(eda_path, 'ensemble_test_predictions.csv')
        if os.path.exists(test_pred_csv):
            tp = pd.read_csv(test_pred_csv); tp['month'] = pd.to_datetime(tp['month'])
            train_cut = global_df[global_df['month'] < tp['month'].min()]
            fig_cnn = px.line(train_cut, x='month', y='review_count', title='Actuals vs CNN-LSTM Forecast', labels={'review_count': 'Review Count'})
            fig_cnn.update_traces(line=dict(color='gray', width=1.5), opacity=0.5, name='Actuals (Global)', showlegend=True)
            fig_cnn.add_scatter(x=tp['month'], y=tp['actuals'], mode='lines+markers', name='Test Actuals', line=dict(color='royalblue', width=2), marker=dict(size=6))
            fig_cnn.add_scatter(x=tp['month'], y=tp['cnn_pred'], mode='lines+markers', name='CNN-LSTM Forecast', line=dict(color='purple', width=2), marker=dict(size=6, symbol='square'))
            st.plotly_chart(fig_cnn, use_container_width=True)
        else:
            st.info("Run `step13_advanced_ensemble.py` first.")
    with tab6:
        st.markdown("**BiLSTM-Attention:** Reads data bidirectionally (past + future context) with weighted attention on seasonal spikes.")
        img = os.path.join(eda_path, '15_bilstm_attention_forecast.png')
        st.image(img, use_container_width=True) if os.path.exists(img) else st.warning("Plot not yet generated. Run step17_bilstm_attention.py.")
    with tab7:
        st.markdown("**Mixed STL-LSTM:** End-to-end architecture that ingests decomposed Trend and Seasonality signals directly as input features.")
        img = os.path.join(eda_path, '23_mixed_stl_lstm_forecast.png')
        st.image(img, use_container_width=True) if os.path.exists(img) else st.warning("Plot not yet generated. Run step23_mixed_stl_lstm.py.")

    st.divider()
    # --- Data Viewer (embedded as expander) ---
    with st.expander("🗂️ Raw Data Viewer", expanded=False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(project_dir, 'data')
        data_type = st.radio("Data Type:", ["Raw Data", "Processed Data"], horizontal=True, key="dv_type")
        folder = os.path.join(data_dir, 'raw' if data_type == "Raw Data" else 'processed')
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            if files:
                selected_file = st.selectbox("Select File:", files, key="dv_file")
                file_path = os.path.join(folder, selected_file)
                try:
                    df_preview = pd.read_csv(file_path, nrows=1000)
                    st.write(f"Preview **{selected_file}** (first 1000 rows):")
                    st.dataframe(df_preview, use_container_width=True)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fv:
                            total_lines = sum(1 for _ in fv) - 1
                    except:
                        total_lines = "Unknown"
                    st.info(f"📁 **File Size:** {file_size:.2f} MB | **Total Rows:** {total_lines}")
                except Exception as e:
                    st.error(f"Could not read file {selected_file}. Error: {e}")
            else:
                st.warning(f"No CSV files found in {data_type}.")
        else:
            st.error(f"Directory not found: {folder}.")



elif mode == "6. ⚙️ AI Pipeline & Source Code":
    st.header("⚙️ Project Architecture & Pipeline Steps")
    st.write("This project was built systematically through multiple Python scripts. Below is the chronological execution pipeline:")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(script_dir):
        # Only show requested files for source code viewer
        files = [f for f in os.listdir(script_dir) if f in ['step13_advanced_ensemble.py', 'step26_compare_baselines.py']]
        
        # Sort files based on their step number (e.g. step10 comes after step2)
        def get_step_num(filename):
            try:
                num_part = filename.split('_')[0].replace('step', '')
                return int(num_part)
            except:
                return 999
                
        files = sorted(files, key=get_step_num)
        
        if files:
            for f in files:
                # Format name nicely
                step_name = f.replace('.py', '').replace('_', ' ').title()
                with st.expander(f"- {step_name}"):
                    st.code(f"python scripts/{f}", language="bash")
                    
                    # Read full source code
                    file_path = os.path.join(script_dir, f)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            full_code = file.read()
                            
                        c1, c2 = st.columns([0.8, 0.2])
                        with c1:
                            st.markdown(f"**Source Code Preview ({len(full_code.splitlines())} lines):**")
                        with c2:
                            st.download_button(
                                label="⬇️ Download .py",
                                data=full_code,
                                file_name=f,
                                mime="text/x-python",
                                use_container_width=True
                            )
                            
                        # Show the actual code block fully
                        st.code(full_code, language="python")
                        
                    except Exception as e:
                        st.write(f"Source code preview unavailable. Error: {e}")
        else:
            st.warning("No script files found.")

elif mode == "5. 🔮 Future Forecasting":
    st.header("🔮 Future Forecasting — Interactive Transformer")
    st.markdown("""
    This lab allows you to run **live autoregressive inference** on the Post-COVID data ecosystem (2022-2024). 
    We deployed an automated Hyper-Arena that tested all 4 major Deep Learning architectures (LSTM, CNN-LSTM, BiLSTM-Attention, Transformer) automatically scoring them by MAPE on recent data.
    """)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    meta_path = os.path.join(models_dir, 'best_model_meta.json')
    model_path = os.path.join(models_dir, 'best_post_covid_model.pt')
    scaler_path = os.path.join(models_dir, 'scaler_post_covid_hyper.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(meta_path):
        st.warning("Arena model assets not found. Please run step25_post_covid_hyper_arena.py first.")
    else:
        try:
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            best_arch = meta.get("winning_architecture", "Unknown")
            
            st.success(f"🏆 **Arena Winner Deployed:** {best_arch}")
            
            # Load the scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
            # Initialize and load model dynamically
            features = ['review_count', 'avg_sentiment', 'rainfall_mm', 'holiday_count']
            num_feats = len(features)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if best_arch == "TimeSeriesTransformer":
                model = TimeSeriesTransformer(num_features=num_feats, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=1)
            elif best_arch == "Simple_LSTM":
                model = Simple_LSTM(input_size=num_feats, hidden_size=64, num_layers=2, dropout=0.2)
            elif best_arch == "CNN_LSTM":
                model = CNN_LSTM(input_size=num_feats, hidden_size=64, num_layers=2, dropout=0.2)
            elif best_arch == "BiLSTMAttention":
                model = BiLSTMAttention(input_size=num_feats, hidden_size=64, num_layers=2, dropout=0.3)
            else:
                st.error(f"Unknown architecture: {best_arch}")
                st.stop()
                
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            # Display Metrics
            st.subheader("Model Validation Performance")
            metrics_csv = os.path.join(project_dir, 'eda_outputs', 'post_covid_arena_metrics.csv')
            if os.path.exists(metrics_csv):
                metrics_df = pd.read_csv(metrics_csv, index_col=0)
                st.dataframe(metrics_df.style.highlight_min(subset=['MAPE', 'MAE'], color='lightgreen', axis=0), use_container_width=True)
                st.caption("Lower is better. The winning model is loaded securely into Streamlit Memory.")
                
            st.divider()
            st.subheader("Simulate the Future")
            
            horizon = st.slider("Select Forecast Horizon (Months Ahead):", min_value=1, max_value=12, value=3, step=1)
            
            if st.button("Generate Forecast"):
                with st.spinner("Running deep learning inference..."):
                    # Prepare data
                    post_covid_df = global_df[global_df['month'] >= '2022-01-01'].copy()
                    post_covid_df['review_count'] = np.log1p(post_covid_df['review_count'])
                    
                    # Get the last 12 months from the dataset
                    lookback = 12
                    target_idx = features.index('review_count')
                    
                    if len(post_covid_df) < lookback:
                        st.error(f"Not enough recent data. Need at least {lookback} months.")
                    else:
                        last_known_data = post_covid_df[features].values[-lookback:]
                        scaled_sequence = scaler.transform(last_known_data)
                        
                        # Autoregressive loop
                        current_sequence = scaled_sequence.copy()
                        predictions_scaled = []
                        
                        # Assuming average constant values for future exogenous features
                        avg_sentiment_future = post_covid_df['avg_sentiment'].mean()
                        rainfall_future = post_covid_df['rainfall_mm'].mean()
                        holiday_future = 0 # Can be adjusted
                        
                        for _ in range(horizon):
                            seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
                            with torch.no_grad():
                                next_pred_scaled = model(seq_tensor).cpu().numpy()[0, 0]
                                
                            predictions_scaled.append(next_pred_scaled)
                            
                            # Build next input step (Predicted review, avg sentiment, avg rainfall, avg holiday)
                            next_step_unscaled = np.array([[0.0, avg_sentiment_future, rainfall_future, holiday_future]])
                            next_step_unscaled[0, target_idx] = 0.0 # Placeholder
                            
                            # Scale it
                            next_step_scaled = scaler.transform(next_step_unscaled)[0]
                            # Inject predicted value
                            next_step_scaled[target_idx] = next_pred_scaled
                            
                            # Shift sequence
                            current_sequence = np.vstack((current_sequence[1:], next_step_scaled))
                            
                        # Inverse transform predictions
                        dummy_inputs = np.zeros((horizon, len(features)))
                        dummy_inputs[:, target_idx] = predictions_scaled
                        predictions_log = scaler.inverse_transform(dummy_inputs)[:, target_idx]
                        predictions_real = np.expm1(predictions_log)
                        
                        # --- PLOTTING ---
                        # Historical Data for Plot (Unlogged)
                        hist_plot = post_covid_df.copy()
                        hist_plot['review_count'] = np.expm1(hist_plot['review_count'])
                        
                        last_date = hist_plot['month'].iloc[-1]
                        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
                        
                        future_df = pd.DataFrame({
                            'month': future_dates,
                            'review_count': predictions_real,
                            'Type': 'Forecast'
                        })
                        hist_plot['Type'] = 'Historical'
                        
                        # Connect the lines
                        connection_row = hist_plot.iloc[-1:].copy()
                        connection_row['Type'] = 'Forecast'
                        
                        plot_df = pd.concat([hist_plot[['month', 'review_count', 'Type']], 
                                            connection_row[['month', 'review_count', 'Type']],
                                            future_df])
                        
                        fig = px.line(plot_df, x='month', y='review_count', color='Type', 
                                      color_discrete_map={'Historical': 'black', 'Forecast': 'purple'},
                                      title="Live Autoregressive Transformer Forecast",
                                      markers=True)
                                      
                        # Highlight future zone
                        fig.add_vrect(x0=last_date, x1=future_dates[-1], fillcolor="purple", opacity=0.1, line_width=0, annotation_text="Forecast Horizon")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Provide a data table
                        st.write("### Predicted Timeline")
                        future_df['review_count'] = future_df['review_count'].astype(int)
                        future_df.rename(columns={'month': 'Month', 'review_count': 'Estimated Tourist Volume'}, inplace=True)
                        st.dataframe(future_df[['Month', 'Estimated Tourist Volume']], hide_index=True)
                        
                        st.success("Successfully extrapolated temporal logic into the future.")
                        
            # --- NEW SECTION: Advanced Ensemble 12-Month Static Forecast ---
            st.divider()
            st.subheader("🌟 The Ultimate 12-Month Future Trajectory (Advanced Multi-Model Ensemble)")
            st.markdown("""
            This section displays the **pre-calculated forecast** from our highest-performing architecture: 
            the **Advanced Multi-Model Ensemble** (Super-Transformer + CNN-LSTM). 
            It leverages the full 2017-2026 dataset to project the next year of tourism.
            """)
            
            future_csv_path = os.path.join(project_dir, 'eda_outputs', '12_month_future_forecast.csv')
            if os.path.exists(future_csv_path):
                future_ensemble_df = pd.read_csv(future_csv_path)
                future_ensemble_df['month'] = pd.to_datetime(future_ensemble_df['month'])
                
                # Slider to choose how many months to show
                max_months = len(future_ensemble_df)
                horizon_ens = st.slider("Hiển thị bao nhiêu tháng tương lai:", min_value=1, max_value=max_months, value=max_months, step=1, key="ens_horizon_slider")
                
                future_filtered = future_ensemble_df.iloc[:horizon_ens].copy()
                
                # Plot the Ensemble Future
                hist_plot = global_df.copy()
                # Exclude Feb 2026 — incomplete month (only ~75 reviews) causes a misleading dip
                hist_plot = hist_plot[hist_plot['month'] < '2026-02-01']
                fig = px.line(hist_plot[hist_plot['month'] >= '2024-01-01'], x='month', y='review_count', title=f"12-Month Strategic Blueprint — Next {horizon_ens} Months (Advanced Ensemble)")
                fig.update_traces(line=dict(color='black', width=2), name='Historical (2024–2026)', showlegend=True)
                
                # Connect history to forecast with a bridge point
                last_hist_row = hist_plot.iloc[-1:]
                bridge = pd.DataFrame({'month': [last_hist_row['month'].values[0]], 'forecasted_review_count': [last_hist_row['review_count'].values[0]]})
                bridge_and_forecast = pd.concat([bridge, future_filtered[['month', 'forecasted_review_count']]])
                
                fig.add_scatter(x=bridge_and_forecast['month'], y=bridge_and_forecast['forecasted_review_count'], mode='lines+markers',
                                name=f'Ensemble Forecast (Feb 2026+)', line=dict(color='orange', width=4, dash='dot'),
                                marker=dict(size=8, symbol='diamond'))
                fig.add_vrect(x0=bridge_and_forecast['month'].iloc[0], x1=future_filtered['month'].max() + pd.DateOffset(months=1), 
                              fillcolor="orange", opacity=0.05, line_width=0, annotation_text=f"Dự báo {horizon_ens} tháng")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display Table with Trend Icons
                st.markdown("### 📋 Bảng dự báo chi tiết")
                
                display_df = future_filtered.copy()
                display_df['forecasted_review_count'] = display_df['forecasted_review_count'].round(0).astype(int)
                display_df.rename(columns={'month': 'Tháng', 'forecasted_review_count': 'Lượt Review Dự Báo (Proxy)', 'Trend': 'Xu Hướng'}, inplace=True)
                
                st.dataframe(display_df.style.highlight_max(subset=['Lượt Review Dự Báo (Proxy)'], color='lightgreen', axis=0)
                            .highlight_min(subset=['Lượt Review Dự Báo (Proxy)'], color='salmon', axis=0), 
                            use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có file dự báo. Vui lòng chạy `step13_advanced_ensemble.py` trước.")


        except Exception as e:
            st.error(f"Error initializing interactive forecasting module: {e}")

