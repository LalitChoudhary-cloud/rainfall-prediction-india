import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Rainfall Prediction & Analysis in India",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df_state = pd.read_csv('data/cleaned_state_wise_rainfall.csv')
    df_hist = pd.read_csv('data/cleaned_historical_rainfall.csv')
    return df_state, df_hist

try:
    df_state, df_hist = load_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/rain.png", width=120)
st.sidebar.title("Configuration")

states_list = sorted(df_state['STATE'].unique())
selected_state = st.sidebar.selectbox("Select Indian State / Region:", states_list, index=states_list.index("Maharashtra") if "Maharashtra" in states_list else 0)

target_year = st.sidebar.number_input(
    "Target Prediction Year:",
    min_value=1901,
    max_value=2050,
    value=2026,
    step=1
)

model_choice = st.sidebar.selectbox(
    "Select Prediction Algorithm:",
    [
        "Random Forest Regressor (Ensemble)",
        "Historical Mean Baseline (Climatological Normal)",
        "Linear Regression (Trend Line)",
        "Decision Tree Regressor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Domain:** Civil Engineering (Hydrology)  
**Data Source:** India Meteorological Department (IMD)  
**Time Horizon:** 115 Years ($1901–2015$)
""")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<p class="main-header">🌧️ State-Wise Annual Rainfall Analysis & Prediction in India</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An Engineering Hydrology and Machine Learning Decision-Support System</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Interactive Prediction",
    "📊 Exploratory Data Analysis (EDA)",
    "⚖️ Model Benchmark & Evaluation",
    "🏗️ Civil Engineering Significance"
])

# Filter State Data
state_df = df_state[df_state['STATE'] == selected_state].sort_values('YEAR').reset_index(drop=True)

# Chronological Train/Test Split (80% / 20%)
train_df = state_df[state_df['YEAR'] <= 1992]
test_df = state_df[state_df['YEAR'] > 1992]

X_train = train_df[['YEAR']]
y_train = train_df['ANNUAL']
X_test = test_df[['YEAR']]
y_test = test_df['ANNUAL']

# Model Training
baseline_mean = y_train.mean()
baseline_std = y_train.std()

lin_reg = LinearRegression().fit(X_train, y_train)
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42).fit(X_train, y_train)

# =============================================================================
# TAB 1: INTERACTIVE PREDICTION
# =============================================================================
with tab1:
    st.subheader(f"Forecast for {selected_state} in {target_year}")

    # Compute prediction
    if "Historical Mean" in model_choice:
        pred_val = baseline_mean
        model_tag = "Climatological Mean"
    elif "Random Forest" in model_choice:
        pred_val = float(rf_reg.predict(pd.DataFrame({'YEAR': [target_year]}))[0])
        model_tag = "Random Forest Regressor"
    elif "Linear Regression" in model_choice:
        pred_val = float(lin_reg.predict(pd.DataFrame({'YEAR': [target_year]}))[0])
        model_tag = "Linear Regression"
    else:
        pred_val = float(dt_reg.predict(pd.DataFrame({'YEAR': [target_year]}))[0])
        model_tag = "Decision Tree Regressor"

    # Percentage deviation from normal
    pct_diff = ((pred_val - baseline_mean) / baseline_mean) * 100
    if pct_diff > 19:
        category = "Excess Rainfall (+19% to +59%)"
        cat_color = "green"
    elif pct_diff < -19 and pct_diff >= -59:
        category = "Deficient Rainfall (-20% to -59%)"
        cat_color = "orange"
    elif pct_diff < -59:
        category = "Scanty Rainfall (≤ -60%)"
        cat_color = "red"
    else:
        category = "Normal Rainfall (-19% to +19%)"
        cat_color = "blue"

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Annual Rainfall", f"{pred_val:.1f} mm")
    with col2:
        st.metric("Historical Normal (Mean)", f"{baseline_mean:.1f} mm")
    with col3:
        st.metric("Deviation from Normal", f"{pct_diff:+.1f}%")
    with col4:
        st.metric("Empirical Fluctuation (±1σ)", f"±{baseline_std:.1f} mm")

    st.info(f"**Hydrological Classification:** <span style='color:{cat_color}; font-weight:bold;'>{category}</span> (IMD Standard Criteria)", icon="ℹ️")

    # Horizon Warning
    if target_year > 2025:
        st.warning(
            f"⚠️ **Forecast Horizon Warning ({target_year}):** Machine learning regression models on calendar year alone carry high variance for distant horizons. "
            f"For civil engineering infrastructure planning beyond 2025, rely on the **Climatological Baseline Normal ({baseline_mean:.1f} ± {baseline_std:.1f} mm)**.",
            icon="⚠️"
        )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(state_df['YEAR'], state_df['ANNUAL'], label='Historical Records (1901-2015)', color='#1f77b4', alpha=0.6, linewidth=1.2)
    state_df['5yr_MA'] = state_df['ANNUAL'].rolling(5, min_periods=1).mean()
    ax.plot(state_df['YEAR'], state_df['5yr_MA'], label='5-Year Moving Average', color='#0d3b66', linewidth=2.2)
    ax.axhline(baseline_mean, color='red', linestyle='--', label=f'Historical Normal ({baseline_mean:.1f} mm)')

    # Shaded ±1 sigma uncertainty band
    ax.fill_between(
        range(1900, max(2020, target_year + 5)),
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color='gray', alpha=0.15, label='Empirical Confidence Band (±1σ)'
    )

    # Scatter prediction point
    ax.scatter([target_year], [pred_val], color='crimson', s=120, zorder=5, label=f'Prediction ({target_year}: {pred_val:.1f} mm)')
    ax.annotate(
        f"{model_tag}\n{pred_val:.1f} mm",
        (target_year, pred_val),
        xytext=(target_year - 8, pred_val + (baseline_std * 0.4)),
        arrowprops=dict(facecolor='crimson', shrink=0.08, width=1.5, headwidth=6),
        fontweight='bold', fontsize=9
    )

    ax.set_title(f"Annual Precipitation Trend & Forecast: {selected_state}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Rainfall (mm)")
    ax.set_xlim(1900, max(2020, target_year + 3))
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

# =============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab2:
    st.subheader("Climatological Patterns & Historical Trends")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**1. Spatial Disparity Across India**")
        st.image("graphs/graph1_state_wise_average_rainfall.png", caption="Average Annual Rainfall by State (1901-2015)")
    with col_b:
        st.markdown("**2. All-India 115-Year Trend & Extremes**")
        st.image("graphs/graph2_year_wise_rainfall_trend.png", caption="Historic Flood Peak (1961) vs Severe Drought (2009)")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**3. Seasonal Monsoon Concentration**")
        st.image("graphs/graph5_rainfall_heatmap.png", caption="Monthly Heatmap confirming >75% rainfall occurs during June-September")
    with col_d:
        st.markdown("**4. Selected State Comparisons**")
        st.image("graphs/graph3_comparison_selected_states.png", caption="Rainfall Patterns across Wet vs Dry Geographies")

# =============================================================================
# TAB 3: MODEL BENCHMARK & EVALUATION
# =============================================================================
with tab3:
    st.subheader(f"Model Evaluation on Unseen Test Horizon (1993–2015): {selected_state}")

    # Evaluate models on test set
    y_pred_base = np.full_like(y_test, baseline_mean)
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_dt = dt_reg.predict(X_test)
    y_pred_rf = rf_reg.predict(X_test)

    eval_data = [
        {
            "Model Name": "Historical Mean Baseline",
            "MAE (mm)": round(mean_absolute_error(y_test, y_pred_base), 2),
            "RMSE (mm)": round(np.sqrt(mean_squared_error(y_test, y_pred_base)), 2),
            "R² Score": round(r2_score(y_test, y_pred_base), 4),
            "Extrapolation Behavior": "Stable constant benchmark; best hydrological normal"
        },
        {
            "Model Name": "Random Forest Regressor",
            "MAE (mm)": round(mean_absolute_error(y_test, y_pred_rf), 2),
            "RMSE (mm)": round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 2),
            "R² Score": round(r2_score(y_test, y_pred_rf), 4),
            "Extrapolation Behavior": "Best ML performer; 100-tree ensemble dampens step errors"
        },
        {
            "Model Name": "Linear Regression",
            "MAE (mm)": round(mean_absolute_error(y_test, y_pred_lin), 2),
            "RMSE (mm)": round(np.sqrt(mean_squared_error(y_test, y_pred_lin)), 2),
            "R² Score": round(r2_score(y_test, y_pred_lin), 4),
            "Extrapolation Behavior": "Over-penalizes cyclical multi-decadal variations"
        },
        {
            "Model Name": "Decision Tree Regressor",
            "MAE (mm)": round(mean_absolute_error(y_test, y_pred_dt), 2),
            "RMSE (mm)": round(np.sqrt(mean_squared_error(y_test, y_pred_dt)), 2),
            "R² Score": round(r2_score(y_test, y_pred_dt), 4),
            "Extrapolation Behavior": "Fails outside training bounds; produces flat step lines"
        }
    ]

    eval_df = pd.DataFrame(eval_data).sort_values("RMSE (mm)")
    st.table(eval_df)

    st.markdown("### Why did the Historical Baseline beat ML models?")
    st.write("""
    1. **Lack of Atmospheric Drivers:** Calendar year alone does not dictate monsoon strength. Indian monsoon is governed by **ENSO (El Niño Southern Oscillation)**, **Indian Ocean Dipole (IOD)**, and Eurasian snow cover.
    2. **Tree Extrapolation Limit:** Decision trees split feature space into rectangular regions. They cannot predict values outside the range observed in training.
    3. **Ensemble Dampening:** Random Forest performed closest to the baseline because averaging 100 randomized trees smooths out erratic steps.
    """)

    st.image("graphs/graph6_model_comparison.png", caption="Quantitative Comparison of MAE and RMSE across Algorithms")

# =============================================================================
# TAB 4: CIVIL ENGINEERING SIGNIFICANCE
# =============================================================================
with tab4:
    st.subheader("Civil Engineering & Hydrological Applications")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(r"""
        #### 1. Dam & Reservoir Storage (Rippl's Method)
        * In water resources engineering, reservoir capacity is designed using a **Mass Inflow Curve (Rippl's Method)**.
        * Knowing long-term mean annual runoff and standard deviation ($\pm 1\sigma$) allows engineers to guarantee water demand during dry season months without risking dry reservoir dead storage.
        
        #### 2. Spillway Design & Maximum Probable Flood (MPF)
        * Spillways protect dams from catastrophic overtopping.
        * Historic rainfall peaks (like the **1961 all-India flood peak of 1,824.5 mm**) dictate peak runoff routing calculations.
        """)

    with col_e2:
        st.markdown("""
        #### 3. Irrigation & Canal Network Planning
        * Arid states like **Rajasthan (474 mm)** and **Punjab (594 mm)** face heavy monsoon deficits.
        * Reliable precipitation bounds determine canal command area sizing, lined canal discharge, and tube-well groundwater recharge needs.
        
        #### 4. Inter-Basin Water Transfers (National River Linking)
        * The **7-fold spatial disparity** between wet northeastern catchments ($>3,400\text{ mm}$) and dry western/peninsular basins ($<500\text{ mm}$) provides physical hydrological evidence for India's National River Linking Project.
        """)
