import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('graphs', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

df_state = pd.read_csv('data/cleaned_state_wise_rainfall.csv')
df_clean = pd.read_csv('data/cleaned_historical_rainfall.csv')

# ------------------------------------------------------------------------------
# GRAPH 1: State-wise Average Annual Rainfall (All States)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
state_means = df_state.groupby('STATE')['ANNUAL'].mean().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(state_means)))

bars = plt.barh(state_means.index, state_means.values, color=colors, edgecolor='black', linewidth=0.5)
plt.axvline(state_means.mean(), color='red', linestyle='--', linewidth=1.5, label=f'National Average ({state_means.mean():.1f} mm)')

plt.title('State-Wise Long-Term Average Annual Rainfall in India (1901 - 2015)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Annual Rainfall (mm)', fontsize=11)
plt.ylabel('State / Region', fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('graphs/graph1_state_wise_average_rainfall.png', dpi=150)
plt.close()

# ------------------------------------------------------------------------------
# GRAPH 2: Year-wise Rainfall Trends (1901 - 2015)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))
nat_trend = df_state.groupby('YEAR')['ANNUAL'].mean().reset_index()
nat_trend['5yr_MA'] = nat_trend['ANNUAL'].rolling(window=5, min_periods=1).mean()

plt.plot(nat_trend['YEAR'], nat_trend['ANNUAL'], color='#1f77b4', alpha=0.45, linewidth=1.2, label='Annual Precipitation')
plt.plot(nat_trend['YEAR'], nat_trend['5yr_MA'], color='#0d3b66', linewidth=2.5, label='5-Year Moving Average (Trend)')
plt.axhline(nat_trend['ANNUAL'].mean(), color='crimson', linestyle='--', linewidth=1.2, label=f'Historical Normal ({nat_trend["ANNUAL"].mean():.1f} mm)')

# Highlight historical extremes: 1961 (Flood) and 2009 (Drought)
plt.scatter([1961], [nat_trend.loc[nat_trend['YEAR']==1961, 'ANNUAL'].values[0]], color='darkgreen', s=70, zorder=5)
plt.annotate('1961: Historic Flood\n(1824.5 mm)', (1961, 1824.5), xytext=(1945, 1780),
             arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1, headwidth=6))

plt.scatter([2009], [nat_trend.loc[nat_trend['YEAR']==2009, 'ANNUAL'].values[0]], color='darkred', s=70, zorder=5)
plt.annotate('2009: Severe Drought\n(1258.9 mm)', (2009, 1258.9), xytext=(1985, 1310),
             arrowprops=dict(facecolor='darkred', shrink=0.08, width=1, headwidth=6))

plt.title('All-India Historical Annual Rainfall Trend & Extremes (1901 - 2015)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Observation Year', fontsize=11)
plt.ylabel('Annual Rainfall (mm)', fontsize=11)
plt.xlim(1900, 2016)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('graphs/graph2_year_wise_rainfall_trend.png', dpi=150)
plt.close()

# ------------------------------------------------------------------------------
# GRAPH 3: Comparison of Rainfall Between Selected States
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))
sample_states = ['Kerala', 'Maharashtra', 'Punjab', 'Rajasthan']
palette = {'Kerala': '#2ca02c', 'Maharashtra': '#1f77b4', 'Punjab': '#ff7f0e', 'Rajasthan': '#d62728'}

for st in sample_states:
    sub = df_state[df_state['STATE'] == st].sort_values('YEAR')
    # Use 3-year smoothed line for clarity
    smoothed = sub['ANNUAL'].rolling(window=3, min_periods=1).mean()
    plt.plot(sub['YEAR'], smoothed, label=f'{st} (Mean: {sub["ANNUAL"].mean():.0f} mm)', color=palette[st], linewidth=2)

plt.title('Multi-State Rainfall Comparison Across Different Hydrological Zones (3-Year Smoothed)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Annual Rainfall (mm)', fontsize=11)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('graphs/graph3_comparison_selected_states.png', dpi=150)
plt.close()

# ------------------------------------------------------------------------------
# GRAPH 4: Highest vs. Lowest Rainfall States
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
top5 = state_means.tail(5)
bot5 = state_means.head(5)
extreme_df = pd.concat([bot5, top5])
bar_colors = ['#d95f02']*5 + ['#1b9e77']*5

bars = plt.bar(extreme_df.index, extreme_df.values, color=bar_colors, edgecolor='black', linewidth=0.6)
plt.title('Hydrological Disparity: Top 5 Driest vs. Top 5 Wettest States in India', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('State / Region', fontsize=11)
plt.ylabel('Average Annual Rainfall (mm)', fontsize=11)
plt.xticks(rotation=30, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.0f} mm', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/graph4_highest_and_lowest_rainfall_states.png', dpi=150)
plt.close()

# ------------------------------------------------------------------------------
# GRAPH 5: Simple Monthly / Seasonal Rainfall Heatmap
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# Calculate monthly average across major representative states
major_states = ['Kerala', 'Goa & Konkan', 'Maharashtra', 'West Bengal', 'Bihar', 'Uttar Pradesh', 'Punjab', 'Rajasthan']
monthly_state = df_clean[df_clean['STATE'].isin(major_states)].groupby('STATE')[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()
# Reorder by descending annual rainfall
monthly_state = monthly_state.loc[major_states]

sns.heatmap(monthly_state, cmap='YlGnBu', annot=True, fmt='.0f', cbar_kws={'label': 'Mean Monthly Rainfall (mm)'}, linewidths=0.5)
plt.title('Monthly Rainfall Distribution Heatmap Across Selected Indian States', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Calendar Month', fontsize=11)
plt.ylabel('State / Region', fontsize=11)
plt.tight_layout()
plt.savefig('graphs/graph5_rainfall_heatmap.png', dpi=150)
plt.close()

print('All 5 graphs successfully generated and saved into graphs/ directory!')

