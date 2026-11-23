import os
import pandas as pd
import numpy as np

os.makedirs('data', exist_ok=True)
raw_historical_path = 'data/rainfall in india 1901-2015.csv'
df_clean = pd.read_csv(raw_historical_path)

months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Subdivision-wise monthly median imputation
for month in months:
    df_clean[month] = df_clean.groupby('SUBDIVISION')[month].transform(
        lambda series: series.fillna(series.median())
    )

# Exact sum for ANNUAL
df_clean['ANNUAL'] = df_clean[months].sum(axis=1)

# Seasons
df_clean['Jan-Feb'] = df_clean['JAN'] + df_clean['FEB']
df_clean['Mar-May'] = df_clean['MAR'] + df_clean['APR'] + df_clean['MAY']
df_clean['Jun-Sep'] = df_clean['JUN'] + df_clean['JUL'] + df_clean['AUG'] + df_clean['SEP']
df_clean['Oct-Dec'] = df_clean['OCT'] + df_clean['NOV'] + df_clean['DEC']

subdivision_to_state = {
    'ANDAMAN & NICOBAR ISLANDS': 'Andaman & Nicobar Islands',
    'ARUNACHAL PRADESH': 'Arunachal Pradesh',
    'ASSAM & MEGHALAYA': 'Assam & Meghalaya',
    'NAGA MANI MIZO TRIPURA': 'Northeastern States (Nagaland, Manipur, Mizoram, Tripura)',
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': 'West Bengal & Sikkim',
    'GANGETIC WEST BENGAL': 'West Bengal',
    'ORISSA': 'Odisha',
    'JHARKHAND': 'Jharkhand',
    'BIHAR': 'Bihar',
    'EAST UTTAR PRADESH': 'Uttar Pradesh',
    'WEST UTTAR PRADESH': 'Uttar Pradesh',
    'UTTARAKHAND': 'Uttarakhand',
    'HARYANA DELHI & CHANDIGARH': 'Haryana & Delhi',
    'PUNJAB': 'Punjab',
    'HIMACHAL PRADESH': 'Himachal Pradesh',
    'JAMMU & KASHMIR': 'Jammu & Kashmir',
    'WEST RAJASTHAN': 'Rajasthan',
    'EAST RAJASTHAN': 'Rajasthan',
    'WEST MADHYA PRADESH': 'Madhya Pradesh',
    'EAST MADHYA PRADESH': 'Madhya Pradesh',
    'GUJARAT REGION': 'Gujarat',
    'SAURASHTRA & KUTCH': 'Gujarat',
    'KONKAN & GOA': 'Goa & Konkan',
    'MADHYA MAHARASHTRA': 'Maharashtra',
    'MATATHWADA': 'Maharashtra',
    'VIDARBHA': 'Maharashtra',
    'CHHATTISGARH': 'Chhattisgarh',
    'COASTAL ANDHRA PRADESH': 'Andhra Pradesh',
    'TELANGANA': 'Telangana',
    'RAYALSEEMA': 'Andhra Pradesh',
    'TAMIL NADU': 'Tamil Nadu',
    'COASTAL KARNATAKA': 'Karnataka',
    'NORTH INTERIOR KARNATAKA': 'Karnataka',
    'SOUTH INTERIOR KARNATAKA': 'Karnataka',
    'KERALA': 'Kerala',
    'LAKSHADWEEP': 'Lakshadweep'
}

df_clean['STATE'] = df_clean['SUBDIVISION'].map(subdivision_to_state)

# State-wise annual aggregation
df_state_annual = df_clean.groupby(['STATE', 'YEAR'])['ANNUAL'].mean().reset_index()
df_state_annual['ANNUAL'] = df_state_annual['ANNUAL'].round(1)

cleaned_subdiv_file = 'data/cleaned_historical_rainfall.csv'
cleaned_state_file = 'data/cleaned_state_wise_rainfall.csv'

df_clean.to_csv(cleaned_subdiv_file, index=False)
df_state_annual.to_csv(cleaned_state_file, index=False)

print(f"Saved: {cleaned_subdiv_file}")
print(f"Saved: {cleaned_state_file}")

