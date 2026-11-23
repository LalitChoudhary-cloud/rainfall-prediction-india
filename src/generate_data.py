import numpy as np
import pandas as pd

def generate_synthetic_weather_data(n_samples=2500, random_state=42):
    """
    Generates realistic weather data modeled after meteorological patterns
    (similar to the Kaggle Australian Weather dataset).
    """
    np.random.seed(random_state)
    
    # Base temperature seasonality
    day_of_year = np.random.randint(1, 366, size=n_samples)
    base_temp = 18 + 8 * np.sin(2 * np.pi * day_of_year / 365)
    
    min_temp = base_temp - np.random.uniform(3, 8, size=n_samples) + np.random.normal(0, 2, n_samples)
    max_temp = base_temp + np.random.uniform(5, 12, size=n_samples) + np.random.normal(0, 2, n_samples)
    
    # Temperature at 9am and 3pm
    temp_9am = min_temp + (max_temp - min_temp) * np.random.uniform(0.2, 0.45, size=n_samples)
    temp_3pm = min_temp + (max_temp - min_temp) * np.random.uniform(0.7, 0.95, size=n_samples)
    
    # Humidity (inversely correlated with temperature, higher on rainy conditions)
    humidity_9am = np.clip(np.random.normal(68, 16, n_samples), 15, 100)
    humidity_3pm = np.clip(humidity_9am - np.random.uniform(10, 30, n_samples) + np.random.normal(0, 8, n_samples), 10, 100)
    
    # Atmospheric pressure (typically 1000 - 1035 hPa; drops when storm systems arrive)
    pressure_9am = np.clip(np.random.normal(1017, 7, n_samples), 990, 1040)
    pressure_diff = np.random.normal(-2.5, 2, n_samples)
    pressure_3pm = np.clip(pressure_9am + pressure_diff, 985, 1038)
    
    # Wind speeds (km/h)
    wind_gust_speed = np.clip(np.random.gamma(shape=4, scale=10, size=n_samples), 10, 120)
    wind_speed_9am = np.clip(wind_gust_speed * np.random.uniform(0.25, 0.55, size=n_samples), 2, 60)
    wind_speed_3pm = np.clip(wind_gust_speed * np.random.uniform(0.35, 0.65, size=n_samples), 2, 70)
    
    # Cloud cover (0 to 8 oktas)
    cloud_9am = np.clip(np.round((humidity_9am / 100) * 8 + np.random.normal(0, 1.5, n_samples)), 0, 8)
    cloud_3pm = np.clip(np.round((humidity_3pm / 100) * 8 + np.random.normal(0, 1.5, n_samples)), 0, 8)
    
    # Sunshine hours (max ~14, lower when clouds/humidity high)
    sunshine = np.clip(13 - (cloud_3pm * 1.3) + np.random.normal(0, 1.2, n_samples), 0, 14.5)
    
    # Evaporation (mm)
    evaporation = np.clip((temp_3pm * 0.2) + (wind_speed_3pm * 0.1) - (humidity_3pm * 0.05) + np.random.normal(2, 1, n_samples), 0.2, 18.0)
    
    # Rainfall today
    rain_today_prob = 1 / (1 + np.exp(-(0.05 * humidity_9am - 0.03 * (pressure_9am - 1015) + 0.3 * cloud_9am - 3.5)))
    rainfall = np.where(np.random.rand(n_samples) < rain_today_prob, np.random.exponential(scale=6.5, size=n_samples), 0.0)
    rainfall = np.round(rainfall, 1)
    rain_today = np.where(rainfall > 1.0, 'Yes', 'No')
    
    # Target: RainTomorrow (Log-odds based on meteorology: high humidity 3pm, dropping pressure, cloud cover, rainfall today, high wind)
    log_odds = (
        0.06 * (humidity_3pm - 50) +
        -0.08 * (pressure_3pm - 1015) +
        0.35 * (cloud_3pm - 4) +
        -0.20 * (sunshine - 7) +
        0.03 * (wind_gust_speed - 40) +
        0.65 * (rainfall > 1.0).astype(float) +
        -0.03 * (temp_3pm - 20) -
        0.9
    )
    prob_rain_tomorrow = 1 / (1 + np.exp(-log_odds))
    rain_tomorrow = np.where(np.random.rand(n_samples) < prob_rain_tomorrow, 'Yes', 'No')
    
    df = pd.DataFrame({
        'MinTemp': np.round(min_temp, 1),
        'MaxTemp': np.round(max_temp, 1),
        'Rainfall': rainfall,
        'Evaporation': np.round(evaporation, 1),
        'Sunshine': np.round(sunshine, 1),
        'WindGustSpeed': np.round(wind_gust_speed, 1),
        'WindSpeed9am': np.round(wind_speed_9am, 1),
        'WindSpeed3pm': np.round(wind_speed_3pm, 1),
        'Humidity9am': np.round(humidity_9am, 1),
        'Humidity3pm': np.round(humidity_3pm, 1),
        'Pressure9am': np.round(pressure_9am, 1),
        'Pressure3pm': np.round(pressure_3pm, 1),
        'Cloud9am': cloud_9am.astype(int),
        'Cloud3pm': cloud_3pm.astype(int),
        'Temp9am': np.round(temp_9am, 1),
        'Temp3pm': np.round(temp_3pm, 1),
        'RainToday': rain_today,
        'RainTomorrow': rain_tomorrow
    })
    
    # Introduce ~1.5% realistic random missing values to practice proper missing value handling
    for col in ['Evaporation', 'Sunshine', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']:
        mask = np.random.rand(n_samples) < 0.02
        df.loc[mask, col] = np.nan
        
    return df

if __name__ == '__main__':
    df = generate_synthetic_weather_data()
    output_path = 'data/weather_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created with {len(df)} rows and {df.shape[1]} columns at {output_path}")
    print("Class distribution for RainTomorrow:")
    print(df['RainTomorrow'].value_counts(normalize=True))

