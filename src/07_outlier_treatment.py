import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("--- STEP 1: LOADING ENCODED DATA ---")
df = pd.read_csv("data/processed/encoded_time_data.csv")

print("\n--- STEP 2: DETECTING AND CLIPPING OUTLIERS ---")
# Find the 99th percentile boundary for Wind Speed to manage the 241.7 mph outlier
wind_upper_limit = df['Wind_Speed(mph)'].quantile(0.99)

# Clip everything above that limit down to the boundary line
df['Wind_Speed(mph)'] = np.clip(df['Wind_Speed(mph)'], 0, wind_upper_limit)
print(f"Wind speed successfully capped at: {wind_upper_limit} mph")

print("\n--- STEP 3: SCALING WEATHER NUMBERS ---")
numeric_weather = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
scaler = StandardScaler()

# Normalize the columns so they share a uniform range centered around 0
df[numeric_weather] = scaler.fit_transform(df[numeric_weather])

# Save this fully scaled and treated data for the final split file
df.to_csv("data/processed/scaled_outlier_treated_data.csv", index=False)
print("\n--- OUTLIER & SCALING COMPLETE: Saved scaled_outlier_treated_data.csv ---")