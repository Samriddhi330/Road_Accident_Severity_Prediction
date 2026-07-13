import pandas as pd

print("--- STEP 1: OPENING THE SAMPLE DATASET ---")
# Load our working spreadsheet
df = pd.read_csv("data/raw/us_accidents_sample.csv")

print("\n--- STEP 2: REMOVING BROKEN OR EMPTY COLUMNS ---")
# We list columns that are completely blank or missing more than 80% of their data
columns_to_drop = [
    'ID', 'Source', 'Description', 'End_Lat', 'End_Lng', 
    'Precipitation(in)', 'Wind_Chill(F)', 'Country', 'Weather_Timestamp',
    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Sunrise_Sunset'
]
# Delete these columns from our spreadsheet layout
df = df.drop(columns=columns_to_drop)
print(f"Columns left after dropping: {df.shape[1]}")

print("\n--- STEP 3: CLEANING EXACT DUPLICATE ROWS ---")
# Check for exact duplicate rows and remove them if they exist
df = df.drop_duplicates()
print(f"Rows left after duplicate check: {df.shape[0]}")

print("\n--- STEP 4: FILLING GAPS IN NUMBER COLUMNS ---")
# Calculate the middle value (median) for numerical weather data
temp_median = df['Temperature(F)'].median()
humidity_median = df['Humidity(%)'].median()
pressure_median = df['Pressure(in)'].median()
visibility_median = df['Visibility(mi)'].median()
wind_median = df['Wind_Speed(mph)'].median()

# Fill the empty spots in those columns using those middle numbers
df['Temperature(F)'] = df['Temperature(F)'].fillna(temp_median)
df['Humidity(%)'] = df['Humidity(%)'].fillna(humidity_median)
df['Pressure(in)'] = df['Pressure(in)'].fillna(pressure_median)
df['Visibility(mi)'] = df['Visibility(mi)'].fillna(visibility_median)
df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(wind_median)

print("\n--- STEP 5: FILLING GAPS IN TEXT COLUMNS ---")
# Find the most common word (mode) for text columns
weather_mode = df['Weather_Condition'].mode()[0]
wind_dir_mode = df['Wind_Direction'].mode()[0]
city_mode = df['City'].mode()[0]
zip_mode = df['Zipcode'].mode()[0]
tz_mode = df['Timezone'].mode()[0]
airport_mode = df['Airport_Code'].mode()[0]

# Fill the text gaps with the most common word found
df['Weather_Condition'] = df['Weather_Condition'].fillna(weather_mode)
df['Wind_Direction'] = df['Wind_Direction'].fillna(wind_dir_mode)
df['City'] = df['City'].fillna(city_mode)
df['Zipcode'] = df['Zipcode'].fillna(zip_mode)
df['Timezone'] = df['Timezone'].fillna(tz_mode)
df['Airport_Code'] = df['Airport_Code'].fillna(airport_mode)

print("\n--- STEP 6: VERIFYING CLEANED DATA ---")
# Sum up all remaining missing cells to prove it equals 0
print(f"Total blank cells remaining: {df.isnull().sum().sum()}")

# Save our newly cleaned spreadsheet into the processed folder
df.to_csv("data/processed/cleaned_accident_data.csv", index=False)
print("Success! Cleaned data saved to data/processed/cleaned_accident_data.csv")