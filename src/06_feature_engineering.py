import pandas as pd

print("--- STEP 1: LOADING CLEAN DATA ---")
df = pd.read_csv("data/processed/cleaned_accident_data.csv")

print("\n--- STEP 2: EXTRACTING TIME COMPONENTS ---")
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df = df.drop(columns=['Start_Time'])

print("\n--- STEP 3: APPLYING HYBRID ENCODING ---")
# Label Encoding for high-volume text columns
df['Street']            = df['Street'].astype('category').cat.codes
df['City']              = df['City'].astype('category').cat.codes
df['Zipcode']           = df['Zipcode'].astype('category').cat.codes
df['Airport_Code']      = df['Airport_Code'].astype('category').cat.codes
df['Weather_Condition'] = df['Weather_Condition'].astype('category').cat.codes
df['Wind_Direction']    = df['Wind_Direction'].astype('category').cat.codes

# One-Hot Encoding for low-volume text columns
df = pd.get_dummies(df, columns=['Timezone', 'State'], dtype=int)

print("\n--- STEP 4: CONVERTING TRUE/FALSE TO 1/0 ---")
df['Traffic_Signal'] = df['Traffic_Signal'].astype(int)
df['Junction']       = df['Junction'].astype(int)
df['Crossing']       = df['Crossing'].astype(int)
df['Stop']           = df['Stop'].astype(int)
df['Amenity']        = df['Amenity'].astype(int)
df['Bump']           = df['Bump'].astype(int)

# Save this intermediate work for the next file to use
df.to_csv("data/processed/encoded_time_data.csv", index=False)
print("\n--- FEATURE ENGINEERING COMPLETE: Saved encoded_time_data.csv ---")