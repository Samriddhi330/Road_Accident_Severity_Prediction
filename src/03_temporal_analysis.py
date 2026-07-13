import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- STEP 1: LOADING THE CLEANED DATASET ---")
df = pd.read_csv("data/processed/cleaned_accident_data.csv")

print("\n--- STEP 2: CONVERTING TEXT TO REAL TIMESTAMPS ---")
# Convert the plain text date column into a functional date-time layout
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

print("\n--- STEP 3: EXTRACTING TIME SUB-COMPONENTS ---")
# Extract clean whole numbers out of the timestamp column
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['Month'] = df['Start_Time'].dt.month

# 1. Count accidents hour by hour
hour_counts = df['Hour'].value_counts().sort_index()
print("\n[1] ACCIDENT COUNTS TIMELINE BY HOUR:")
print(hour_counts)

# 2. Count accidents day by day
day_counts = df['DayOfWeek'].value_counts().sort_index()
print("\n[2] ACCIDENT COUNTS TIMELINE BY DAY OF WEEK (0=Mon, 6=Sun):")
print(day_counts)


print("\n--- STEP 4: GENERATING VISUALIZATION 1 - HOURLY TREND LINE ---")
plt.figure(figsize=(10, 5))
# Draw a line plot to track rolling hourly accident trends
sns.lineplot(x=hour_counts.index, y=hour_counts.values, marker="o", color="red", linewidth=2.5)
plt.title("Accident Trends Across Hours of the Day")
plt.xlabel("Hour of the Day (24-Hour Clock)")
plt.ylabel("Total Number of Accidents")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("figures/temporal/temporal_hour_trend.png")
print("Saved: figures/temporal/temporal_hour_trend.png")


print("\n--- STEP 5: GENERATING VISUALIZATION 2 - WEEKDAY VS WEEKEND RISK ---")
plt.figure(figsize=(10, 5))
# Draw a bar plot to compare distinct days of the week side by side
sns.barplot(x=day_counts.index, y=day_counts.values, hue=day_counts.index, palette="Blues_d", legend=False)
plt.title("Accident Frequency by Day of the Week")
plt.xlabel("Day of the Week (0 = Monday, 6 = Sunday)")
plt.ylabel("Total Number of Accidents")
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.savefig("figures/temporal/temporal_weekday_comparison.png")
print("Saved: figures/temporal/temporal_weekday_comparison.png")

print("\n--- TEMPORAL ANALYSIS PHASE COMPLETE ---")