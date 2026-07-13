import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- STEP 1: LOADING THE CLEANED DATASET ---")
df = pd.read_csv("data/processed/cleaned_accident_data.csv")

print("\n--- STEP 2: ANALYZING ROAD FEATURES VS ACCIDENT SEVERITY ---")
# Calculate the percentage of severe accidents (Severity 3 and 4) when a Traffic Signal is present
signal_crosstab = pd.crosstab(df['Traffic_Signal'], df['Severity'], normalize='index') * 100
print("Percentage Breakdown of Severity Tiers Based on Traffic Signals:")
print(signal_crosstab)

print("\n--- STEP 3: ANALYZING WEATHER CONDITION TEXT ---")
# Count how many accidents happened during the top 5 weather types
top_weather = df['Weather_Condition'].value_counts().head(5)
print("Top 5 Weather Conditions for Accidents:")
print(top_weather)


print("\n--- STEP 4: GENERATING VISUALIZATION 1 - TRAFFIC SIGNAL IMPACT ---")
plt.figure(figsize=(8, 5))

# Draw a bar chart showing how accident counts change near traffic signals
sns.countplot(data=df, x='Traffic_Signal', hue='Severity', palette='Set1')

plt.title("Accident Severity Counts Near Traffic Signals")
plt.xlabel("Is a Traffic Signal Present? (True / False)")
plt.ylabel("Total Number of Accidents")
plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.savefig("figures/categorical/categorical_signal_impact.png")
print("Saved: figures/categorical/categorical_signal_impact.png")


print("\n--- STEP 5: GENERATING VISUALIZATION 2 - TOP WEATHER CONDITIONS ---")
plt.figure(figsize=(10, 5))

# Draw a clean bar chart showing the most common weather descriptions
sns.barplot(x=top_weather.index, y=top_weather.values, hue=top_weather.index, palette='magma', legend=False)

plt.title("Top 5 Weather Conditions During Accidents")
plt.xlabel("Weather Condition Text")
plt.ylabel("Total Number of Accidents")
plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.savefig("figures/categorical/categorical_weather_counts.png")
print("Saved: figures/categorical/categorical_weather_counts.png")

print("\n--- CATEGORICAL ANALYSIS PHASE COMPLETE ---")