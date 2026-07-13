import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- STEP 1: LOADING THE CLEANED DATASET ---")
df = pd.read_csv("data/processed/cleaned_accident_data.csv")

print("\n--- STEP 2: CALCULATING CORRELATION MATRIX ---")
# Select only the weather columns and our target 'Severity' column
weather_columns = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']

# Calculate how these numbers move together (Correlation)
correlation_matrix = df[weather_columns].corr()
print("Weather Correlation Matrix with Severity:")
print(correlation_matrix['Severity'])


print("\n--- STEP 3: GENERATING VISUALIZATION 1 - CORRELATION HEATMAP ---")
plt.figure(figsize=(8, 6))

# Draw a color grid showing which weather features have the strongest connection
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Weather Correlation Matrix Chart")
plt.savefig("figures/numerical/numerical_weather_correlation.png")
print("Saved: figures/numerical/numerical_weather_correlation.png")


print("\n--- STEP 4: GENERATING VISUALIZATION 2 - TEMPERATURE VS SEVERITY DISTRIBUTION ---")
plt.figure(figsize=(10, 6))

# Draw a box plot to see how temperatures behave at different accident severities
sns.boxplot(data=df, x='Severity', y='Temperature(F)', hue='Severity', palette='Set2', legend=False)

plt.title("Distribution of Temperature Across Accident Severity Tiers")
plt.xlabel("Accident Severity Level (1-4)")
plt.ylabel("Temperature (Fahrenheit)")
plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.savefig("figures/numerical/numerical_temperature_boxplot.png")
print("Saved: figures/numerical/numerical_temperature_boxplot.png")

print("\n--- NUMERICAL ANALYSIS PHASE COMPLETE ---")