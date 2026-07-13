import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- STEP 1: LOADING THE CLEANED DATASET ---")
df = pd.read_csv("data/processed/cleaned_accident_data.csv")

print("\n--- STEP 2: CALCULATING EXACT SPATIAL METRICS ---")
# 1. Get exact value counts for the top states
state_counts = df['State'].value_counts()
print("Complete State Distribution Breakdown:")
print(state_counts)

# 2. Group the data by State and Severity to see the cross-tabulation risk profile
state_severity_matrix = pd.crosstab(df['State'], df['Severity'])
print("\nGeographic Risk Profile (Accident Counts Split by Severity Tier per State):")
print(state_severity_matrix)


print("\n--- STEP 3: GENERATING VISUALIZATION 1 - GEOGRAPHIC COORDINATE HEATMAP ---")
# Create a canvas for our spatial coordinate mapping
plt.figure(figsize=(10, 6))

# Use a scatter plot with high transparency (alpha) to create a natural density heatmap
# X-axis = Longitude (West/East position), Y-axis = Latitude (North/South position)
sns.scatterplot(
    data=df, 
    x='Start_Lng', 
    y='Start_Lat', 
    hue='Severity', 
    palette='viridis', 
    alpha=0.1, 
    edgecolor=None
)

plt.title("Geographic Accident Density Heatmap (Latitude vs Longitude)")
plt.xlabel("Longitude (Horizontal Coordinate)")
plt.ylabel("Latitude (Vertical Coordinate)")
plt.grid(True, linestyle="--", alpha=0.5)

# Save the spatial heatmap visualization
plt.savefig("figures/spatial/spatial_coordinate_heatmap.png")
print("Success! Heatmap saved as 'figures/spatial/spatial_coordinate_heatmap.png'")


print("\n--- STEP 4: GENERATING VISUALIZATION 2 - STACKED GEOGRAPHIC RISK PROFILE ---")
# Create a fresh canvas for our stacked risk comparison chart
plt.figure(figsize=(10, 6))

# Plot the cross-tabulated matrix as a stacked bar chart to show proportions clearly
state_severity_matrix.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#4caf50', '#ffeb3b', '#ff9800', '#f44336'])

plt.title("Accident Severity Composition Across States")
plt.xlabel("US State Code")
plt.ylabel("Total Count of Accidents")
plt.xticks(rotation=0)
plt.legend(title="Traffic Severity Tier (1-4)")
plt.grid(axis='y', linestyle="--", alpha=0.5)

# Save the stacked severity risk profile chart
plt.savefig("figures/spatial/spatial_severity_stacked.png")
print("Success! Stacked risk chart saved as 'figures/spatial/spatial_severity_stacked.png'")