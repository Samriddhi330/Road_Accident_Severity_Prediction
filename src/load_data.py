import pandas as pd

# 1. Open and load the new sample dataset
print("Loading the dataset...")
df = pd.read_csv("data/raw/us_accidents_sample.csv")

# 2. Inspect the total rows and columns
print("\n[1] DATASET SIZE (ROWS, COLUMNS):")
print(df.shape)

# 3. Inspect column names and missing slots
print("\n[2] DATASET STRUCTURE AND DATA TYPES:")
print(df.info())

# 4. Inspect the mathematical spread of numbers
print("\n[3] MATHEMATICAL SUMMARY OF NUMERIC COLUMNS:")
print(df.describe())

# 5. Inspect the target classes we want to predict
print("\n[4] ACCIDENT COUNT PER SEVERITY TIER:")
print(df["Severity"].value_counts())