import pandas as pd

# 1. Open the massive raw dataset file
print("Step 1: Opening the massive dataset...")
df = pd.read_csv("data/raw/US_Accidents_March23.csv", nrows=100000)

# 2. Save the smaller subset into a new file
print("Step 2: Saving the smaller 100,000 row version...")
df.to_csv("data/raw/us_accidents_sample.csv", index=False)

print("Done! Your small dataset is completely ready.")