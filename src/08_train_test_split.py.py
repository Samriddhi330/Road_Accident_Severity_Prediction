import pandas as pd
from sklearn.model_selection import train_test_split

print("--- STEP 1: LOADING FINAL PREPARED DATA ---")
df = pd.read_csv("data/processed/scaled_outlier_treated_data.csv")

print("\n--- STEP 2: CREATING TRAIN-TEST SPLITS ---")
X = df.drop(columns=['Severity'])
y = df['Severity']

# Divide into 80% study data and 20% secret test exam data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- STEP 3: SAVING FINAL DELIVERABLES TO DISK ---")
# Combine the clues and answers back together for clean saving
X_train['Severity'] = y_train
X_test['Severity'] = y_test

X_train.to_csv("data/processed/train_data_final.csv", index=False)
X_test.to_csv("data/processed/test_data_final.csv", index=False)

print(f"Success! Final Training Dataset Shape: {X_train.shape}")
print(f"Success! Final Testing Dataset Shape: {X_test.shape}")
print("\n--- PIPELINE PROCESSED COMPLETELY IN STRUCTURED FILES ---")