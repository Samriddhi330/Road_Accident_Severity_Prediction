import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

print("--- LOADING DATA ---")
train_df = pd.read_csv("data/processed/train_data_final.csv")
test_df = pd.read_csv("data/processed/test_data_final.csv")

# A stratified sample keeps the same low/high severity proportions
knn_train_df, _ = train_test_split(
    train_df,
    train_size=30000,
    stratify=(train_df["Severity"] >= 3),
    random_state=42,
)

# 0 = Low severity (original 1 or 2); 1 = High severity (original 3 or 4)
y_train = (knn_train_df["Severity"] >= 3).astype(int)
y_test = (test_df["Severity"] >= 3).astype(int)

# Exclude End_Time because it is known only after the accident ends
X_train = knn_train_df.drop(columns=["Severity", "End_Time"])
X_test = test_df.drop(columns=["Severity", "End_Time"])

categorical_columns = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_columns = [
    column for column in X_train.columns
    if column not in categorical_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "county",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns,
        ),
        ("numeric", StandardScaler(), numerical_columns),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                n_jobs=-1,
            ),
        ),
    ]
)

print("--- TRAINING BINARY KNN ON 30,000 STRATIFIED ROWS ---")
model.fit(X_train, y_train)

print("--- EVALUATING MODEL ---")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

pd.DataFrame(
    [{
        "Model": "Binary KNN (30,000-row stratified training sample)",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
    }]
).to_csv("reports/binary_knn_metrics.csv", index=False)

report = classification_report(
    y_test,
    predictions,
    target_names=["Low severity (1-2)", "High severity (3-4)"],
    zero_division=0,
)
with open("reports/binary_knn_classification_report.txt", "w") as file:
    file.write(report)

matrix = confusion_matrix(y_test, predictions)
pd.DataFrame(
    matrix,
    index=["Actual low", "Actual high"],
    columns=["Predicted low", "Predicted high"],
).to_csv("reports/binary_knn_confusion_matrix.csv")

joblib.dump(model, "models/binary_knn.joblib")
print("--- BINARY KNN COMPLETE ---")