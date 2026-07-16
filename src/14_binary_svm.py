import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
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

# 0 = Low severity (original 1 or 2); 1 = High severity (original 3 or 4)
y_train = (train_df["Severity"] >= 3).astype(int)
y_test = (test_df["Severity"] >= 3).astype(int)

# Exclude End_Time because it is only known after the accident ends
X_train = train_df.drop(columns=["Severity", "End_Time"])
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

# LinearSVC is the practical SVM version for this large dataset
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "svm",
            LinearSVC(
                C=1.0,
                max_iter=10000,
                dual="auto",
                random_state=42,
            ),
        ),
    ]
)

print("--- TRAINING BINARY LINEAR SVM ---")
model.fit(X_train, y_train)

print("--- EVALUATING MODEL ---")
predictions = model.predict(X_test)

# LinearSVC does not create probabilities; decision scores are valid for ROC-AUC
decision_scores = model.decision_function(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, decision_scores)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

pd.DataFrame(
    [{
        "Model": "Binary Linear SVM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
    }]
).to_csv("reports/binary_svm_metrics.csv", index=False)

report = classification_report(
    y_test,
    predictions,
    target_names=["Low severity (1-2)", "High severity (3-4)"],
    zero_division=0,
)
with open("reports/binary_svm_classification_report.txt", "w") as file:
    file.write(report)

matrix = confusion_matrix(y_test, predictions)
pd.DataFrame(
    matrix,
    index=["Actual low", "Actual high"],
    columns=["Predicted low", "Predicted high"],
).to_csv("reports/binary_svm_confusion_matrix.csv")

joblib.dump(model, "models/binary_svm.joblib")
print("--- BINARY LINEAR SVM COMPLETE ---")