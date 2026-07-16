import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Create folders where we will save results
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

print("--- LOADING TRAINING AND TEST DATA ---")
train_df = pd.read_csv("data/processed/train_data_final.csv")
test_df = pd.read_csv("data/processed/test_data_final.csv")

# End_Time is removed because it is known after the accident ends
X_train = train_df.drop(columns=["Severity", "End_Time"])
y_train = train_df["Severity"]

X_test = test_df.drop(columns=["Severity", "End_Time"])
y_test = test_df["Severity"]

# County is the only text column; convert it safely into numeric columns
categorical_columns = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("county_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

# class_weight helps the model pay attention to rare Severity 1 and 4 cases
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
        ),
    ]
)

print("--- TRAINING LOGISTIC REGRESSION ---")
model.fit(X_train, y_train)

print("--- EVALUATING MODEL ---")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, average="weighted", zero_division=0
)
roc_auc = roc_auc_score(
    y_test, probabilities, multi_class="ovr", average="weighted"
)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

metrics_df = pd.DataFrame(
    [{
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision (weighted)": precision,
        "Recall (weighted)": recall,
        "F1-score (weighted)": f1,
        "ROC-AUC (weighted OvR)": roc_auc,
    }]
)
metrics_df.to_csv("reports/logistic_regression_metrics.csv", index=False)

report = classification_report(y_test, predictions, zero_division=0)
with open("reports/logistic_regression_classification_report.txt", "w") as file:
    file.write(report)

classes = sorted(y_test.unique())
matrix = confusion_matrix(y_test, predictions, labels=classes)
pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
    "reports/logistic_regression_confusion_matrix.csv"
)

joblib.dump(model, "models/logistic_regression.joblib")
print("--- LOGISTIC REGRESSION COMPLETE ---")