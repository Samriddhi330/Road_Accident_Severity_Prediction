import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

Path("figures").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

metric_files = [
    "reports/binary_logistic_regression_metrics.csv",
    "reports/binary_decision_tree_metrics.csv",
    "reports/binary_random_forest_metrics.csv",
    "reports/binary_xgboost_metrics.csv",
    "reports/binary_svm_metrics.csv",
    "reports/binary_knn_metrics.csv",
    "reports/tuned_decision_tree_metrics.csv",
]

print("--- CREATING PERFORMANCE COMPARISON TABLE ---")
comparison = pd.concat(
    [pd.read_csv(file) for file in metric_files],
    ignore_index=True,
)

metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
comparison = comparison[["Model"] + metric_columns]
comparison = comparison.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
comparison.index = comparison.index + 1
comparison.to_csv("reports/model_performance_comparison.csv", index_label="Rank")

print(comparison.to_string())

sns.set_theme(style="whitegrid")

# Graph 1: All key evaluation metrics
plot_data = comparison.melt(
    id_vars="Model",
    value_vars=metric_columns,
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(15, 8))
sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric")
plt.title("Binary Accident Severity Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/model_performance_comparison.png", dpi=300)
plt.close()

# Graph 2: ROC-AUC ranking
plt.figure(figsize=(12, 7))
roc_order = comparison.sort_values("ROC-AUC", ascending=True)
colors = ["#4C78A8"] * len(roc_order)
colors[-1] = "#E45756"

plt.barh(roc_order["Model"], roc_order["ROC-AUC"], color=colors)
plt.title("ROC-AUC Comparison of Severity Prediction Models")
plt.xlabel("ROC-AUC")
plt.xlim(0, 1.0)

for index, value in enumerate(roc_order["ROC-AUC"]):
    plt.text(value + 0.005, index, f"{value:.4f}", va="center")

plt.tight_layout()
plt.savefig("figures/roc_auc_comparison.png", dpi=300)
plt.close()

# Graph 3: Confusion matrices for every model
matrix_files = {
    "Logistic Regression": "reports/binary_logistic_regression_confusion_matrix.csv",
    "Decision Tree": "reports/binary_decision_tree_confusion_matrix.csv",
    "Random Forest": "reports/binary_random_forest_confusion_matrix.csv",
    "XGBoost": "reports/binary_xgboost_confusion_matrix.csv",
    "Linear SVM": "reports/binary_svm_confusion_matrix.csv",
    "KNN": "reports/binary_knn_confusion_matrix.csv",
    "Tuned Decision Tree": "reports/tuned_decision_tree_confusion_matrix.csv",
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for axis, (model_name, file_name) in zip(axes, matrix_files.items()):
    matrix = pd.read_csv(file_name, index_col=0)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axis,
    )
    axis.set_title(model_name)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

axes[-1].axis("off")
plt.suptitle("Confusion Matrices: Low vs High Accident Severity", fontsize=16)
plt.tight_layout()
plt.savefig("figures/confusion_matrices_all_models.png", dpi=300)
plt.close()

# Best model selection uses ROC-AUC, which measures ranking performance across thresholds
best_model = comparison.iloc[0]

with open("reports/best_model_selection.txt", "w") as file:
    file.write("Best Model Selection\n")
    file.write("====================\n\n")
    file.write(
        f"Selected model: {best_model['Model']}\n"
        f"Selection criterion: highest ROC-AUC\n"
        f"ROC-AUC: {best_model['ROC-AUC']:.4f}\n"
        f"Accuracy: {best_model['Accuracy']:.4f}\n"
        f"F1-score: {best_model['F1-score']:.4f}\n\n"
    )
    file.write(
        "Note: The tuned Decision Tree achieved the highest Accuracy and F1-score, "
        "while XGBoost achieved the highest ROC-AUC. XGBoost is selected as the "
        "overall best model because ROC-AUC evaluates discrimination across all "
        "classification thresholds."
    )

print("\n--- BEST MODEL ---")
print(f"Selected model: {best_model['Model']}")
print(f"ROC-AUC: {best_model['ROC-AUC']:.4f}")
print("\nCreated:")
print("- reports/model_performance_comparison.csv")
print("- reports/best_model_selection.txt")
print("- figures/model_performance_comparison.png")
print("- figures/roc_auc_comparison.png")
print("- figures/confusion_matrices_all_models.png")
print("--- EVALUATION COMPLETE ---")