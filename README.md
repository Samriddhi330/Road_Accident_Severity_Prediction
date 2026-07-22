# An Explainable and Robust Machine Learning Framework for Road Accident Severity Prediction

## 📌 Problem Statement
Road accidents are one of the major causes of injuries and fatalities worldwide. Predicting accident severity using historical accident data can help authorities identify high-risk situations, improve emergency response, and implement preventive road safety measures.

---

## 🎯 Objectives
* Develop a machine learning model for accident severity prediction.
* Preprocess and analyze accident data.
* Compare Decision Tree and XGBoost models.
* Improve interpretability using SHAP.
* Provide insights for road safety planning.

---

## 🔬 Research Gaps

| Existing Research | Research Gap | Proposed Solution |
| :--- | :--- | :--- |
| Focus on accuracy | Limited explainability | Use SHAP |
| Limited features | Environmental factors ignored | Use weather & road features |
| Few comparisons | Limited evaluation | Compare ML models |
| Prediction only | No feature insights | Feature importance analysis |

---

## 📊 Dataset Details

* **Dataset Name:** US Accidents Dataset
* **Source:** [Kaggle - US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

| Parameter | Details |
| :--- | :--- |
| **Original Size** | Approximately 2.8 million records |
| **Features** | 38 |
| **Target** | Severity |

---

## ⚙️ Data Preprocessing
* Removed unnecessary columns
* Handled missing values
* One-Hot Encoding
* Feature engineering (`Hour`, `Day`, `Month`)
* Binary target creation
* Train-test split

---

## 🤖 Machine Learning Models
* Decision Tree
* XGBoost

---

## 📈 Current Findings
* Dataset preprocessed successfully
* XGBoost achieved ~92% accuracy
* Model evaluated and saved
* Initial SHAP pipeline prepared

---

## 🏆 Best Model & Performance Metrics

**Best Model:** Binary XGBoost Classifier

| Metric | Result |
| :--- | :--- |
| **Accuracy** | **92%** |
| **Precision** | Evaluated |
| **Recall** | Evaluated |
| **F1-Score** | Evaluated |
| **ROC-AUC** | Evaluated |

---

## ⚠️ Challenges
* Large dataset size
* SHAP dependency issues
* DLL / library compatibility
* GitHub synchronization conflicts

---

## 🚀 Next Steps
* Complete SHAP analysis
* Generate feature importance plots
* Integrate execution modules
* Prepare final project report

---

## 📝 Conclusion
The team has completed the literature review, preprocessing, feature engineering, model development, and evaluation. Binary XGBoost was selected as the best performing model. The remaining work focuses on finalizing SHAP explainability and completing the final documentation.
