Project Title
An Explainable and Robust Machine Learning Framework for Road Accident Severity Prediction
Problem Statement
Road accidents are one of the major causes of injuries and fatalities worldwide. Predicting accident severity using historical accident data can help authorities identify high-risk situations, improve emergency response, and implement preventive road safety measures.
Objective
•	Develop a machine learning model for accident severity prediction.
•	Preprocess and analyze accident data.
•	Compare Decision Tree and XGBoost models.
•	Improve interpretability using SHAP.
•	Provide insights for road safety planning.
Research Gaps
Existing Research	Research Gap	Proposed Solution
Focus on accuracy	Limited explainability	Use SHAP
Limited features	Environmental factors ignored	Use weather & road features
Few comparisons	Limited evaluation	Compare ML models
Prediction only	No feature insights	Feature importance analysis
Dataset
US Accidents Dataset
Source: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
Dataset Details
Parameter	Details
Original Size	Approximately 2.8 million records
Features	38
Target	Severity
Data Preprocessing
•	Removed unnecessary columns
•	Handled missing values
•	One-Hot Encoding
•	Feature engineering (Hour, Day, Month)
•	Binary target creation
•	Train-test split
Machine Learning Models
•	Decision Tree
•	XGBoost
Current Findings
•	Dataset preprocessed successfully
•	XGBoost achieved ~92% accuracy
•	Model evaluated and saved
•	Initial SHAP pipeline prepared
Challenges
•	Large dataset
•	SHAP dependency issues
•	DLL/library compatibility
•	GitHub conflicts
Next Steps
•	Complete SHAP analysis
•	Generate feature importance plots
•	Integrate modules
•	Prepare final report
Best Model
Binary XGBoost Classifier
Performance Metrics
Metric	Result
Accuracy	92%
Precision	Evaluated
Recall	Evaluated
F1-score	Evaluated
ROC-AUC	Evaluated
Conclusion
The team has completed literature review, preprocessing, feature engineering, model development and evaluation. Binary XGBoost was selected as the best model. The remaining work focuses on SHAP explainability and final documentation.
