print("ANALYSIS FILE STARTED")
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# Create output folders
# -------------------------------------------------------

os.makedirs("reports", exist_ok=True)
os.makedirs("reports/plots", exist_ok=True)

print("="*60)
print("ROAD ACCIDENT SEVERITY ANALYSIS")
print("="*60)

# -------------------------------------------------------
# Load trained model and test data
# -------------------------------------------------------

print("\nLoading model...")

model = joblib.load("models/binary_xgboost.joblib")

print("Loading test dataset...")

test_df = pd.read_csv("data/processed/test_data_final.csv")

# Create X_test exactly like in training
X_test = test_df.drop(columns=["Severity", "End_Time"])

print("Model loaded successfully.")
print("Test dataset loaded successfully.")
# -------------------------------------------------------
# Extract pipeline components
# -------------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["xgboost"]

print("Pipeline extracted.")

# -------------------------------------------------------
# Transform test data
# -------------------------------------------------------

print("\nTransforming data...")

feature_names = preprocessor.get_feature_names_out()

X_test_processed = preprocessor.transform(X_test)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

print("Shape:", X_test_processed.shape)

print("\nGenerating XGBoost Feature Importance...")

importance = xgb_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(12,8))

sns.barplot(
    data=feature_importance.head(20),
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Important Features (XGBoost)")
plt.tight_layout()

plt.savefig(
    "reports/plots/xgboost_feature_importance.png",
    dpi=300
)

plt.close()

feature_importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("Feature importance generated.")
# -------------------------------------------------------
# FEATURE IMPORTANCE TABLE
# -------------------------------------------------------

print("\nCalculating Feature Importance...")


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("\nTop 20 Important Features")

print(feature_importance.head(20))

# -------------------------------------------------------
# LOAD ORIGINAL TEST DATA
# -------------------------------------------------------

df = pd.read_csv("data/processed/test_data_final.csv")

# -------------------------------------------------------
# HOUR-WISE ANALYSIS
# -------------------------------------------------------

print("\nCreating Hour-wise Severity Graph...")

plt.figure(figsize=(12,6))

hourly = (
    df.groupby("Hour")["Severity"]
      .mean()
      .reset_index()
)

sns.lineplot(
    data=hourly,
    x="Hour",
    y="Severity",
    marker="o"
)

plt.title("Average Accident Severity by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Severity")

plt.grid(True)

plt.savefig(
    "reports/plots/hourwise_severity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: hourwise_severity.png")

# -------------------------------------------------------
# DAY-WISE ANALYSIS
# -------------------------------------------------------

print("\nCreating Day-wise Analysis...")

day_names = {
    0:"Monday",
    1:"Tuesday",
    2:"Wednesday",
    3:"Thursday",
    4:"Friday",
    5:"Saturday",
    6:"Sunday"
}

df["DayName"] = df["DayOfWeek"].map(day_names)

order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily = (
    df.groupby("DayName")["Severity"]
      .mean()
      .reindex(order)
      .reset_index()
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=daily,
    x="DayName",
    y="Severity"
)

plt.xticks(rotation=30)

plt.title("Average Severity by Day")

plt.savefig(
    "reports/plots/daywise_severity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: daywise_severity.png")

# -------------------------------------------------------
# MONTH-WISE ANALYSIS
# -------------------------------------------------------

print("\nCreating Month-wise Trend...")

months = {
    1:"Jan",
    2:"Feb",
    3:"Mar",
    4:"Apr",
    5:"May",
    6:"Jun",
    7:"Jul",
    8:"Aug",
    9:"Sep",
    10:"Oct",
    11:"Nov",
    12:"Dec"
}

df["MonthName"] = df["Month"].map(months)

month_order = [
    "Jan","Feb","Mar","Apr",
    "May","Jun","Jul","Aug",
    "Sep","Oct","Nov","Dec"
]

monthly = (
    df.groupby("MonthName")["Severity"]
      .mean()
      .reindex(month_order)
      .reset_index()
)

plt.figure(figsize=(12,6))

sns.lineplot(
    data=monthly,
    x="MonthName",
    y="Severity",
    marker="o"
)

plt.title("Average Severity by Month")

plt.grid(True)

plt.savefig(
    "reports/plots/monthwise_severity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: monthwise_severity.png")

# -------------------------------------------------------
# WEATHER ANALYSIS
# -------------------------------------------------------

print("\nCreating Weather Analysis...")

try:

    top_weather = (
        df["Weather_Condition"]
        .value_counts()
        .head(10)
        .index
    )

    weather_df = df[df["Weather_Condition"].isin(top_weather)]

    plt.figure(figsize=(12,7))

    sns.countplot(
        data=weather_df,
        y="Weather_Condition",
        hue="Severity"
    )

    plt.title("Top Weather Conditions vs Severity")

    plt.tight_layout()

    plt.savefig(
        "reports/plots/weather_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: weather_analysis.png")

except Exception as e:

    print("Weather analysis skipped:", e)

# -------------------------------------------------------
# HOTSPOT ANALYSIS
# -------------------------------------------------------

print("\nCreating Hotspot Analysis...")

try:

    top_cities = (
        df["City"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        x=top_cities.values,
        y=top_cities.index
    )

    plt.xlabel("Number of Accidents")
    plt.ylabel("City")
    plt.title("Top 10 Accident Hotspots")

    plt.tight_layout()

    plt.savefig(
        "reports/plots/hotspot_cities.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: hotspot_cities.png")

except Exception as e:

    print("Hotspot analysis skipped:", e)

# -------------------------------------------------------
# SAVE TOP 20 IMPORTANT FEATURES
# -------------------------------------------------------

feature_importance.head(20).to_csv(
    "reports/top20_features.csv",
    index=False
)

# -------------------------------------------------------
# PRINT FINAL INSIGHTS
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL INSIGHTS")
print("=" * 60)

print("\nTop 10 Important Features:\n")

print(feature_importance.head(10))

print("\nHighest Average Severity Hour:")

print(
    hourly.loc[
        hourly["Severity"].idxmax()
    ]
)

print("\nHighest Average Severity Day:")

print(
    daily.loc[
        daily["Severity"].idxmax()
    ]
)

print("\nHighest Average Severity Month:")

print(
    monthly.loc[
        monthly["Severity"].idxmax()
    ]
)

print("\nTop Accident Hotspot Cities:")

try:
    print(top_cities)
except:
    pass

print("\n")
print("=" * 60)
print("ALL REPORTS GENERATED SUCCESSFULLY")
print("=" * 60)

print("\nSaved Files:")

print("""
reports/
│
├── feature_importance.csv
├── top20_features.csv
│
└── plots/
      ├── shap_summary.png
      ├── shap_bar.png
      ├── shap_waterfall.png
      ├── hourwise_severity.png
      ├── daywise_severity.png
      ├── monthwise_severity.png
      ├── weather_analysis.png
      └── hotspot_cities.png
""")


<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> b7f9eef (Added accident severity analysis and feature importance)
=======

>>>>>>> b486741 (Add analysis.py)
