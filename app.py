
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Road Accident Severity Prediction")
st.write("An Explainable and Robust Machine Learning Framework for Road Accident Severity Prediction")
st.caption("Model: XGBoost | Dataset: US Road Accidents")

@st.cache_resource
def load_model():
    return joblib.load("models/binary_xgboost.joblib")

@st.cache_data
def load_sample():
    return pd.read_csv("data/processed/train_data_final.csv")

try:
    model = load_model()
    data = load_sample()

    st.success("XGBoost model loaded successfully!")

    st.subheader("Accident Input")
    st.write(
        "For this demonstration, select a real processed accident "
        "record from the training dataset."
    )

    row_number = st.number_input(
        "Select sample record",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1
    )

    if st.button("Predict Accident Severity", type="primary"):
        features = model.feature_names_in_
        X = data.loc[[int(row_number)], features]

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Severity Accident")
        else:
            st.success("✅ Low Severity Accident")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Low Severity Probability",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "High Severity Probability",
                f"{probabilities[1] * 100:.2f}%"
            )

        st.subheader("Input Record")
        st.dataframe(X, use_container_width=True)

except Exception as e:
    st.error(f"Application error: {e}")
