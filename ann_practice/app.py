import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==================================
# LOAD MODEL
# ==================================

model = tf.keras.models.load_model(
    "models/ann_model.keras"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

training_columns = joblib.load(
    "models/training_columns.pkl"
)

# ==================================
# SIDEBAR
# ==================================

with st.sidebar:

    st.title("ℹ️ Model Info")

    st.markdown("---")

    st.write("Model: ANN")

    st.write(
        "Predict whether a customer "
        "will leave the bank."
    )

# ==================================
# TITLE
# ==================================

st.title(
    "🏦 Bank Customer Churn Prediction"
)

st.markdown("---")

# ==================================
# INPUTS
# ==================================

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        300,
        900,
        650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        18,
        100,
        35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

with col2:

    balance = st.number_input(
        "Balance",
        value=50000.0
    )

    products = st.selectbox(
        "Number Of Products",
        [1, 2, 3, 4]
    )

    card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    active = st.selectbox(
        "Active Member",
        ["Yes", "No"]
    )

    salary = st.number_input(
        "Estimated Salary",
        value=50000.0
    )

# ==================================
# PREDICTION
# ==================================

if st.button(
    "🚀 Predict Churn"
):

    final_df = pd.DataFrame(
        0,
        index=[0],
        columns=training_columns
    )

    final_df["CreditScore"] = credit_score
    final_df["Age"] = age
    final_df["Tenure"] = tenure
    final_df["Balance"] = balance
    final_df["NumOfProducts"] = products
    final_df["HasCrCard"] = (
        1 if card == "Yes" else 0
    )
    final_df["IsActiveMember"] = (
        1 if active == "Yes" else 0
    )
    final_df["EstimatedSalary"] = salary

    if "Geography_Germany" in final_df.columns:
        final_df["Geography_Germany"] = (
            1 if geography == "Germany"
            else 0
        )

    if "Geography_Spain" in final_df.columns:
        final_df["Geography_Spain"] = (
            1 if geography == "Spain"
            else 0
        )

    if "Gender_Male" in final_df.columns:
        final_df["Gender_Male"] = (
            1 if gender == "Male"
            else 0
        )

    scaled = scaler.transform(
        final_df
    )

    prediction = model.predict(
        scaled,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    st.markdown("---")

    if probability > 0.5:
        st.error(
            f"Customer likely to leave ({probability:.2%})"
        )
    else:
        st.success(
            f"Customer likely to stay ({1-probability:.2%})"
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={
                "text":
                "Churn Probability (%)"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                }
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )