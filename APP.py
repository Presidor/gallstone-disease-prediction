import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Gallstone Prediction App")

st.title("Gallstone Disease Prediction System")
st.write("Enter patient clinical details to predict gallstone risk")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("gallstone_prediction_model.pkl")

# load scaler if used
# scaler = joblib.load("scaler.pkl")

# -----------------------------
# USER INPUTS (EDIT BASED ON YOUR FEATURES)
# -----------------------------

age = st.number_input("Age", 1, 100)
bmi = st.number_input("BMI")
cholesterol = st.number_input("Cholesterol Level")
glucose = st.number_input("Glucose Level")

# Add ALL features used in your notebook

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict"):

    data = pd.DataFrame({
        "Age":[age],
        "BMI":[bmi],
        "Cholesterol":[cholesterol],
        "Glucose":[glucose]
    })

    # apply scaling if used
    # data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High Risk of Gallstone")
    else:
        st.success("Low Risk of Gallstone")