
# app.py
import streamlit as st
import numpy as np
import joblib

st.title("Diabetes Prediction App")

try:
    model = joblib.load(open('diabetes_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    data = np.array([[glucose, bp, bmi, age]])
    pred = model.predict(data)
    if pred[0] == 1:
        st.error("⚠️ You may have diabetes.")
    else:
        st.success("✅ You are unlikely to have diabetes.")
