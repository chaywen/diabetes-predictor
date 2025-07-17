import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime

# App title
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.markdown("""
This app uses an AI model to predict whether you are likely to have diabetes 
using your **Glucose**, **Blood Pressure**, **BMI**, and **Age**.
""")

# Load the model
try:
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ diabetes_model.pkl not found. Please upload it.")
    st.stop()

# Sidebar for user info
with st.sidebar:
    st.header("🧍 User Info")
    name = st.text_input("Name", "John Doe")
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    today = datetime.date.today()
    prediction_date = st.date_input("Date", today)

# Input fields
st.subheader("📝 Enter Your Health Data")
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)

    st.subheader("📋 Prediction Report")
    result_df = pd.DataFrame({
        "Name": [name],
        "Gender": [gender],
        "Date": [prediction_date.strftime("%Y-%m-%d")],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "BMI": [bmi],
        "Age": [age],
        "Prediction": ["Positive" if prediction[0] == 1 else "Negative"]
    })
    st.dataframe(result_df, use_container_width=True)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts: You may have diabetes.")
        st.markdown("""
        ### 🧠 Health Advice:
        - Please consult a healthcare professional for further diagnosis.
        - Adopt a healthy diet low in sugar.
        - Stay physically active and monitor your blood glucose levels.
        """)
    else:
        st.success("✅ The model predicts: You are unlikely to have diabetes.")
        st.markdown("""
        ### 🧘 Keep It Up!
        - Continue a balanced diet.
        - Maintain a regular exercise routine.
        - Schedule regular checkups with your doctor.
        """)

    # CSV download
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Prediction Report (CSV)", data=csv, file_name="diabetes_report.csv", mime="text/csv")

# Footer
st.markdown("""
---
Developed by [Your Name] | Powered by Streamlit & Scikit-learn
""")
