import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Page Config
st.set_page_config(page_title="🩺 Diabetes Predictor", layout="centered")

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app uses an AI model to predict whether you are likely to have diabetes
based on your **Glucose**, **Blood Pressure**, **BMI**, and **Age**.
""")

# Load model
try:
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ diabetes_model.pkl not found. Please place the file in the same folder as this app.")
    st.stop()

# Section: User Inputs
st.header("🔎 Enter Your Info")
name = st.text_input("👤 Name", placeholder="e.g. Alice")
gender = st.radio("⚧️ Gender", ["Male", "Female"], horizontal=True)
age = st.slider("🎂 Age", 1, 120, 30)
glucose = st.slider("🍬 Glucose Level", 0, 200, 120)
bp = st.slider("💓 Blood Pressure", 0, 150, 70)
height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=160)
weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=60)
date = st.date_input("📅 Date", datetime.date.today())

# BMI Calculation
def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

bmi = calculate_bmi(weight, height)
st.info(f"📌 Calculated BMI: {bmi}")

# Predict button
if st.button("🚀 Predict Now"):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)

    st.subheader("📋 Prediction Summary")
    result = "Positive" if prediction[0] == 1 else "Negative"
    report = pd.DataFrame({
        "Name": [name],
        "Gender": [gender],
        "Date": [date.strftime("%Y-%m-%d")],
        "Glucose": [glucose],
        "Blood Pressure": [bp],
        "BMI": [bmi],
        "Age": [age],
        "Prediction": [result]
    })
    st.dataframe(report, use_container_width=True)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts: You may have diabetes.")
        st.markdown("""
        ### 🩺 Health Suggestions
        - Consult a healthcare professional.
        - Maintain a healthy diet & active lifestyle.
        - Monitor blood glucose & blood pressure regularly.
        """)
    else:
        st.success("✅ The model predicts: You are unlikely to have diabetes.")
        st.markdown("""
        ### ✅ Keep It Up!
        - Stay active and eat balanced meals.
        - Continue regular health checkups.
        """)

    # Visualization
    st.subheader("📊 Visual Overview")
    chart_data = pd.DataFrame({
        "Indicator": ["Glucose", "Blood Pressure", "BMI"],
        "Value": [glucose, bp, bmi]
    })
    fig, ax = plt.subplots()
    sns.barplot(data=chart_data, x="Indicator", y="Value", ax=ax, palette="Set2")
    ax.set_title("Health Metrics")
    st.pyplot(fig)

    # Download report
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Report",
        data=csv,
        file_name="diabetes_prediction_report.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit  
AI Model powered by scikit-learn
""")
