import streamlit as st
import pickle
import numpy as np

# Title
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title('🩺 Diabetes Prediction App')

# Description
st.write("""
This app uses an AI model to predict whether you are likely to have **diabetes**  
based on your **Glucose**, **Blood Pressure**, **BMI** (auto-calculated), and **Age**.
""")

# Load model
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ diabetes_model.pkl not found. Please place the file in the same folder as this app.")
    st.stop()

# User Inputs
st.subheader("🧍 Personal Information")
name = st.text_input("Name", placeholder="e.g. John Doe")
gender = st.selectbox("Gender", ["Male", "Female"])

st.subheader("🩸 Health Information")
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
bp = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
height = st.number_input('Height (cm)', min_value=100, max_value=250, value=160)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=60)

# Auto-calculate BMI
bmi = weight / ((height / 100) ** 2)
st.write(f"📏 **Calculated BMI:** {bmi:.2f}")

age = st.number_input('Age', min_value=1, max_value=120, value=30)

# Predict button
if st.button('🔍 Predict'):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)

    st.subheader("🧾 Prediction Report")
    st.write(f"👤 Name: **{name or 'N/A'}**")
    st.write(f"⚧ Gender: **{gender}**")
    st.write(f"📊 Glucose: **{glucose}**, Blood Pressure: **{bp}**, BMI: **{bmi:.2f}**, Age: **{age}**")

    if prediction[0] == 1:
        st.error("⚠️ The model predicts: You may have diabetes.")
    else:
        st.success("✅ The model predicts: You are unlikely to have diabetes.")

