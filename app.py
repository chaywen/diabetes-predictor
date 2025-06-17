
# app.py

import streamlit as st
import pickle
import numpy as np

# Title
st.title('Diabetes Prediction App')

# Description
st.write("""
This app uses an AI model to predict whether you are likely to have diabetes  
based on your **Glucose**, **Blood Pressure**, **BMI**, and **Age**.
""")

# Load model
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ diabetes_model.pkl not found. Please place the file in the same folder as this app.")
    st.stop()

# Input fields
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
bp = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
age = st.number_input('Age', min_value=1, max_value=120, value=30)

# Predict button
if st.button('Predict'):
    input_data = np.array([[glucose, bp, bmi, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ The model predicts: You may have diabetes.")
    else:
        st.success("✅ The model predicts: You are unlikely to have diabetes.")
