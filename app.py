import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("Predict your risk of diabetes based on health indicators.")

# Load model
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload 'diabetes_model.pkl'.")
    st.stop()

# Layout: Left - Input | Right - Output
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔎 Enter Your Info")
    name = st.text_input("👤 Name", placeholder="e.g. Alice")
    gender = st.radio("⚧️ Gender", ["Male", "Female"], horizontal=True)
    age = st.slider("🎂 Age", 1, 120, 30)
    glucose = st.slider("🍬 Glucose Level", 0, 200, 120)
    bp = st.slider("💓 Blood Pressure", 0, 150, 70)
    height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=160)
    weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=60)

    bmi = weight / ((height / 100) ** 2)
    st.write(f"📊 **Calculated BMI:** `{bmi:.2f}`")

    if st.button("📈 Predict Diabetes Risk"):
        if not name:
            st.warning("🚨 Please enter your name to continue.")
        else:
            input_data = np.array([[glucose, bp, bmi, age]])
            prediction = model.predict(input_data)
            result_text = "⚠️ Likely to Have Diabetes" if prediction[0] == 1 else "✅ Unlikely to Have Diabetes"

            with col2:
                st.header("📋 Prediction Report")
                st.success(f"👤 Name: **{name}**")
                st.write(f"🔹 Gender: `{gender}`")
                st.write(f"🔹 Age: `{age}`")
                st.write(f"🔹 Glucose: `{glucose}`")
                st.write(f"🔹 Blood Pressure: `{bp}`")
                st.write(f"🔹 BMI: `{bmi:.2f}`")

                # Result box
                if prediction[0] == 1:
                    st.error(result_text)
                else:
                    st.success(result_text)

                # Graph: Pie chart
                labels = ['No Diabetes', 'Diabetes']
                sizes = [1, 0] if prediction[0] == 0 else [0, 1]
                colors = ['#4CAF50', '#FF5252']
                fig, ax = plt.subplots()
                ax.pie([0.01, 0.99] if prediction[0] else [0.99, 0.01], labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
