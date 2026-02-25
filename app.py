import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("🫀 AI Heart Disease Prediction System")
st.write("Enter patient details below:")

age = st.slider("Age", 20, 80, 40)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol Level", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])

sex = 1 if sex == "Male" else 0

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Heart Disease ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({(1-probability)*100:.2f}%)")