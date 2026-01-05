
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- Page config (MUST be first Streamlit call) ----------------
st.set_page_config(
    page_title="Vehicle Insurance Claim Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Insurance Claim Prediction")
st.write("Predict whether an insurance claim will be filed")

# ---------------- Load model & encoders ----------------
try:
    model = joblib.load("vehicle_insurance_rf.pkl")
    vehicle_encoder = joblib.load("vehicle_encoder.pkl")
    policy_encoder = joblib.load("policy_encoder.pkl")
    st.success("✅ Model & Encoders Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# ---------------- User Inputs ----------------
st.subheader("Enter Vehicle & Driver Details")

driver_age = st.number_input("Driver Age", min_value=18, max_value=80, value=30)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=20, value=5)
annual_mileage = st.number_input("Annual Mileage (km)", min_value=1000, max_value=50000, value=12000)
accident_history = st.slider("Number of Past Accidents", min_value=0, max_value=10, value=1)

vehicle_type = st.selectbox("Vehicle Type", ["Car", "Bike", "Truck"])
policy_type = st.selectbox("Policy Type", ["Basic", "Premium"])

# ---------------- Encode categorical inputs (LabelEncoder) ----------------
vehicle_encoded = vehicle_encoder.transform([vehicle_type])[0]
policy_encoded = policy_encoder.transform([policy_type])[0]

# ---------------- Prediction ----------------
if st.button("Predict Claim"):
    input_df = pd.DataFrame([{
        "Driver_Age": driver_age,
        "Vehicle_Age": vehicle_age,
        "Annual_Mileage": annual_mileage,
        "Accident_History": accident_history,
        "Vehicle_Type": vehicle_encoded,
        "Policy_Type": policy_encoded
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Insurance Claim Likely")
    else:
        st.success("✅ No Insurance Claim Expected")
