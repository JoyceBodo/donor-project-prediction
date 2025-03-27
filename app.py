import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("models/model.pkl")  

# Title
st.title("Donor-Funded Project Success Prediction")

# Sidebar Inputs
st.sidebar.header("Enter Project Details")
funding_amount = st.sidebar.number_input("Total Funding (KES)", min_value=0)
duration_months = st.sidebar.number_input("Project Duration (Months)", min_value=1)
funding_type = st.sidebar.selectbox("Funding Type", ["Grant", "Loan", "Equity", "Other"])
mtef_sector = st.sidebar.selectbox("Sector", ["Health", "Education", "Infrastructure", "Agriculture", "Other"])

# Encode categorical inputs
funding_type_encoded = {"Grant": 0, "Loan": 1, "Equity": 2, "Other": 3}[funding_type]
sector_encoded = {"Health": 0, "Education": 1, "Infrastructure": 2, "Agriculture": 3, "Other": 4}[mtef_sector]

# Prepare input features
input_data = np.array([[funding_amount, duration_months, funding_type_encoded, sector_encoded]])

# Prediction Button
if st.sidebar.button("Predict Success"):
    prediction = model.predict(input_data)
    st.write("### Prediction:")
    st.success("✅ Project Likely to Succeed" if prediction == 1 else "❌ Project Likely to Fail")
