import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="amit14official/tourism_project_model", filename="best_tourism_project_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Product Prediction
st.title("Tourism Product Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
age = st.number_input("age (max)", min_value=18, max_value=100, value=100)
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("CityTier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business Owner"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=1)
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=10, value=1)
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("OwnCar", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
  'Age': age                                 
}])


if st.button("Predict Tourim Package"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
