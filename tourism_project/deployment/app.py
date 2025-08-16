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
type_of_contact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("CityTier", ["Tier 1", "Tier 2", "Tier 3"])
cccupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business Owner"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=1)
preferre_Property_star = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=1)
marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("NumberOfTrips", min_value=1, max_value=10, value=1)
passport = st.selectbox("Passport", ["Yes", "No"])
own_car = st.selectbox("OwnCar", ["Yes", "No"])
number_of_children_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
monthly_income = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=0)
product_pitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "King", "Standard","Super Deluxe"])
duration_of_pitch = st.number_input("DurationOfPitch", min_value=1, max_value=100, value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=0)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=1)  

# Assemble input into DataFrame
input_data = pd.DataFrame([{
  'Age': age,
  'Type of Contact': type_of_contact,
  'City Tier': city_tier,
  'Occupation': cccupation,
  'Gender': gender,
  'Number of Person Visiting': number_of_person_visiting,
  'Preferred Property Star': preferre_Property_star,
  'Marital Status': marital_status,
  'Number of Trips': number_of_trips,
  'Passport': passport,
  'Own Car': own_car,
  'Number of Children Visiting': number_of_children_visiting,
  'Designation': designation,
  'MonthlyIncome': monthly_income,
  'Product Pitched': product_pitched,
  'Duration of Pitch': duration_of_pitch,
  'Number of Followups': NumberOfFollowups,
  'Pitch Satisfaction Score': PitchSatisfactionScore                                 
}])


if st.button("Predict Wellness Tourim Package"):
    prediction = model.predict(input_data)[0]
    result = "Wellness Tourim Package accepted" if prediction == 1 else "not Accepted"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
