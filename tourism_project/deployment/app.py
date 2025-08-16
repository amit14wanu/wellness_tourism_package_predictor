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
age = st.number_input("age (min)", min_value=18, max_value=100, value=18)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
cccupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business Owner"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=1)
preferre_Property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=1)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=10, value=1)
passport = st.selectbox("Passport", ["Yes", "No"])
own_car = st.selectbox("Own Car", ["Yes", "No"])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=0)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard","Super Deluxe"])
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1, max_value=100, value=1)
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1)  

# Assemble input into DataFrame
input_data = pd.DataFrame([{
  'Age': age,
  'TypeOfContact': type_of_contact,
  'CityTier': city_tier,
  'Occupation': cccupation,
  'Gender': gender,
  'NumberOfPersonVisiting': number_of_person_visiting,
  'PreferredPropertyStar': preferre_Property_star,
  'MaritalStatus': marital_status,
  'NumberOfTrips': number_of_trips,
  'Passport': passport,
  'OwnCar': own_car,
  'NumberOfChildrenVisiting': number_of_children_visiting,
  'Designation': designation,
  'MonthlyIncome': monthly_income,
  'ProductPitched': product_pitched,
  'DurationOfPitch': duration_of_pitch,
  'NumberOfFollowups': number_of_followups,
  'PitchSatisfactionScore': pitch_satisfaction_score                                 
}])


if st.button("Predict Wellness Tourim Package"):
    prediction = model.predict(input_data)[0]
    result = "Wellness Tourim Package accepted" if prediction == 1 else "not Accepted"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
