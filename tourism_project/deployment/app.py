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
This application predicts the likelihood of a customer accepting a wellness tourism package based on their demographic and behavioral characteristics.
Please enter the customer information below to get a prediction.
""")

# Create two columns for better organization
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Details")

    # Personal Information
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    # Professional Information
    occupation = st.selectbox("Occupation", ["Free Lancer","Large Business", "Salaried", "Small Business"])
    designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)

    # Location & Assets
    city_tier_display = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    city_tier = int(city_tier_display.split()[-1])  # Extract numeric value (1, 2, or 3)
    own_car_display = st.selectbox("Own Car", ["Yes", "No"])
    own_car = 1 if own_car_display == "Yes" else 0  # Convert to numeric (1 for Yes, 0 for No)
    passport_display = st.selectbox("Passport", ["Yes", "No"])
    passport = 1 if passport_display == "Yes" else 0  # Convert to numeric (1 for Yes, 0 for No)

    # Travel Group Details
    number_of_person_visiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)

    # Travel Preferences
    preferre_Property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
    number_of_trips = st.number_input("Previous Number of Trips", min_value=0, max_value=20, value=1)

with col2:
    st.subheader("📞 Customer Interaction Data")

    # Contact Information
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])

    # Product & Pitch Details
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)

    # Follow-up & Satisfaction
    number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
    pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

    # Add some spacing to align with the left column
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
  'Age': age,
  'TypeofContact': type_of_contact,
  'CityTier': city_tier,  # numeric (1, 2, or 3)
  'Occupation': occupation,
  'Gender': gender,
  'NumberOfPersonVisiting': number_of_person_visiting,
  'PreferredPropertyStar': preferre_Property_star,
  'MaritalStatus': marital_status,
  'NumberOfTrips': number_of_trips,
  'Passport': passport,  # numeric (1 for Yes, 0 for No)
  'OwnCar': own_car,  # numeric (1 for Yes, 0 for No)
  'NumberOfChildrenVisiting': number_of_children_visiting,
  'Designation': designation,
  'MonthlyIncome': monthly_income,
  'ProductPitched': product_pitched,
  'DurationOfPitch': duration_of_pitch,
  'NumberOfFollowups': number_of_followups,
  'PitchSatisfactionScore': pitch_satisfaction_score
}])

# Add a separator and prediction section
st.markdown("---")
st.subheader("🔮 Prediction")

if st.button("Predict Wellness Tourism Package", type="primary"):
    prediction = model.predict(input_data)[0]
    result = "Wellness Tourism Package accepted" if prediction == 1 else "Wellness Tourism Package not accepted"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
