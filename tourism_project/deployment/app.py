import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Product Prediction App")
st.write("""
This application predicts the likelihood of a Tourism Product being accepted by Customers.
Enter the parameters below to get a prediction.
""")

age                       = st.number_input("Age", 18, 61, 18, 1)
TypeofContact             = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])  
CityTier                  = st.selectbox("CityTier", [1,2,3])  
DurationOfPitch           = st.number_input("Duration Of Pitch")
Occupation                = st.selectbox("Occupation", ["Free Lancer",	"Large Business",	"Salaried",	"Small Business"])  
Gender                    = st.selectbox("Gender", ["Male","Female"])
NumberOfPersonVisiting    = st.number_input("Number Of Person Visiting", 1, 5, 1, 1)
NumberOfFollowups         = st.number_input("Number Of Follow ups", 1, 6, 1, 1)
ProductPitched            = st.selectbox("Product Pitched", ["Basic","Deluxe","King","Standard","Super Deluxe"])
PreferredPropertyStar     = st.selectbox("Preferred Property Star", [3,4,5])  
MaritalStatus             = st.selectbox("MaritalStatus", ["Single","Married","Divorced"])
NumberOfTrips             = st.number_input("Number Of Trips")
Passport                  = st.selectbox("Passport", [0,1])  
PitchSatisfactionScore    = st.selectbox("Pitch Satisfaction Score", [1,2,3,4,5])  
OwnCar                    = st.selectbox("Own Car", [0,1])  
NumberOfChildrenVisiting  = st.number_input("Number Of Children Visiting", 0, 3, 0, 1)
Designation               = st.selectbox("Designation", ["AVP","Executive","Manager","Senior Manager","VP"])
MonthlyIncome             = st.number_input("Monthly Income")

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
    
}])

if st.button("Predict Tourism Product"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Product Taken" if prediction == 1 else "Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
