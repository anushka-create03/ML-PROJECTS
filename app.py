import streamlit as st
import pandas as pd
import joblib

investment_model = joblib.load("investment_model.pkl")
price_model = joblib.load("price_model.pkl")

st.title("Real Estate Investment Advisor")
st.write("Enter property details to see the model estimates.")

state = st.text_input("State", "Tamil Nadu")
city = st.text_input("City", "Chennai")
locality = st.text_input("Locality", "Locality_84")
property_type = st.selectbox(
    "Property type", ["Apartment", "Villa", "Independent House"]
)
bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=10, value=2)
size = st.number_input("Size (sq ft)", min_value=100, max_value=20000, value=1000)
price = st.number_input("Current price (lakhs)", min_value=1.0, value=100.0)
year_built = st.number_input(
    "Year built", min_value=1900, max_value=2025, value=2010
)
furnished = st.selectbox(
    "Furnished status", ["Unfurnished", "Semi-furnished", "Furnished"]
)
schools = st.number_input("Nearby schools (dataset value)", min_value=0, value=5)
hospitals = st.number_input("Nearby hospitals (dataset value)", min_value=0, value=5)
transport = st.selectbox(
    "Public transport access", ["Low", "Medium", "High"]
)
parking = st.selectbox("Parking", ["No", "Yes"])
security = st.selectbox("Security", ["No", "Yes"])
amenities = st.text_input("Amenities", "Gym, Garden")
facing = st.selectbox("Facing", ["North", "South", "East", "West"])
owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker"])
availability = st.selectbox(
    "Availability", ["Ready_to_Move", "Under_Construction", "Sold"]
)

if st.button("Get estimate"):
    age = 2025 - year_built

    property_data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner,
        "Availability_Status": availability,
    }

    classification_input = pd.DataFrame([property_data])
    investment_result = investment_model.predict(classification_input)[0]

    regression_data = property_data.copy()
    regression_data["Price_in_Lakhs"] = price
    regression_input = pd.DataFrame([regression_data])
    future_price = price_model.predict(regression_input)[0]

    st.subheader("Estimates")
    st.write(
        "Good investment"
        if investment_result == 1
        else "Not marked as a good investment"
    )
    st.write(f"Estimated price after 5 years: ₹{future_price:.2f} lakhs")
    st.caption(
        "These are project estimates based on the dataset's rules, "
        "including an assumed 8% annual price increase."
    )
