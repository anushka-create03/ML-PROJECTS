import streamlit as st
import pandas as pd
import joblib


# Load the saved models and city summary file
investment_model = joblib.load("investment_model.pkl")
xgb_investment_model = joblib.load("xgb_investment_model.pkl")
price_model = joblib.load("price_model.pkl")
city_summary = pd.read_csv("city_summary.csv")


st.title("Real Estate Investment Advisor")
st.write("Enter property details to see the model estimates.")


# Property details
state = st.text_input("State", "Tamil Nadu")
city = st.text_input("City", "Chennai")
locality = st.text_input("Locality", "Locality_84")

property_type = st.selectbox(
    "Property type",
    ["Apartment", "Villa", "Independent House"]
)

bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=10, value=2)
size = st.number_input("Size (sq ft)", min_value=100, max_value=20000, value=1000)
price = st.number_input("Current price (lakhs)", min_value=1.0, value=100.0)
year_built = st.number_input("Year built", min_value=1900, max_value=2025, value=2010)

furnished = st.selectbox(
    "Furnished status",
    ["Unfurnished", "Semi-furnished", "Furnished"]
)

schools = st.number_input("Nearby schools (dataset value)", min_value=0, value=5)
hospitals = st.number_input("Nearby hospitals (dataset value)", min_value=0, value=5)

transport = st.selectbox(
    "Public transport access",
    ["Low", "Medium", "High"]
)

parking = st.selectbox("Parking", ["No", "Yes"])
security = st.selectbox("Security", ["No", "Yes"])
amenities = st.text_input("Amenities", "Gym, Garden")
facing = st.selectbox("Facing", ["North", "South", "East", "West"])
owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker"])

availability = st.selectbox(
    "Availability",
    ["Ready_to_Move", "Under_Construction", "Sold"]
)


# Generate estimates and charts after the button is clicked
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

    # Rule-based investment model prediction
    classification_input = pd.DataFrame([property_data])
    investment_result = investment_model.predict(classification_input)[0]

    # Regression model prediction
    regression_data = property_data.copy()
    regression_data["Price_in_Lakhs"] = price
    regression_input = pd.DataFrame([regression_data])
    future_price = price_model.predict(regression_input)[0]

    # Show estimates
    st.subheader("Estimates")

    if investment_result == 1:
        st.write("Good investment according to the project's rule.")
    else:
        st.write("Not marked as a good investment according to the project's rule.")

    st.metric("Regression model estimate after 5 years", f"₹{future_price:.2f} lakhs")

    st.caption(
        "The five-year estimate is based on the project's assumed 8% annual "
        "price increase. It is an illustrative scenario, not a guaranteed forecast."
    )

    # Illustrative five-year price chart
    st.subheader("Illustrative five-year price scenario")

    forecast_chart = pd.DataFrame(
        {
            "Estimated price (lakhs)": [
                price * (1.08 ** year) for year in range(6)
            ]
        },
        index=range(6),
    )
    forecast_chart.index.name = "Year"
    st.line_chart(forecast_chart)

    # XGBoost prediction and probability
    xgb_result = xgb_investment_model.predict(classification_input)[0]
    st.subheader("XGBoost investment classification")

    if xgb_result == 1:
        st.write("XGBoost classifies this property as a good investment.")
    else:
        st.write("XGBoost does not classify this property as a good investment.")

    if hasattr(xgb_investment_model, "predict_proba"):
        xgb_probability = xgb_investment_model.predict_proba(classification_input)[0][
            int(xgb_result)
        ]
        st.metric("XGBoost model score for its predicted label", f"{xgb_probability:.1%}")

    st.caption(
        "This score describes the model's prediction of the project's rule-based "
        "label. It is not the probability of making a real-world profit."
    )

    # XGBoost feature-importance chart
    xgb_classifier = xgb_investment_model.named_steps["classifier"]
    preprocessor = xgb_investment_model.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()
    importance_values = xgb_classifier.feature_importances_

    top_features = (
        pd.Series(importance_values, index=feature_names)
        .nlargest(10)
        .sort_values()
        .to_frame(name="Importance")
    )

    st.subheader("XGBoost feature importance")
    st.bar_chart(top_features)


# City comparison chart (shown even before clicking Get estimate)
st.subheader("City comparison")

chart_options = {
    "Median property price (lakhs)": "Median_Price_Lakhs",
    "Average price per sq ft": "Average_Price_Per_SqFt",
}

chosen_label = st.selectbox("Compare cities by", list(chart_options.keys()))
chosen_column = chart_options[chosen_label]

top_cities = (
    city_summary.sort_values(chosen_column, ascending=False)
    .head(10)
    .set_index("City")[[chosen_column]]
)

st.bar_chart(top_cities)