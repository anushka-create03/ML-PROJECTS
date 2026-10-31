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

    if hasattr(investment_model, "predict_proba"):
        probability = investment_model.predict_proba(classification_input)[0][
            int(investment_result)
        ]
        st.metric("Model probability for this label", f"{probability:.1%}")
        st.caption(
            "This is the model's score for its rule-based label, "
            "not the chance of making a profit."
        )

    classifier = investment_model.named_steps["classifier"]
    preprocessor = investment_model.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        chart_title = "Top feature importances"
    elif hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_[0])
        chart_title = "Largest coefficient magnitudes"
    else:
        importance_values = None

    if importance_values is not None:
        top_features = (
            pd.Series(importance_values, index=feature_names)
            .nlargest(10)
            .sort_values()
            .to_frame(name="Score")
        )
        st.subheader(chart_title)
        st.bar_chart(top_features)

# City chart, shown outside the estimate button
city_summary = pd.read_csv("city_summary.csv")
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

st.subheader("City comparison")
st.bar_chart(top_cities)
investment_model = joblib.load("investment_model.pkl")
price_model = joblib.load("price_model.pkl")
xgb_investment_model = joblib.load("xgb_investment_model.pkl")
classifier = xgb_investment_model.named_steps["classifier"]
preprocessor = xgb_investment_model.named_steps["preprocessing"]
feature_names = preprocessor.get_feature_names_out()

importance_values = classifier.feature_importances_
chart_title = "XGBoost feature importance"