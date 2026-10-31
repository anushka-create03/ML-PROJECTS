# Real Estate Investment Advisor

## Project goal
This project explores property data and creates a Streamlit app that estimates whether a property meets a simple investment rule and calculates an estimated price after five years.

## Dataset
- 250,000 rows and 23 columns
- No missing values or exact duplicate rows
- `Age_of_Property` matches `2025 - Year_Built` for every row
- `Floor_No` is greater than `Total_Floors` in 116,304 rows; these values were flagged for review and not corrected
- The price and size charts looked unusually uniform
- Size and price correlation: -0.003

## Methods
- Created a working copy of the data and flagged inconsistent floor records
- Defined “Good Investment” as a calculated price per square foot at or below the dataset median
- Estimated five-year price using an assumed 8% annual increase
- Trained a logistic regression classifier and a linear regression model
- Evaluated classification with accuracy, precision, recall, and F1 score
- Evaluated regression with MAE, RMSE, and R²
- Logged parameters, metrics, and model files in MLflow
- Created a Streamlit app for entering property details and viewing estimates

## Important limitations
The dataset has signs of being synthetic or randomly generated. The future-price target was calculated from current price using the 8% assumption, so the perfect regression score shows that the model reproduced this formula; it does not validate real market forecasts. The investment label is a simple price-per-square-foot rule, not proof of profitability. Floor fields were inconsistent and excluded from the models.

## Results
- Classification accuracy: 0.712
- Classification precision: 0.71
- Classification recall: 0.71
- Classification F1 score: 0.71
- Regression MAE: 0.00 lakhs
- Regression RMSE: 0.00 lakhs
- Regression R²: 1.00

## How to run the app
Install the packages listed in `requirements.txt`, then run:

    python -m streamlit run app.py